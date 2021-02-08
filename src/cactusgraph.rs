use log::{debug, trace};
use petgraph::prelude::*;
use rayon::prelude::*;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    biedgedgraph::{BiedgedGraph, BiedgedWeight},
    netgraph::NetGraph,
    projection::{end_to_black_edge, opposite_vertex, Projection},
    ultrabubble::{BridgePair, ChainPair},
};

#[cfg(feature = "progress_bars")]
use indicatif::ParallelProgressIterator;

#[cfg(feature = "progress_bars")]
fn progress_bar(len: usize, steady: bool) -> indicatif::ProgressBar {
    use indicatif::{ProgressBar, ProgressStyle};
    let len = len as u64;
    let p_bar = ProgressBar::new(len);
    p_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos:>12}/{len:12}")
            .progress_chars("##-"),
    );
    if steady {
        p_bar.enable_steady_tick(1000);
    }
    p_bar
}

macro_rules! impl_biedged_wrapper {
    ($for:ty) => {
        impl<'a> BiedgedWrapper for $for {
            #[inline]
            fn base_graph(&self) -> &UnGraphMap<u64, BiedgedWeight> {
                &self.graph.graph
            }

            #[inline]
            fn biedged_graph(&self) -> &BiedgedGraph {
                &self.graph
            }

            #[inline]
            fn projection(&self) -> &Projection {
                &self.projection
            }
        }
    };
}

/// Convenience trait for providing a unified interface when accessing
/// the underlying graph structure across the various graph types.
pub trait BiedgedWrapper {
    fn base_graph(&self) -> &UnGraphMap<u64, BiedgedWeight>;

    fn biedged_graph(&self) -> &BiedgedGraph;

    fn projection(&self) -> &Projection;

    #[inline]
    fn projected_node(&self, n: u64) -> u64 {
        let graph = self.biedged_graph();
        let proj = self.projection();
        graph.projected_node(proj, n)
    }

    #[inline]
    fn projected_edge(&self, (x, y): (u64, u64)) -> (u64, u64) {
        let proj = self.projection();
        proj.find_edge(x, y)
    }
}

/// A cactus graph constructed from a biedged graph. The constructor
/// clones the original graph to mutate, but also keeps a reference to
/// original. This ensures the original can't be accidentally mutated
/// while the CactusGraph exists, and makes it easy to access the
/// untouched original graph. The mapping of vertices is tracked using
/// the embedded `Projection` struct.
#[derive(Clone)]
pub struct CactusGraph<'a> {
    pub original_graph: &'a BiedgedGraph,
    pub graph: BiedgedGraph,
    pub projection: Projection,
    pub cycles: Vec<Vec<(u64, u64)>>,
    pub cycle_map: FxHashMap<(u64, u64), Vec<usize>>,
}

impl_biedged_wrapper!(CactusGraph<'a>);

impl<'a> CactusGraph<'a> {
    /// Construct a cactus graph from a biedged graph. Clones the
    /// input graph before mutating, and keeps a reference to the
    /// original.
    pub fn from_biedged_graph(biedged_graph: &'a BiedgedGraph) -> Self {
        debug!("  ~~~  building cactus graph  ~~~");
        debug!("cloning biedged graph");
        let t = std::time::Instant::now();
        // let mut graph = biedged_graph.clone();

        let mut graph = biedged_graph.shrink_clone();
        debug!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

        let (node_count, node_cap) = graph.node_count_capacity();
        let (edge_count, edge_cap) = graph.edge_count_capacity();

        trace!(" | cactus, start, nodes | {} | {} |", node_count, node_cap);
        trace!(" | cactus, start, edges | {} | {} |", edge_count, edge_cap);

        trace!("cloning projection");
        let t = std::time::Instant::now();
        let mut projection = Projection::new_for_biedged_graph(&graph);
        trace!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

        debug!("contracting gray edges");
        let t = std::time::Instant::now();
        Self::contract_all_gray_edges(&mut graph, &mut projection);
        debug!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

        debug!("finding 3-edge-connected components");
        let t = std::time::Instant::now();
        let components = Self::find_3_edge_connected_components(&graph);
        debug!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

        debug!("merging 3-edge-connected components");
        let t = std::time::Instant::now();
        Self::merge_components(&mut graph, components, &mut projection);
        debug!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

        graph.shrink_to_fit();

        let (node_count, node_cap) = graph.node_count_capacity();
        let (edge_count, edge_cap) = graph.edge_count_capacity();

        trace!(
            " | cactus, start, nodes, end | {} | {} |",
            node_count,
            node_cap
        );
        trace!(
            " | cactus, start, edges, end | {} | {} |",
            edge_count,
            edge_cap
        );

        debug!("finding cycles");
        let t = std::time::Instant::now();
        let mut cycles = Self::find_cycles(&graph);
        trace!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

        let mut cycle_map: FxHashMap<(u64, u64), Vec<usize>> =
            FxHashMap::default();

        trace!("building cycle map using {} cycles", cycles.len());
        let t = std::time::Instant::now();

        let mut total_vals = 0;
        let mut total_cap = 0;

        for (i, cycle) in cycles.iter_mut().enumerate() {
            total_vals += cycle.len();
            total_cap += cycle.capacity();
            for &(a, b) in cycle.iter() {
                let l = a.min(b);
                let r = a.max(b);
                cycle_map.entry((l, r)).or_default().push(i);
            }
            cycle.shrink_to_fit();
        }

        cycles.shrink_to_fit();

        trace!(
            "| cactus, cycles, outer | {} | {} |",
            cycles.len(),
            cycles.capacity()
        );
        trace!(
            " | cactus, cycles, inner | {} | {} |",
            total_vals,
            total_cap
        );

        trace!(
            "| cactus, cycle_map, outer | {} | {} |",
            cycle_map.len(),
            cycle_map.capacity()
        );

        let mut total_vals = 0;
        let mut total_cap = 0;

        for cycle_vec in cycle_map.values_mut() {
            cycle_vec.shrink_to_fit();
        }
        cycle_map.shrink_to_fit();

        for v in cycle_map.values() {
            total_vals += v.len();
            total_cap += v.capacity();
        }

        trace!(
            "| cactus, cycle_map, inner | {} | {} |",
            total_vals,
            total_cap
        );

        trace!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

        trace!("building inverse projection map");
        let t = std::time::Instant::now();
        projection.build_inverse();
        trace!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);
        debug!("  ~~~  cactus graph constructed  ~~~");

        CactusGraph {
            original_graph: biedged_graph,
            graph,
            projection,
            cycles,
            cycle_map,
        }
    }

    pub fn contract_all_gray_edges(
        biedged: &mut BiedgedGraph,
        projection: &mut Projection,
    ) {
        let _p_bar;

        #[cfg(not(feature = "progress_bars"))]
        {
            _p_bar = ();
        }

        trace!("calculating gray edge count");
        let t = std::time::Instant::now();
        let gray_edge_count = biedged.gray_edge_count();
        trace!(
            " gray edge count took {:.3} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
        trace!("contracting {} gray edges", gray_edge_count);

        #[cfg(feature = "progress_bars")]
        {
            use indicatif::{ProgressBar, ProgressStyle};
            _p_bar = ProgressBar::new(gray_edge_count as u64);
            _p_bar.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40} {pos:>10}/{len:10}")
                    .progress_chars("##-"),
            );
        }

        trace!("collecting gray edges");
        let t = std::time::Instant::now();
        let gray_edges = biedged
            .gray_edges()
            .map(|(a, b, _w)| (a, b))
            .collect::<Vec<_>>();
        trace!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);
        trace!("collected gray edges");

        trace!(
            "gray_edges len: {}, capacity: {}",
            gray_edges.len(),
            gray_edges.capacity()
        );

        for (from, to) in gray_edges {
            let from_ = projection.find(from);
            let to_ = projection.find(to);
            let edge = biedged.graph.edge_weight(from_, to_).copied();
            if let Some(w) = edge {
                if w.gray > 0 {
                    let _proj_from =
                        biedged.contract_edge(from_, to_, projection).unwrap();
                }
            }

            #[cfg(feature = "progress_bars")]
            {
                _p_bar.inc(1);
            }
        }

        #[cfg(feature = "progress_bars")]
        {
            _p_bar.finish();
        }
    }

    pub fn find_3_edge_connected_components(
        biedged: &BiedgedGraph,
    ) -> Vec<Vec<usize>> {
        let edges = biedged.graph.all_edges().flat_map(|(a, b, w)| {
            std::iter::repeat((a as usize, b as usize)).take(w.black)
        });

        let graph = three_edge_connected::Graph::from_edges(edges);

        let components = three_edge_connected::find_components(&graph.graph);
        // Many of the components returned by the algorithm can be singletons, which we don't need to do anything with, hence we filter them out.
        let components: Vec<_> =
            components.into_iter().filter(|c| c.len() > 1).collect();

        // The 3EC library maps the graph into node IDs starting from
        // zero; even if the input biedged graph also does so, it's
        // better to make sure the node IDs are mapped backed to their
        // input IDs.
        graph.invert_components(components)
    }

    pub fn merge_components(
        biedged: &mut BiedgedGraph,
        components: Vec<Vec<usize>>,
        projection: &mut Projection,
    ) {
        for comp in components {
            let mut iter = comp.into_iter();
            let head = iter.next().unwrap() as u64;
            for other in iter {
                let other = other as u64;
                if biedged.graph.contains_node(head)
                    && biedged.graph.contains_node(other)
                {
                    if biedged.graph.contains_edge(head, other) {
                        biedged.contract_edge(head, other, projection);
                    } else {
                        biedged.merge_vertices(head, other, projection);
                    }
                }
            }
        }
    }

    /// Find the simple cycles in a cactus graph and return them. A
    /// cycle is represented as a vector of vertices, with the same
    /// start and end vertex.
    fn find_cycles(biedged: &BiedgedGraph) -> Vec<Vec<(u64, u64)>> {
        let graph = &biedged.graph;

        let mut visited: FxHashSet<u64> = FxHashSet::default();
        let mut parents: FxHashMap<u64, u64> = FxHashMap::default();

        let mut stack: Vec<u64> = Vec::new();

        let mut cycles = Vec::new();
        let mut cycle_ends: Vec<(u64, u64)> = Vec::new();

        for node in graph.nodes() {
            if !visited.contains(&node) {
                stack.push(node);
                while let Some(current) = stack.pop() {
                    if !visited.contains(&current) {
                        visited.insert(current);
                        for (_, adj, weight) in graph.edges(current) {
                            if adj == current {
                                for _ in 0..weight.black {
                                    cycles.push(vec![(current, current)]);
                                }
                            } else if !visited.contains(&adj) {
                                if weight.black == 2 {
                                    cycles.push(vec![
                                        (current, adj),
                                        (adj, current),
                                    ]);
                                }
                                stack.push(adj);
                                parents.insert(adj, current);
                            } else if parents.get(&current) != Some(&adj) {
                                cycle_ends.push((adj, current));
                            }
                        }
                    }
                }
            }
        }

        for (start, end) in cycle_ends {
            let mut cycle: Vec<(u64, u64)> = vec![];
            let mut current = end;

            while current != start {
                if let Some(parent) = parents.get(&current) {
                    cycle.push((current, *parent));
                    current = *parent;
                }
            }

            cycle.push((start, end));
            cycles.push(cycle);
        }

        cycles
    }

    #[inline]
    fn black_edge_projection(&self, x: u64) -> (u64, u64) {
        let (left, right) = end_to_black_edge(x);
        let p_l = self.projection.find(left);
        let p_r = self.projection.find(right);
        let from = p_l.min(p_r);
        let to = p_l.max(p_r);
        (from, to)
    }

    /// Given a vertex ID in the original biedged graph, find the
    /// simple cycle its incident black edge maps to.
    #[inline]
    fn black_edge_cycle(&self, x: u64) -> Option<&Vec<usize>> {
        let edge = self.black_edge_projection(x);
        let cycles = self.cycle_map.get(&edge)?;
        Some(&cycles)
    }
}

/// A cactus tree derived from a cactus graph. Like the CactusGraph
/// struct, this clones the underlying graph before mutating it into a
/// cactus tree, and keeps a reference both to the original biedged
/// graph as well as the cactus graph. Because the cactus tree only
/// adds chain vertices, and only removes edges, no vertices, there's
/// no need to track vertex projections.
pub struct CactusTree<'a> {
    pub original_graph: &'a BiedgedGraph,
    pub cactus_graph: &'a CactusGraph<'a>,
    pub graph: BiedgedGraph,
    pub chain_vertices: FxHashSet<u64>,
    pub cycle_chain_map: FxHashMap<(u64, u64), u64>,
}

impl<'a> BiedgedWrapper for CactusTree<'a> {
    #[inline]
    fn base_graph(&self) -> &UnGraphMap<u64, BiedgedWeight> {
        &self.graph.graph
    }

    #[inline]
    fn biedged_graph(&self) -> &BiedgedGraph {
        &self.graph
    }

    #[inline]
    fn projection(&self) -> &Projection {
        &self.cactus_graph.projection
    }
}

impl<'a> CactusTree<'a> {
    pub fn from_cactus_graph(cactus_graph: &'a CactusGraph<'a>) -> Self {
        debug!("  ~~~  building cactus tree  ~~~");
        // let mut graph = cactus_graph.graph.clone();
        let mut graph = cactus_graph.graph.shrink_clone();

        let (node_count, node_cap) = graph.node_count_capacity();
        let (edge_count, edge_cap) = graph.edge_count_capacity();

        trace!(
            " | cactus tree, start, nodes | {} | {} |",
            node_count,
            node_cap
        );
        trace!(
            " | cactus tree, start, edges | {} | {} |",
            edge_count,
            edge_cap
        );

        let cycles = cactus_graph.cycles.clone();

        let mut total_vals = 0;
        let mut total_cap = 0;

        for cycle in cycles.iter() {
            total_vals += cycle.len();
            total_cap += cycle.capacity();
        }

        trace!(
            "| cactus tree, cycles, outer | {} | {} |",
            cycles.len(),
            cycles.capacity()
        );
        trace!(
            " | cactus tree, cycles, inner | {} | {} |",
            total_vals,
            total_cap
        );

        debug!("constructing chain vertices");
        let (mut cycle_chain_map, mut chain_vertices) =
            Self::construct_chain_vertices(&mut graph, &cycles);

        cycle_chain_map.shrink_to_fit();
        chain_vertices.shrink_to_fit();

        trace!(
            " | cactus tree, chain_map | {} | {} |",
            cycle_chain_map.len(),
            cycle_chain_map.capacity()
        );
        trace!(
            " | cactus tree, chain vertices | {} | {} |",
            chain_vertices.len(),
            chain_vertices.capacity()
        );

        graph.shrink_to_fit();

        let (node_count, node_cap) = graph.node_count_capacity();
        let (edge_count, edge_cap) = graph.edge_count_capacity();

        trace!(
            " | cactus tree, end, nodes | {} | {} |",
            node_count,
            node_cap
        );
        trace!(
            " | cactus tree, end, edges | {} | {} |",
            edge_count,
            edge_cap
        );

        debug!("  ~~~  cactus tree constructed  ~~~");

        Self {
            original_graph: cactus_graph.original_graph,
            graph,
            cycle_chain_map,
            chain_vertices,
            cactus_graph,
        }
    }

    /// Adds a chain vertex for each cycle, with edges to each of the
    /// elements in the cycle, and removes the edges within the cycle.
    /// Returns a vector of tuples, where the first element is the chain
    /// vertex ID in the graph, and the second element is the index of the
    /// corresponding cycle in the provided cycles vector.
    fn construct_chain_vertices(
        biedged: &mut BiedgedGraph,
        cycles: &[Vec<(u64, u64)>],
    ) -> (FxHashMap<(u64, u64), u64>, FxHashSet<u64>) {
        let mut cycle_chain_map = FxHashMap::default();
        let mut chain_vertices = FxHashSet::default();

        for cycle in cycles.iter() {
            let chain_vx = biedged.add_chain_vertex();

            for &(from, to) in cycle {
                let l = from.min(to);
                let r = from.max(to);
                cycle_chain_map.insert((l, r), chain_vx);
                biedged.add_edge(to, chain_vx, BiedgedWeight::black(1));
                biedged.remove_one_black_edge(from, to);
            }

            chain_vertices.insert(chain_vx);
        }

        (cycle_chain_map, chain_vertices)
    }

    /// Given a vertex in the original biedged graph, find the chain
    /// vertex its incident black edge projects to.
    #[inline]
    pub fn black_edge_chain_vertex(&self, x: u64) -> Option<u64> {
        let (from, to) = self.cactus_graph.black_edge_projection(x);
        let chain_vx = self.cycle_chain_map.get(&(from, to))?;
        Some(*chain_vx)
    }

    /// Find the chain pairs using the chain vertices in the cactus
    /// tree, and return them as a set of snarls.
    pub fn find_chain_pairs(&self) -> FxHashSet<ChainPair> {
        let mut chain_pairs: FxHashSet<ChainPair> = FxHashSet::default();

        let cactus_graph_inverse =
            self.cactus_graph.projection.get_inverse().unwrap();

        let _p_bar;

        #[cfg(not(feature = "progress_bars"))]
        {
            _p_bar = ();
        }

        #[cfg(feature = "progress_bars")]
        {
            use indicatif::{ProgressBar, ProgressStyle};
            _p_bar = ProgressBar::new(self.base_graph().node_count() as u64);
            _p_bar.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40} {pos:>10}/{len:10}")
                    .progress_chars("##-"),
            );
            _p_bar.enable_steady_tick(1000);
        }

        for n in self.base_graph().nodes() {
            if self.graph.is_net_vertex(n) {
                let b_ns = cactus_graph_inverse.get(&n).unwrap();
                for &a in b_ns.iter() {
                    for &b in b_ns.iter() {
                        if a != b && opposite_vertex(a) != b {
                            let c_a = self.cactus_graph.black_edge_cycle(a);
                            let c_b = self.cactus_graph.black_edge_cycle(b);
                            if c_a.is_some() && c_a == c_b {
                                let a_ = a.min(b);
                                let b_ = a.max(b);
                                chain_pairs.insert(ChainPair { x: a_, y: b_ });
                            }
                        }
                    }
                }
            }

            #[cfg(feature = "progress_bars")]
            {
                _p_bar.inc(1);
            }
        }

        #[cfg(feature = "progress_bars")]
        {
            _p_bar.finish();
        }

        chain_pairs.shrink_to_fit();

        trace!(
            "chain_pairs len: {}, capacity: {}",
            chain_pairs.len(),
            chain_pairs.capacity()
        );

        chain_pairs
    }

    /// Find the path between two vertices, used when constructing net
    /// graphs.
    fn snarl_cactus_tree_path(
        &self,
        projection: &Projection,
        x: u64,
        y: u64,
    ) -> Option<Vec<u64>> {
        let p_x = projection.find(x);
        let p_y = projection.find(y);

        let mut path = Vec::new();

        let cactus_tree = &self.graph;

        if p_x == p_y {
            // If {x, y} is a chain pair
            path.push(p_x);
        } else {
            // If {x, y} is not a chain pair
            let mut visited: FxHashSet<u64> = FxHashSet::default();
            let mut parents: FxHashMap<u64, u64> = FxHashMap::default();

            let mut stack: Vec<u64> = Vec::new();

            stack.push(p_x);

            while let Some(current) = stack.pop() {
                if !current != p_y && !visited.contains(&current) {
                    visited.insert(current);

                    let current_net_vertex = cactus_tree.is_net_vertex(current);
                    let neighbors = cactus_tree.graph.neighbors(current);

                    let neighbors = neighbors.filter(|&n| {
                        if current_net_vertex {
                            cactus_tree.is_chain_vertex(n) && n != current
                        } else {
                            cactus_tree.is_net_vertex(n) && n != current
                        }
                    });

                    for n in neighbors {
                        if !visited.contains(&n) {
                            stack.push(n);
                            parents.insert(n, current);
                        }
                    }
                }
            }

            let mut current = p_y;
            let mut path_ = vec![p_y];
            while current != p_x {
                let parent = parents.get(&current)?;
                path_.push(*parent);
                current = *parent;
            }

            path_.reverse();
            path.append(&mut path_);
        }

        path.shrink_to_fit();

        Some(path)
    }

    fn net_graph_black_edge_walk(
        vertices: &FxHashSet<u64>,
        biedged: &BiedgedGraph,
        x: u64,
        y: u64,
    ) -> bool {
        let start = x;
        let end = y;
        let adj_end = opposite_vertex(y);

        let mut visited: FxHashSet<u64> = FxHashSet::default();
        let mut stack: Vec<u64> = Vec::new();

        stack.push(start);

        while let Some(current) = stack.pop() {
            if current == end {
                return true;
            }

            if !visited.contains(&current) {
                visited.insert(current);

                let edges = biedged.graph.edges(current);

                if current == start || current == adj_end {
                    for (_, n, w) in edges {
                        if w.black > 0
                            && !visited.contains(&n)
                            && vertices.contains(&n)
                        {
                            stack.push(n);
                        }
                    }
                } else {
                    for (_, n, _) in edges {
                        if !visited.contains(&n)
                            && n != end
                            && vertices.contains(&n)
                        {
                            stack.push(n);
                        }
                    }
                }
            }
        }

        false
    }

    /// Build the net graph for the given snarl.
    pub fn build_net_graph(&self, x: u64, y: u64) -> NetGraph {
        let orig_graph = self.original_graph;

        let path = self
            .snarl_cactus_tree_path(&self.cactus_graph.projection, x, y)
            .unwrap();

        let proj_inv = self.cactus_graph.projection.get_inverse().unwrap();

        let tree_graph = &self.graph;

        let mut graph: UnGraphMap<u64, BiedgedWeight> = UnGraphMap::new();

        let mut vertices: Vec<u64> = path
            .iter()
            .filter_map(|&n| {
                if n == x || n == y || tree_graph.is_net_vertex(n) {
                    proj_inv.get(&n)
                } else {
                    None
                }
            })
            .flatten()
            .copied()
            .collect();

        vertices.sort_unstable();

        let mut black_edges: fnv::FnvHashSet<(u64, u64)> =
            fnv::FnvHashSet::default();
        let mut black_vertices: FxHashSet<u64> = FxHashSet::default();

        let vertices_set = vertices.iter().copied().collect::<FxHashSet<_>>();

        // Treat the edges of the snarl as if they already have black
        // edges, since they shouldn't have any in the net graph
        black_vertices.insert(x);
        black_vertices.insert(y);

        for v in vertices.iter() {
            for u in vertices.iter() {
                let mut add_pair = false;

                if opposite_vertex(*v) == *u {
                    add_pair = true;
                } else if v != u {
                    let b_v = self.cactus_graph.black_edge_cycle(*v);
                    let b_u = self.cactus_graph.black_edge_cycle(*u);

                    if b_v.is_some()
                        && b_v == b_u
                        && !black_vertices.contains(v)
                        && !black_vertices.contains(u)
                        && Self::net_graph_black_edge_walk(
                            &vertices_set,
                            orig_graph,
                            *v,
                            *u,
                        )
                    {
                        add_pair = true;
                    }
                }

                if add_pair {
                    let a = v.min(u);
                    let b = v.max(u);
                    black_edges.insert((*a, *b));
                    black_vertices.insert(*a);
                    black_vertices.insert(*b);
                }
            }
        }

        for (a, b) in black_edges.into_iter() {
            graph.add_edge(a, b, BiedgedWeight::black(1));
        }

        let gray_edges: fnv::FnvHashSet<(u64, u64)> = vertices
            .iter()
            .flat_map(|v| orig_graph.graph.edges(*v))
            .filter_map(|(v, n, w)| {
                if (n == x || n == y || vertices.contains(&n)) && w.gray > 0 {
                    let a = v.min(n);
                    let b = v.max(n);
                    Some((a, b))
                } else {
                    None
                }
            })
            .collect();

        for (a, b) in gray_edges.into_iter() {
            graph.add_edge(a, b, BiedgedWeight::gray(1));
        }

        // graph.shrink_to_fit();

        // let (node_count, node_cap) = graph.node_count_capacity();
        // let (edge_count, edge_cap) = graph.edge_count_capacity();

        // trace!("net_graph node count & capacity: ({}, {}); edges: ({}, {})",
        //        node_count, node_cap, edge_count, edge_cap);

        let net_graph = BiedgedGraph {
            graph,
            max_net_vertex: self.original_graph.max_net_vertex,
            max_chain_vertex: self.original_graph.max_chain_vertex,
        };

        NetGraph {
            graph: net_graph,

            x,
            y,
            path,
        }
    }

    /// Recursively check whether all of the chain pairs contained
    /// within the given chain pair are ultrabubbles. If so, returns a
    /// Some containing a vector of the contained chain pairs as chain
    /// edges, otherwise, if the chain pair isn't an ultrabubble,
    /// returns None.
    pub fn is_chain_pair_ultrabubble(
        &self,
        labels: &mut FxHashMap<(u64, u64), bool>,
        x: u64,
        chain_vx: u64,
    ) -> Option<Vec<(u64, u64)>> {
        let p_x = self.projected_node(x);
        if let Some(is_ultrabubble) = labels.get(&(p_x, chain_vx)) {
            if !is_ultrabubble {
                return None;
            }
        }

        let mut visited: FxHashSet<u64> = FxHashSet::default();
        let mut stack: Vec<(u64, u64)> = Vec::new();
        visited.insert(chain_vx);
        stack.push((chain_vx, p_x));

        let mut children = Vec::new();

        while let Some((prev, current)) = stack.pop() {
            if !visited.contains(&current) {
                visited.insert(current);

                if prev > current {
                    if let Some(is_ultrabubble) = labels.get(&(current, prev)) {
                        if !is_ultrabubble {
                            labels.insert((p_x, chain_vx), false);
                            return None;
                        } else if (current, prev) != (p_x, chain_vx) {
                            children.push((current, prev));
                        }
                    }
                }

                for n in self.base_graph().neighbors(current) {
                    if !visited.contains(&n) {
                        stack.push((current, n));
                    }
                }
            }
        }

        Some(children)
    }

    /// Recursively check whether all of the chain pairs contained
    /// within the given bridge pair are ultrabubbles. If so, returns
    /// a Some containing a vector of the contained chain pairs as
    /// chain edges, otherwise, if the bridge pair isn't an
    /// ultrabubble, returns None.
    pub fn is_bridge_pair_ultrabubble(
        &self,
        labels: &FxHashMap<(u64, u64), bool>,
        x: u64,
        y: u64,
        path: &[u64],
    ) -> Option<Vec<(u64, u64)>> {
        let a = opposite_vertex(x);
        let b = opposite_vertex(y);
        let p_a = self.projected_node(a);
        let p_b = self.projected_node(b);

        let mut path_vertices =
            path.iter().copied().collect::<FxHashSet<u64>>();

        path_vertices.insert(p_a);
        path_vertices.insert(p_b);

        let mut contained_chain_pairs: Vec<(u64, u64)> = Vec::new();

        let mut chain_vertices: FxHashSet<u64> = FxHashSet::default();

        for &v in path {
            if self.graph.is_chain_vertex(v) {
                chain_vertices.insert(v);
            } else {
                chain_vertices.extend(self.base_graph().neighbors(v).filter(
                    |&n| {
                        !path_vertices.contains(&n)
                            && self.graph.is_chain_vertex(n)
                    },
                ));
            }
        }

        for &cx in chain_vertices.iter() {
            let net_neighbors = self.base_graph().neighbors(cx).filter(|n| {
                !path_vertices.contains(&n) && !chain_vertices.contains(&n)
            });

            for nx in net_neighbors {
                let is_ultrabubble = labels.get(&(nx, cx))?;
                if !is_ultrabubble {
                    return None;
                } else {
                    contained_chain_pairs.push((nx, cx));
                }
            }
        }

        Some(contained_chain_pairs)
    }
}

/// A bridge forest derived from a cactus graph. Holds a reference to
/// the original biedged graph used to build the cactus graph, and
/// tracks the vertex projections from the original graph.
pub struct BridgeForest<'a> {
    pub original_graph: &'a BiedgedGraph,
    pub graph: BiedgedGraph,
    pub projection: Projection,
}

impl_biedged_wrapper!(BridgeForest<'a>);

impl<'a> BridgeForest<'a> {
    pub fn from_cactus_graph(cactus_graph: &'_ CactusGraph<'a>) -> Self {
        debug!("  ~~~  building bridge forest  ~~~");
        trace!("cloning cactus graph");
        let t = std::time::Instant::now();
        // let mut graph = cactus_graph.graph.clone();
        let mut graph = cactus_graph.graph.shrink_clone();
        trace!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

        let (node_count, node_cap) = graph.node_count_capacity();
        let (edge_count, edge_cap) = graph.edge_count_capacity();

        trace!(
            " | bridge forest, start, nodes | {} | {} |",
            node_count,
            node_cap
        );
        trace!(
            " | bridge forest, start, edges | {} | {} |",
            edge_count,
            edge_cap
        );

        trace!("cloning projection");
        let t = std::time::Instant::now();
        let mut projection = cactus_graph.projection.copy_without_inverse();
        trace!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

        debug!("contracting {} cycles", cactus_graph.cycles.len());
        let t = std::time::Instant::now();
        Self::contract_cycles(
            &mut graph,
            &cactus_graph.cycles,
            &mut projection,
        );
        debug!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

        trace!("building inverse projection map");
        let t = std::time::Instant::now();
        projection.build_inverse();
        trace!("  took {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

        graph.shrink_to_fit();

        let (node_count, node_cap) = graph.node_count_capacity();
        let (edge_count, edge_cap) = graph.edge_count_capacity();

        trace!(
            " | bridge forest, end, nodes | {} | {} |",
            node_count,
            node_cap
        );
        trace!(
            " | bridge forest, end, edges | {} | {} |",
            edge_count,
            edge_cap
        );

        debug!("  ~~~  bridge forest constructed  ~~~");

        Self {
            original_graph: cactus_graph.original_graph,
            graph,
            projection,
        }
    }

    /// Contracts each cycle into a single vertex, updating the projection
    /// map accordingly.
    pub fn contract_cycles(
        biedged: &mut BiedgedGraph,
        cycles: &[Vec<(u64, u64)>],
        projection: &mut Projection,
    ) {
        let _p_bar;

        #[cfg(not(feature = "progress_bars"))]
        {
            _p_bar = ();
        }

        #[cfg(feature = "progress_bars")]
        {
            use indicatif::{ProgressBar, ProgressStyle};
            _p_bar = ProgressBar::new(cycles.len() as u64);
            _p_bar.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40} {pos:>10}/{len:10}")
                    .progress_chars("##-"),
            );
        }

        for cycle in cycles {
            for &(from, to) in cycle {
                let from = projection.find(from);
                let to = projection.find(to);

                if from != to {
                    biedged.merge_vertices(from, to, projection);
                }
            }

            #[cfg(feature = "progress_bars")]
            {
                _p_bar.inc(1);
            }
        }

        #[cfg(feature = "progress_bars")]
        {
            _p_bar.finish();
        }
    }

    /// Find the bridge pairs in the graph, returning them as a set of
    /// snarls.
    pub fn find_bridge_pairs(&self) -> FxHashSet<BridgePair> {
        trace!(" ~~~ in find_bridge_pairs() ~~~ ");

        let mut bridge_pairs: FxHashSet<BridgePair> = FxHashSet::default();

        trace!(" getting inverse projection ");
        let proj_inv = self.projection.get_inverse().unwrap();

        for p_x in self.base_graph().nodes() {
            let neighbors = self
                .base_graph()
                .neighbors(p_x)
                .filter(|&n| n != p_x)
                .collect::<Vec<_>>();

            if neighbors.len() == 2 {
                let mut all_neighbors: Vec<u64> = Vec::new();

                for &n in neighbors.iter() {
                    let filtered = proj_inv
                        .get(&n)
                        .unwrap()
                        .iter()
                        .filter(|&b_n| {
                            let bn_n_ = opposite_vertex(*b_n);
                            let pn_n_ = self.projected_node(bn_n_);
                            pn_n_ == p_x
                        })
                        .copied();
                    all_neighbors.extend(filtered);
                }

                for &a in all_neighbors.iter() {
                    for &b in all_neighbors.iter() {
                        let x = a.min(b);
                        let y = a.max(b);
                        if x != y {
                            let x_ = opposite_vertex(x);
                            let y_ = opposite_vertex(y);

                            bridge_pairs.insert(BridgePair { x: x_, y: y_ });
                        }
                    }
                }
            }
        }

        bridge_pairs.shrink_to_fit();

        trace!(
            "bridge_pairs len: {}, capacity: {}",
            bridge_pairs.len(),
            bridge_pairs.capacity()
        );
        trace!(" ~~~ find_bridge_pairs() done ~~~ ");

        bridge_pairs
    }
}

/// Return the chain edges in the cactus tree as a map from pairs of
/// net and chain vertices to chain pair snarls.
pub fn chain_edges<'a>(
    chain_pairs: &'a FxHashSet<ChainPair>,
    cactus_tree: &'a CactusTree<'a>,
) -> FxHashMap<(u64, u64), (u64, u64)> {
    chain_pairs
        .iter()
        .map(move |&snarl| {
            let ChainPair { x, y } = snarl;
            let net = cactus_tree.projected_node(x);
            let chain = cactus_tree.black_edge_chain_vertex(x).unwrap();
            ((net, chain), (x, y))
        })
        .collect()
}

/// Labels chain edges as ultrabubbles if their net graphs are acyclic
/// and bridgeless, returning a map from chain edges to true/false.
pub fn chain_pair_ultrabubble_labels(
    cactus_tree: &CactusTree<'_>,
    chain_pairs: &FxHashSet<ChainPair>,
) -> FxHashMap<(u64, u64), bool> {
    trace!(" ~~~ in chain_pair_ultrabubble_labels ~~~ ");
    let mut chain_edges = chain_edges(chain_pairs, cactus_tree);

    chain_edges.shrink_to_fit();

    trace!(
        "chain_edges len: {}, capacity: {}",
        chain_edges.len(),
        chain_edges.capacity()
    );

    let mut chain_edge_labels = FxHashMap::default();

    let iter;

    #[cfg(feature = "progress_bars")]
    {
        iter = chain_edges
            .par_iter()
            .progress_with(progress_bar(chain_edges.len(), false));
    }
    #[cfg(not(feature = "progress_bars"))]
    {
        iter = chain_edges.par_iter();
    }

    trace!("labeling chain edges using net graphs");
    chain_edge_labels.par_extend(iter.map(|(&(net, chain), &(x, y))| {
        let net_graph = cactus_tree.build_net_graph(x, y);
        let result = net_graph.is_ultrabubble();
        ((net, chain), result)
    }));

    chain_edge_labels.shrink_to_fit();

    trace!(
        "chain_edge_labels len: {}, capacity: {}",
        chain_edge_labels.len(),
        chain_edge_labels.capacity()
    );

    trace!(" ~~~ chain_pair_ultrabubble_labels done ~~~ ");
    chain_edge_labels
}

/// Using the provided chain edge labels, returns a map from chain
/// pairs to contained ultrabubbles. Each entry is an ultrabubble, if
/// the entry has an empty vector, it doesn't contain any
/// ultrabubbles.
pub fn chain_pair_contained_ultrabubbles(
    cactus_tree: &CactusTree<'_>,
    chain_pairs: &FxHashSet<ChainPair>,
    chain_edge_labels: &mut FxHashMap<(u64, u64), bool>,
) -> FxHashMap<(u64, u64), Vec<(u64, u64)>> {
    trace!(" ~~~ in chain_pair_contained_ultrabubbles ~~~ ");
    let mut chain_pair_ultrabubbles = FxHashMap::default();

    for &snarl in chain_pairs.iter() {
        let ChainPair { x, y } = snarl;
        let c_x = cactus_tree.black_edge_chain_vertex(x).unwrap();

        let contained_chain_pairs =
            cactus_tree.is_chain_pair_ultrabubble(chain_edge_labels, x, c_x);

        if let Some(children) = contained_chain_pairs {
            chain_pair_ultrabubbles.insert((x, y), children);
        }
    }

    chain_pair_ultrabubbles.shrink_to_fit();

    trace!(
        "chain_pair_ultrabubbles len: {}, capacity: {}",
        chain_pair_ultrabubbles.len(),
        chain_pair_ultrabubbles.capacity()
    );

    trace!(" ~~~ chain_pair_contained_ultrabubbles done ~~~ ");
    chain_pair_ultrabubbles
}

/// Using the provided chain edge labels, returns a map from bridge
/// pairs to their contained ultrabubbles. Runs in parallel.
pub fn bridge_pair_ultrabubbles(
    cactus_tree: &CactusTree<'_>,
    bridge_pairs: &FxHashSet<BridgePair>,
    chain_edge_labels: &FxHashMap<(u64, u64), bool>,
) -> FxHashMap<(u64, u64), Vec<(u64, u64)>> {
    trace!(" ~~~ in bridge_pair_ultrabubbles ~~~ ");
    let mut bridge_pair_ultrabubbles = FxHashMap::default();

    let mut bridge_pair_labels: FxHashMap<(u64, u64), Vec<u64>> =
        FxHashMap::default();

    trace!("Bridge pairs - checking net graphs");
    let bridge_pair_iter;
    #[cfg(feature = "progress_bars")]
    {
        bridge_pair_iter = bridge_pairs
            .par_iter()
            .progress_with(progress_bar(bridge_pairs.len(), true));
    }
    #[cfg(not(feature = "progress_bars"))]
    {
        bridge_pair_iter = bridge_pairs.par_iter();
    }

    bridge_pair_labels.par_extend(bridge_pair_iter.filter_map(|&snarl| {
        let BridgePair { x, y } = snarl;
        let net_graph = cactus_tree.build_net_graph(x, y);

        if net_graph.is_ultrabubble() {
            Some(((x, y), net_graph.path))
        } else {
            None
        }
    }));

    bridge_pair_labels.shrink_to_fit();

    trace!("Bridge pairs - checking contained");

    let label_iter;
    #[cfg(feature = "progress_bars")]
    {
        label_iter = bridge_pair_labels
            .par_iter()
            .progress_with(progress_bar(bridge_pair_labels.len(), true));
    }
    #[cfg(not(feature = "progress_bars"))]
    {
        label_iter = bridge_pair_labels.par_iter();
    }

    bridge_pair_ultrabubbles.par_extend(label_iter.filter_map(
        |(&(x, y), path)| {
            let contained_chain_pairs = cactus_tree.is_bridge_pair_ultrabubble(
                &chain_edge_labels,
                x,
                y,
                path,
            );

            contained_chain_pairs.map(|c| ((x, y), c))
        },
    ));

    bridge_pair_ultrabubbles.shrink_to_fit();

    trace!(
        "bridge_pair_ultrabubbles len: {}, capacity: {}",
        bridge_pair_ultrabubbles.len(),
        bridge_pair_ultrabubbles.capacity()
    );

    trace!(" ~~~ bridge_pair_ultrabubbles done ~~~ ");

    bridge_pair_ultrabubbles
}

/// Using the provided cactus tree and bridge forest for a biedged
/// graph, find the ultrabubbles and their nesting in the graph. Runs
/// in parallel.
pub fn find_ultrabubbles(
    cactus_tree: &CactusTree<'_>,
    bridge_forest: &BridgeForest<'_>,
) -> FxHashMap<(u64, u64), Vec<(u64, u64)>> {
    debug!("Finding chain pairs");
    let chain_pairs = cactus_tree.find_chain_pairs();
    debug!("Found {} chain pairs", chain_pairs.len());

    debug!("Finding bridge pairs");
    let bridge_pairs = bridge_forest.find_bridge_pairs();
    debug!("Found {} bridge pairs", bridge_pairs.len());

    debug!("Labeling chain edges");
    let mut chain_edge_labels =
        chain_pair_ultrabubble_labels(cactus_tree, &chain_pairs);

    debug!("Checking nested chain pairs");
    let chain_ultrabubbles = chain_pair_contained_ultrabubbles(
        cactus_tree,
        &chain_pairs,
        &mut chain_edge_labels,
    );

    debug!("Checking bridge pairs");
    let bridge_ultrabubbles = bridge_pair_ultrabubbles(
        cactus_tree,
        &bridge_pairs,
        &chain_edge_labels,
    );

    let chain_edges_map = chain_edges(&chain_pairs, &cactus_tree);

    chain_ultrabubbles
        .into_iter()
        .chain(bridge_ultrabubbles.into_iter())
        .map(|(key, cont)| {
            (
                key,
                cont.into_iter()
                    .filter_map(|e| chain_edges_map.get(&e).cloned())
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

/// Inverses the vertex projection of the provided ultrabubbles to the
/// node ID space of the graph used to construct the original biedged
/// graph.
pub fn inverse_map_ultrabubbles(
    ultrabubbles: FxHashMap<(u64, u64), Vec<(u64, u64)>>,
) -> FxHashMap<(u64, u64), Vec<(u64, u64)>> {
    ultrabubbles
        .into_iter()
        .map(|((x, y), contained)| {
            use crate::projection::id_from_black_edge;
            let x = id_from_black_edge(x);
            let y = id_from_black_edge(y);
            let contained = contained
                .into_iter()
                .map(|(a, b)| (id_from_black_edge(a), id_from_black_edge(b)))
                .collect();
            ((x, y), contained)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn graph_from_paper() -> BiedgedGraph {
        let edges = vec![
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (3, 5),
            (4, 6),
            (5, 6),
            (5, 7),
            (6, 10),
            (6, 11),
            (7, 8),
            (7, 9),
            (8, 9),
            (9, 11),
            (10, 11),
            (11, 12),
            (12, 13),
            (12, 14),
            (13, 15),
            (14, 15),
            (15, 16),
            (15, 17),
            (15, 12),
        ];

        BiedgedGraph::from_directed_edges(edges).unwrap()
    }

    fn example_graph() -> BiedgedGraph {
        /*               -i
                 &     &/
        a--b==c--e==f--h--j
               \ |   \ |
                -d    -g
                 &     &

        & self cycles
        - 1 black edge
        = 2 black edges
                */

        let mut graph: BiedgedGraph = BiedgedGraph::new();

        for i in 0..=9 {
            graph.add_node(i);
        }

        let edges = vec![
            (0, 1),
            (1, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (3, 3),
            (3, 4),
            (4, 4),
            (4, 5),
            (4, 5),
            (5, 6),
            (5, 7),
            (6, 6),
            (7, 6),
            (7, 7),
            (7, 8),
            (7, 9),
        ];

        for (a, b) in edges {
            graph.add_edge(a, b, BiedgedWeight::black(1));
        }

        graph.max_net_vertex = (graph.graph.node_count() - 1) as u64;
        graph.max_chain_vertex = graph.max_net_vertex;

        graph
    }

    #[test]
    fn simple_contract_all_gray_edges() {
        let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3)];

        let mut graph = BiedgedGraph::from_directed_edges(edges).unwrap();

        let mut proj = Projection::new_for_biedged_graph(&graph);

        CactusGraph::contract_all_gray_edges(&mut graph, &mut proj);

        let a = proj.find(0);
        let b = proj.find(1);
        let c = proj.find(3);
        let d = proj.find(7);

        assert_eq!(
            graph.graph.edge_weight(a, b),
            Some(&BiedgedWeight::black(1))
        );
        assert_eq!(
            graph.graph.edge_weight(c, d),
            Some(&BiedgedWeight::black(1))
        );
        assert_eq!(
            graph.graph.edge_weight(b, c),
            Some(&BiedgedWeight::black(2))
        );

        assert_eq!(graph.graph.node_count(), 4);
        assert_eq!(graph.black_edge_count(), 4);
        assert_eq!(graph.gray_edge_count(), 0);
        assert_eq!(graph.graph.edge_count(), 3);
    }

    #[test]
    fn paper_contract_all_gray_edges() {
        let mut graph: BiedgedGraph = graph_from_paper();

        let mut proj = Projection::new_for_biedged_graph(&graph);
        CactusGraph::contract_all_gray_edges(&mut graph, &mut proj);

        assert_eq!(graph.gray_edge_count(), 0);
        assert_eq!(
            graph.black_edge_count(),
            18,
            "Expected 18 black edges, is actually {:#?}",
            graph.black_edge_count()
        );
        assert_eq!(graph.graph.node_count(), 12);
    }

    fn segment_split_name(
        name_map: &gfa::gfa::name_conversion::NameMap,
        n: u64,
    ) -> Option<String> {
        use crate::projection::id_from_black_edge;
        let not_orig = n % 2 != 0;
        let id = id_from_black_edge(n);
        let mut name: String = {
            let bytes = name_map.inverse_map_name(id as usize)?;
            let name_str = std::str::from_utf8(bytes).unwrap();
            name_str.into()
        };
        if not_orig {
            name.push('_');
        }
        Some(name)
    }

    #[test]
    fn edge_contraction_projection() {
        use crate::projection::id_to_black_edge;
        use gfa::{
            gfa::{name_conversion::NameMap, GFA},
            parser::GFAParser,
        };

        let parser = GFAParser::new();
        let vec_gfa: GFA<Vec<u8>, ()> =
            parser.parse_file("./test/gfas/paper.gfa").unwrap();

        let name_map = NameMap::build_from_gfa(&vec_gfa);
        let gfa = name_map.gfa_bytestring_to_usize(&vec_gfa, false).unwrap();

        let mut graph = BiedgedGraph::from_gfa(&gfa);

        let mut proj = Projection::new_for_biedged_graph(&graph);

        CactusGraph::contract_all_gray_edges(&mut graph, &mut proj);

        let proj_names = vec_gfa
            .segments
            .iter()
            .map(|s| {
                let orig = name_map.map_name(&s.name).unwrap();
                let orig_str = std::str::from_utf8(&s.name).unwrap();
                let orig_name = orig_str.to_string();
                let (l, r) = id_to_black_edge(orig as u64);
                let l_end = proj.find(l);
                let r_end = proj.find(r);
                let l_end = segment_split_name(&name_map, l_end).unwrap();
                let r_end = segment_split_name(&name_map, r_end).unwrap();
                (orig_name, (l_end, r_end))
            })
            .collect::<Vec<_>>();

        let expected_names: Vec<_> = vec![
            ("a", ("a", "a_")),
            ("b", ("a_", "b_")),
            ("c", ("a_", "b_")),
            ("d", ("b_", "d_")),
            ("e", ("d_", "e_")),
            ("f", ("d_", "e_")),
            ("g", ("e_", "k_")),
            ("h", ("e_", "h_")),
            ("i", ("h_", "h_")),
            ("j", ("h_", "k_")),
            ("k", ("k_", "k_")),
            ("l", ("k_", "p_")),
            ("m", ("p_", "m_")),
            ("n", ("m_", "n_")),
            ("o", ("m_", "n_")),
            ("p", ("n_", "p_")),
            ("q", ("p_", "q_")),
            ("r", ("p_", "r_")),
        ]
        .into_iter()
        .map(|(a, (l, r))| (a.to_string(), (l.to_string(), r.to_string())))
        .collect();

        assert_eq!(expected_names, proj_names);
    }

    #[test]
    fn cycle_detection() {
        let graph = example_graph();

        let cycles = CactusGraph::find_cycles(&graph);

        assert_eq!(
            cycles,
            vec![
                vec![(1, 2), (2, 1)],
                vec![(4, 4)],
                vec![(4, 5), (5, 4)],
                vec![(7, 7)],
                vec![(6, 6)],
                vec![(3, 3)],
                vec![(6, 7), (7, 5), (5, 6)],
                vec![(3, 4), (4, 2), (2, 3)],
            ]
        );
    }

    #[test]
    fn test_build_cactus_tree() {
        let mut graph = example_graph();

        let cycles = CactusGraph::find_cycles(&graph);

        let (cycle_chain_map, chain_vertices) =
            CactusTree::construct_chain_vertices(&mut graph, &cycles);

        assert_eq!(cycles.len(), chain_vertices.len());

        for (edge, chain_vx) in cycle_chain_map.iter() {
            let chain_edges = graph
                .graph
                .edges(*chain_vx)
                .map(|x| x.1)
                .collect::<Vec<_>>();

            assert!(chain_edges.contains(&edge.0));
            assert!(chain_edges.contains(&edge.1));
        }
    }
}
