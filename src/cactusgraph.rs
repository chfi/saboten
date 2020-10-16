use fnv::{FnvHashMap, FnvHashSet};
use petgraph::prelude::*;

use rayon::prelude::*;

use log::debug;

use crate::{
    biedgedgraph::{
        end_to_black_edge, opposite_vertex, BiedgedGraph, BiedgedWeight,
    },
    netgraph::NetGraph,
    projection::Projection,
    ultrabubble::{ChainEdge, ChainPair, Snarl},
};

#[cfg(feature = "progress_bars")]
use indicatif::ParallelProgressIterator;

#[cfg(feature = "progress_bars")]
fn progress_bar(len: usize) -> indicatif::ProgressBar {
    use indicatif::{ProgressBar, ProgressStyle};
    let len = len as u64;
    let p_bar = ProgressBar::new(len);
    p_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:80} {pos:>7}/{len:7}")
            .progress_chars("##-"),
    );
    p_bar.enable_steady_tick(1000);
    p_bar
}

macro_rules! impl_biedged_wrapper {
    ($for:ty) => {
        impl<'a> BiedgedWrapper for $for {
            fn base_graph(&self) -> &UnGraphMap<u64, BiedgedWeight> {
                &self.graph.graph
            }

            fn biedged_graph(&self) -> &BiedgedGraph {
                &self.graph
            }

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

    fn projected_node(&self, n: u64) -> u64 {
        let graph = self.biedged_graph();
        let proj = self.projection();
        graph.projected_node(proj, n)
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
    pub cycle_map: FnvHashMap<(u64, u64), Vec<usize>>,
}

impl_biedged_wrapper!(CactusGraph<'a>);

impl<'a> CactusGraph<'a> {
    /// Construct a cactus graph from a biedged graph. Clones the
    /// input graph before mutating, and keeps a reference to the
    /// original.
    pub fn from_biedged_graph(biedged_graph: &'a BiedgedGraph) -> Self {
        let mut graph = biedged_graph.clone();

        let mut projection = Projection::new_for_biedged_graph(&graph);

        Self::contract_all_gray_edges(&mut graph, &mut projection);

        let components = Self::find_3_edge_connected_components(&graph);

        Self::merge_components(&mut graph, components, &mut projection);

        let cycles = Self::find_cycles(&graph);

        let mut cycle_map: FnvHashMap<(u64, u64), Vec<usize>> =
            FnvHashMap::default();

        for (i, cycle) in cycles.iter().enumerate() {
            for &(a, b) in cycle.iter() {
                let l = a.min(b);
                let r = a.max(b);
                cycle_map.entry((l, r)).or_default().push(i);
            }
        }

        projection.build_inverse();

        CactusGraph {
            original_graph: biedged_graph,
            graph,
            projection,
            cycles,
            cycle_map,
        }
    }

    fn contract_all_gray_edges(
        biedged: &mut BiedgedGraph,
        projection: &mut Projection,
    ) {
        while let Some((from, to)) = biedged.next_gray_edge() {
            biedged.contract_edge(from, to, projection).unwrap();
        }
    }

    fn find_3_edge_connected_components(
        biedged: &BiedgedGraph,
    ) -> Vec<Vec<usize>> {
        let edges = biedged
            .graph
            .all_edges()
            .flat_map(|(a, b, w)| {
                std::iter::repeat((a as usize, b as usize)).take(w.black)
            })
            .collect::<Vec<_>>();

        let graph = three_edge_connected::Graph::from_edges(edges.into_iter());

        let components = three_edge_connected::find_components(&graph.graph);
        // Many of the components returned by the algorithm can be singletons, which we don't need to do anything with, hence we filter them out.
        let components: Vec<_> =
            components.into_iter().filter(|c| c.len() > 1).collect();

        // The 3EC library maps the graph into node IDs starting from
        // zero; even if the input biedged graph also does so, it's
        // better to make sure the node IDs are mapped backed to their
        // input IDs.
        let components = graph.invert_components(components);

        components
    }
    fn merge_components(
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

        let mut visited: FnvHashSet<u64> = FnvHashSet::default();
        let mut parents: FnvHashMap<u64, u64> = FnvHashMap::default();

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
                            } else {
                                if !visited.contains(&adj) {
                                    if weight.black == 2 {
                                        cycles.push(vec![
                                            (current, adj),
                                            (adj, current),
                                        ]);
                                    }
                                    stack.push(adj);
                                    parents.insert(adj, current);
                                } else {
                                    if parents.get(&current) != Some(&adj) {
                                        cycle_ends.push((adj, current));
                                    }
                                }
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

    /// Given a vertex ID in the original biedged graph, find the
    /// simple cycle its incident black edge maps to.
    fn black_edge_cycle(&self, x: u64) -> Option<&Vec<usize>> {
        let (l, r) = end_to_black_edge(x);
        let p_l = self.projection.find(l);
        let p_r = self.projection.find(r);
        let a = p_l.min(p_r);
        let b = p_l.max(p_r);
        let cycles = self.cycle_map.get(&(a, b))?;
        Some(&cycles)
    }

    pub fn is_chain_pair(&self, x: u64, y: u64) -> bool {
        if x == y {
            return false;
        }

        let p_x = self.projection.find(x);
        let p_y = self.projection.find(y);

        if p_x != p_y {
            return false;
        }

        let x_cycles = self.black_edge_cycle(x);
        let y_cycles = self.black_edge_cycle(y);

        if x_cycles.is_none() || y_cycles.is_none() {
            return false;
        }

        x_cycles == y_cycles
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
    pub chain_vertices: FnvHashSet<u64>,
    pub cycle_chain_map: FnvHashMap<(u64, u64), u64>,
}

impl<'a> BiedgedWrapper for CactusTree<'a> {
    fn base_graph(&self) -> &UnGraphMap<u64, BiedgedWeight> {
        &self.graph.graph
    }

    fn biedged_graph(&self) -> &BiedgedGraph {
        &self.graph
    }

    fn projection(&self) -> &Projection {
        &self.cactus_graph.projection
    }
}

impl<'a> CactusTree<'a> {
    pub fn from_cactus_graph(cactus_graph: &'a CactusGraph<'a>) -> Self {
        let mut graph = cactus_graph.graph.clone();

        let cycles = cactus_graph.cycles.clone();

        let (cycle_chain_map, chain_vertices) =
            Self::construct_chain_vertices(&mut graph, &cycles);

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
    ) -> (FnvHashMap<(u64, u64), u64>, FnvHashSet<u64>) {
        let mut cycle_chain_map = FnvHashMap::default();
        let mut chain_vertices = FnvHashSet::default();

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
    pub fn black_edge_chain_vertex(&self, b: u64) -> Option<u64> {
        let (l, r) = end_to_black_edge(b);
        let p_l = self.projected_node(l);
        let p_r = self.projected_node(r);
        let a = p_l.min(p_r);
        let b = p_l.max(p_r);
        let chain_vx = self.cycle_chain_map.get(&(a, b))?;
        Some(*chain_vx)
    }

    pub fn chain_pair_to_chain_edge(
        &self,
        x: u64,
        y: u64,
    ) -> Option<ChainEdge> {
        let chain = self.black_edge_chain_vertex(x)?;
        let chain_y = self.black_edge_chain_vertex(y)?;

        let p_x = self.projected_node(x);
        let p_y = self.projected_node(y);

        if chain != chain_y || p_x != p_y {
            return None;
        }

        let net = p_x;
        Some(ChainEdge { net, chain })
    }

    pub fn is_chain_edge(&self, a: u64, b: u64) -> bool {
        let be_graph = self.biedged_graph();

        let a = self.projected_node(a);
        let b = self.projected_node(b);

        if self.base_graph().contains_edge(a, b) {
            let n = a.min(b);
            let c = a.max(b);
            be_graph.is_net_vertex(n) && be_graph.is_chain_vertex(c)
        } else {
            false
        }
    }

    pub fn is_bridge_edge(&self, a: u64, b: u64) -> bool {
        let a = self.projected_node(a);
        let b = self.projected_node(b);

        let be_graph = self.biedged_graph();

        if be_graph.graph.contains_edge(a, b) {
            be_graph.is_net_vertex(a) && be_graph.is_net_vertex(b)
        } else {
            false
        }
    }

    /// Find the chain pairs using the chain vertices in the cactus
    /// tree, and return them as a set of snarls.
    pub fn find_chain_pairs(&self) -> FnvHashSet<Snarl> {
        let mut chain_pairs: FnvHashSet<Snarl> = FnvHashSet::default();

        let cactus_graph_inverse =
            self.cactus_graph.projection.get_inverse().unwrap();

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
                                chain_pairs.insert(Snarl::chain_pair(a_, b_));
                            }
                        }
                    }
                }
            }
        }

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
            let mut visited: FnvHashSet<u64> = FnvHashSet::default();
            let mut parents: FnvHashMap<u64, u64> = FnvHashMap::default();

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

        Some(path)
    }

    fn net_graph_black_edge_walk(
        biedged: &BiedgedGraph,
        x: u64,
        y: u64,
    ) -> bool {
        let start = x;
        let end = y;
        let adj_end = opposite_vertex(y);

        let mut visited: FnvHashSet<u64> = FnvHashSet::default();
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
                        if w.black > 0 {
                            if !visited.contains(&n) {
                                stack.push(n);
                            }
                        }
                    }
                } else {
                    for (_, n, _) in edges {
                        if !visited.contains(&n) && n != end {
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

        vertices.sort();

        let gray_edges: FnvHashSet<(u64, u64)> = vertices
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

        let mut black_edges: FnvHashSet<(u64, u64)> = FnvHashSet::default();
        let mut black_vertices: FnvHashSet<u64> = FnvHashSet::default();

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
                    {
                        if Self::net_graph_black_edge_walk(orig_graph, *v, *u) {
                            add_pair = true;
                        }
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

        for &(a, b) in black_edges.iter() {
            graph.add_edge(a, b, BiedgedWeight::black(1));
        }

        for &(a, b) in gray_edges.iter() {
            graph.add_edge(a, b, BiedgedWeight::gray(1));
        }

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
        labels: &mut FnvHashMap<(u64, u64), bool>,
        x: u64,
        y: u64,
        chain_vx: u64,
    ) -> Option<Vec<(u64, u64)>> {
        let p_x = self.projected_node(x);
        if let Some(is_ultrabubble) = labels.get(&(p_x, chain_vx)) {
            if !is_ultrabubble {
                return None;
            }
        }

        let mut visited: FnvHashSet<u64> = FnvHashSet::default();
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
        labels: &FnvHashMap<(u64, u64), bool>,
        x: u64,
        y: u64,
        path: &[u64],
    ) -> Option<Vec<(u64, u64)>> {
        let a = opposite_vertex(x);
        let b = opposite_vertex(y);
        let p_a = self.projected_node(a);
        let p_b = self.projected_node(b);

        let mut path_vertices =
            path.iter().copied().collect::<FnvHashSet<u64>>();

        path_vertices.insert(p_a);
        path_vertices.insert(p_b);

        let mut contained_chain_pairs: Vec<(u64, u64)> = Vec::new();

        let mut chain_vertices: FnvHashSet<u64> = FnvHashSet::default();

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
        let mut graph = cactus_graph.graph.clone();
        let mut projection = cactus_graph.projection.copy_without_inverse();

        Self::contract_cycles(
            &mut graph,
            &cactus_graph.cycles,
            &mut projection,
        );

        projection.build_inverse();

        Self {
            original_graph: cactus_graph.original_graph,
            graph,
            projection,
        }
    }

    /// Contracts each cycle into a single vertex, updating the projection
    /// map accordingly.
    fn contract_cycles(
        biedged: &mut BiedgedGraph,
        cycles: &[Vec<(u64, u64)>],
        projection: &mut Projection,
    ) {
        for cycle in cycles {
            for &(from, to) in cycle {
                let from = if biedged.graph.contains_node(from) {
                    from
                } else {
                    projection.find(from)
                };

                let to = if biedged.graph.contains_node(to) {
                    to
                } else {
                    projection.find(to)
                };

                biedged.merge_vertices(from, to, projection);
            }
        }
    }

    /// Find the bridge pairs in the graph, returning them as a set of
    /// snarls.
    pub fn find_bridge_pairs(&self) -> FnvHashSet<Snarl> {
        let mut bridge_pairs: FnvHashSet<Snarl> = FnvHashSet::default();

        let proj_inv = self.projection.get_inverse().unwrap();

        for p_x in self.base_graph().nodes() {
            let neighbors =
                self.base_graph().neighbors(p_x).collect::<Vec<_>>();

            if neighbors.len() == 2 {
                let mut all_neighbors: Vec<u64> = Vec::new();
                for n in neighbors {
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
                            bridge_pairs.insert(Snarl::bridge_pair(x_, y_));
                        }
                    }
                }
            }
        }

        bridge_pairs
    }
}

pub fn chain_edges_sorted(
    c_edges: &FnvHashMap<ChainEdge, FnvHashSet<ChainPair>>,
) -> Vec<(ChainEdge, usize)> {
    let mut edge_counts: Vec<_> = c_edges
        .iter()
        .map(|(c_e, pairs)| (*c_e, pairs.len()))
        .collect();

    edge_counts.sort_by(|a, b| a.1.cmp(&b.1));

    edge_counts
}

/// Return the chain edges in the cactus tree as a map from pairs of
/// net and chain vertices to chain pair snarls.
pub fn chain_edges<'a>(
    chain_pairs: &'a FnvHashSet<Snarl>,
    cactus_tree: &'a CactusTree<'a>,
) -> FnvHashMap<(u64, u64), (u64, u64)> {
    chain_pairs
        .iter()
        .filter_map(move |&snarl| {
            if let Snarl::ChainPair { x, y } = snarl {
                let net = cactus_tree.projected_node(x);
                let chain = cactus_tree.black_edge_chain_vertex(x).unwrap();
                Some(((net, chain), (x, y)))
            } else {
                None
            }
        })
        .collect()
}

pub fn chain_edges_all<'a>(
    chain_pairs: &'a FnvHashSet<Snarl>,
    cactus_tree: &'a CactusTree<'a>,
) -> FnvHashMap<(u64, u64), FnvHashSet<(u64, u64)>> {
    let mut chain_edges_map: FnvHashMap<(u64, u64), FnvHashSet<(u64, u64)>> =
        FnvHashMap::default();

    for &snarl in chain_pairs.iter() {
        if let Snarl::ChainPair { x, y } = snarl {
            let net = cactus_tree.projected_node(x);
            let chain = cactus_tree.black_edge_chain_vertex(x).unwrap();
            let entry = chain_edges_map.entry((net, chain)).or_default();
            entry.insert((x, y));
        }
    }

    chain_edges_map
}

/// Labels chain edges as ultrabubbles if their net graphs are acyclic
/// and bridgeless, returning a map from chain edges to true/false.
pub fn chain_pair_ultrabubble_labels(
    cactus_tree: &CactusTree<'_>,
    chain_pairs: &FnvHashSet<Snarl>,
) -> FnvHashMap<(u64, u64), bool> {
    let chain_edges = chain_edges(chain_pairs, cactus_tree);

    let mut chain_edge_labels = FnvHashMap::default();

    let iter;

    #[cfg(feature = "progress_bars")]
    {
        iter = chain_edges
            .par_iter()
            .progress_with(progress_bar(chain_edges.len()));
    }
    #[cfg(not(feature = "progress_bars"))]
    {
        iter = chain_edges.par_iter();
    }

    chain_edge_labels.par_extend(iter.map(|(&(net, chain), &(x, y))| {
        let net_graph = cactus_tree.build_net_graph(x, y);
        let result = net_graph.is_ultrabubble();
        ((net, chain), result)
    }));

    chain_edge_labels
}

/// Using the provided chain edge labels, returns a map from chain
/// pairs to contained ultrabubbles. Each entry is an ultrabubble, if
/// the entry has an empty vector, it doesn't contain any
/// ultrabubbles.
pub fn chain_pair_contained_ultrabubbles(
    cactus_tree: &CactusTree<'_>,
    chain_pairs: &FnvHashSet<Snarl>,
    chain_edge_labels: &mut FnvHashMap<(u64, u64), bool>,
) -> FnvHashMap<(u64, u64), Vec<(u64, u64)>> {
    let mut chain_pair_ultrabubbles = FnvHashMap::default();

    for &snarl in chain_pairs.iter() {
        if let Snarl::ChainPair { x, y } = snarl {
            let c_x = cactus_tree.black_edge_chain_vertex(x).unwrap();

            let contained_chain_pairs = cactus_tree.is_chain_pair_ultrabubble(
                chain_edge_labels,
                x,
                y,
                c_x,
            );

            if let Some(children) = contained_chain_pairs {
                chain_pair_ultrabubbles.insert((x, y), children);
            }
        }
    }

    chain_pair_ultrabubbles
}

/// Using the provided chain edge labels, returns a map from bridge
/// pairs to their contained ultrabubbles. Runs in parallel.
pub fn bridge_pair_ultrabubbles(
    cactus_tree: &CactusTree<'_>,
    bridge_pairs: &FnvHashSet<Snarl>,
    chain_edge_labels: &FnvHashMap<(u64, u64), bool>,
) -> FnvHashMap<(u64, u64), Vec<(u64, u64)>> {
    let mut bridge_pair_ultrabubbles = FnvHashMap::default();

    let mut bridge_pair_labels: FnvHashMap<(u64, u64), Vec<u64>> =
        FnvHashMap::default();

    let bridge_pair_iter;
    #[cfg(feature = "progress_bars")]
    {
        bridge_pair_iter = bridge_pairs
            .par_iter()
            .progress_with(progress_bar(bridge_pairs.len()));
    }
    #[cfg(not(feature = "progress_bars"))]
    {
        bridge_pair_iter = bridge_pairs.par_iter();
    }

    debug!("Bridge pairs - checking net graphs");
    bridge_pair_labels.par_extend(bridge_pair_iter.filter_map(|&snarl| {
        if let Snarl::BridgePair { x, y } = snarl {
            let net_graph = cactus_tree.build_net_graph(x, y);

            if net_graph.is_ultrabubble() {
                return Some(((x, y), net_graph.path.clone()));
            }
        }
        None
    }));

    let label_iter;
    #[cfg(feature = "progress_bars")]
    {
        label_iter = bridge_pair_labels
            .par_iter()
            .progress_with(progress_bar(bridge_pair_labels.len()));
    }
    #[cfg(not(feature = "progress_bars"))]
    {
        label_iter = bridge_pair_labels.par_iter();
    }

    debug!("Bridge pairs - checking contained");
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

    bridge_pair_ultrabubbles
}

/// Using the provided cactus tree and bridge forest for a biedged
/// graph, find the ultrabubbles and their nesting in the graph. Runs
/// in parallel.
pub fn find_ultrabubbles(
    cactus_tree: &CactusTree<'_>,
    bridge_forest: &BridgeForest<'_>,
) -> FnvHashMap<(u64, u64), Vec<(u64, u64)>> {
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

    let ultrabubbles = chain_ultrabubbles
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
        .collect();

    ultrabubbles
}

pub fn inverse_map_ultrabubbles(
    ultrabubbles: FnvHashMap<(u64, u64), Vec<(u64, u64)>>,
) -> FnvHashMap<(u64, u64), Vec<(u64, u64)>> {
    ultrabubbles
        .into_iter()
        .map(|((x, y), contained)| {
            use crate::biedgedgraph::id_from_black_edge;
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

        let a = proj.projected(0);
        let b = proj.projected(1);
        let c = proj.projected(3);
        let d = proj.projected(7);

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
    ) -> Option<bstr::BString> {
        use crate::biedgedgraph::id_from_black_edge;
        let not_orig = n % 2 != 0;
        let id = id_from_black_edge(n);
        let mut name: bstr::BString =
            name_map.inverse_map_name(id as usize)?.to_owned();
        if not_orig {
            name.push(b'_');
        }
        Some(name)
    }

    #[test]
    fn edge_contraction_projection() {
        use crate::biedgedgraph::id_to_black_edge;
        use bstr::BString;
        use gfa::{
            gfa::{name_conversion::NameMap, GFA},
            parser::GFAParser,
        };

        let parser = GFAParser::new();
        let bstr_gfa: GFA<bstr::BString, ()> =
            parser.parse_file("./test/gfas/paper.gfa").unwrap();

        let name_map = NameMap::build_from_gfa(&bstr_gfa);
        let gfa = name_map.gfa_bstring_to_usize(&bstr_gfa, false).unwrap();

        let mut graph = BiedgedGraph::from_gfa(&gfa);

        let mut proj = Projection::new_for_biedged_graph(&graph);

        CactusGraph::contract_all_gray_edges(&mut graph, &mut proj);

        let proj_names = bstr_gfa
            .segments
            .iter()
            .map(|s| {
                let orig = name_map.map_name(&s.name).unwrap();
                let orig_name = s.name.to_owned();
                let (l, r) = id_to_black_edge(orig as u64);
                let l_end = proj.projected(l);
                let r_end = proj.projected(r);
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
        .map(|(a, (l, r))| {
            (BString::from(a), (BString::from(l), BString::from(r)))
        })
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
