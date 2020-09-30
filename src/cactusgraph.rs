use fnv::{FnvHashMap, FnvHashSet};
use petgraph::prelude::*;

use crate::{
    biedged_to_cactus,
    biedgedgraph::{
        end_to_black_edge, opposite_vertex, BiedgedGraph, BiedgedWeight,
    },
    netgraph::NetGraph,
    projection::Projection,
};

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
    pub fn from_biedged_graph(biedged_graph: &'a BiedgedGraph) -> Self {
        let mut graph = biedged_graph.clone();

        let mut projection = Projection::new_for_biedged_graph(&graph);

        // Contract gray edges
        biedged_to_cactus::contract_all_gray_edges(&mut graph, &mut projection);

        // Find 3EC components
        let components =
            biedged_to_cactus::find_3_edge_connected_components(&graph);

        // Merge 3EC components
        biedged_to_cactus::merge_components(
            &mut graph,
            components,
            &mut projection,
        );
        // Find cycles

        let cycles = biedged_to_cactus::find_cycles(&graph);

        let mut cycle_map: FnvHashMap<(u64, u64), Vec<usize>> =
            FnvHashMap::default();

        for (i, cycle) in cycles.iter().enumerate() {
            for &(a, b) in cycle.iter() {
                cycle_map.entry((a, b)).or_default().push(i);
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

    fn black_edge_cycle(&self, x: u64) -> Option<&Vec<usize>> {
        let (l, r) = end_to_black_edge(x);
        let p_l = self.projection.find(l);
        let p_r = self.projection.find(r);
        let intersection = self.cycle_map.get(&(p_r, p_l))?;
        Some(&intersection)
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

pub struct CactusTree<'a> {
    pub original_graph: &'a BiedgedGraph,
    pub cactus_graph: &'a CactusGraph<'a>,
    pub graph: BiedgedGraph,
    pub chain_vertices: Vec<(u64, usize)>,
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

        let chain_vertices =
            biedged_to_cactus::build_cactus_tree(&mut graph, &cycles);

        Self {
            original_graph: cactus_graph.original_graph,
            graph,
            chain_vertices,
            cactus_graph,
        }
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

    pub fn chain_edges_(&self) -> FnvHashMap<u64, (Vec<u64>, usize)> {
        // maps net vertices to chain vertices & length of cycle
        let mut chain_edges_help: FnvHashMap<u64, (Vec<u64>, usize)> =
            FnvHashMap::default();

        for (chain_ix, x) in self.chain_vertices.iter() {
            let cycle = &self.cactus_graph.cycles[*x];
            let cycle_len = cycle.len();

            for (p_x, _) in cycle {
                let ix = *p_x;
                if let Some(ref mut entry) = chain_edges_help.get_mut(&ix) {
                    if cycle_len < entry.1 {
                        entry.1 = cycle_len;
                        entry.0 = vec![*chain_ix];
                    } else if cycle_len == entry.1 {
                        entry.0.push(*chain_ix);
                    }
                } else {
                    chain_edges_help.insert(ix, (vec![*chain_ix], cycle.len()));
                }
            }
        }

        chain_edges_help
    }

    pub fn chain_edges(&self) -> Vec<(u64, u64)> {
        let mut chain_edges = Vec::new();
        for (chain_ix, x) in self.chain_vertices.iter() {
            let cycle = &self.cactus_graph.cycles[*x];
            println!("chain {}\t{}  {:?}", chain_ix, x, cycle);
            chain_edges.extend(
                self.base_graph().edges(*chain_ix).map(|(a, b, _)| (a, b)),
            );
        }
        chain_edges
    }

    pub fn bridge_edges(&self) -> Vec<(u64, u64)> {
        self.original_graph
            .graph
            .all_edges()
            .filter_map(|(a, b, w)| {
                if w.black > 0 && self.is_bridge_edge(a, b) {
                    Some((a, b))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_chain_vertex_cycle(
        &self,
        chain_ix: u64,
    ) -> Option<&[(u64, u64)]> {
        let be_graph = self.biedged_graph();

        if be_graph.is_chain_vertex(chain_ix) {
            let ix = chain_ix - self.graph.max_net_vertex - 1;
            let chain_vx = &self.chain_vertices[ix as usize];
            let cycle = &self.cactus_graph.cycles[chain_vx.1];

            Some(cycle)
        } else {
            None
        }
    }

    pub fn find_chain_pairs(&self) -> FnvHashMap<(u64, u64), Vec<usize>> {
        let mut chain_pairs: FnvHashMap<(u64, u64), Vec<usize>> =
            FnvHashMap::default();

        let chain_edges = self.chain_edges();

        let cactus_graph_inverse =
            self.cactus_graph.projection.get_inverse().unwrap();

        for (chain_ix, _) in chain_edges.iter() {
            // println!(" in chain edge {}", chain_ix);
            let cycle = self.get_chain_vertex_cycle(*chain_ix).unwrap();
            // println!(" ---- \t{:?}", cycle);

            for (x, y) in cycle.iter() {
                let orig_xs = cactus_graph_inverse.get(&x).unwrap();

                // println!("         \t{:?}", orig_xs);
                let filtered: Vec<_> = orig_xs
                    .iter()
                    .filter(|&&u| {
                        let (v, w) = end_to_black_edge(u as u64);
                        if orig_xs.contains(&w) && orig_xs.contains(&v) {
                            false
                        } else {
                            true
                        }
                    })
                    .copied()
                    .collect();

                // println!("   filtered\t{:?}", filtered);

                for x_a in filtered.iter() {
                    for x_b in filtered.iter() {
                        if x_a != x_b {
                            let (x_a, x_b) = (*x_a as u64, *x_b as u64);
                            let a = x_a.min(x_b);
                            let b = x_a.max(x_b);
                            let is_chain_pair =
                                self.cactus_graph.is_chain_pair(a, b);
                            // println!(
                            //     "{} {} chain pair? {}",
                            //     a, b, is_chain_pair
                            // );
                            if is_chain_pair {
                                chain_pairs
                                    .entry((a, b))
                                    .or_default()
                                    .push(*chain_ix as usize);
                            }
                        }
                    }
                }
            }
        }

        chain_pairs
    }

    pub fn build_net_graph(&self, x: u64, y: u64) -> Option<NetGraph> {
        use biedged_to_cactus::{
            net_graph_black_edge_walk, snarl_cactus_tree_path,
        };

        let orig_graph = self.original_graph;

        let path = snarl_cactus_tree_path(
            &self.graph,
            &self.cactus_graph.projection,
            x,
            y,
        )?;

        let proj_inv = self.cactus_graph.projection.get_inverse()?;

        let tree_graph = &self.graph;

        let vertices: FnvHashSet<u64> = path
            .into_iter()
            .filter_map(|n| {
                if tree_graph.is_net_vertex(n) {
                    proj_inv.get(&n)
                } else {
                    None
                }
            })
            .flatten()
            .copied()
            .collect();

        let gray_edges: FnvHashSet<(u64, u64)> = vertices
            .iter()
            .flat_map(|v| orig_graph.graph.edges(*v))
            .filter_map(|(v, n, w)| {
                if vertices.contains(&n) && w.gray > 0 {
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
                        if net_graph_black_edge_walk(orig_graph, *v, *u) {
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

        let mut net_graph = BiedgedGraph::new();
        for v in vertices.iter() {
            net_graph.add_node(*v);
        }

        for (a, b) in gray_edges.iter() {
            net_graph.add_edge(*a, *b, BiedgedWeight::gray(1));
        }

        for (a, b) in black_edges.iter() {
            net_graph.add_edge(*a, *b, BiedgedWeight::black(1));
        }

        Some(NetGraph {
            graph: net_graph,
            x,
            y,
        })
    }

    pub fn is_chain_pair_ultrabubble(
        &self,
        // labels: &mut FnvHashMap<u64, bool>,
        labels: &mut FnvHashMap<(u64, u64, u64), bool>,
        x: u64,
        y: u64,
        chain_vx: u64,
    ) -> bool {
        if !self.cactus_graph.is_chain_pair(x, y) {
            return false;
        }
        let p_x = self.projected_node(x);

        // let chain_edge =

        if let Some(is_ultrabubble) = labels.get(&(x, y, chain_vx)) {
            if !is_ultrabubble {
                // println!("\t{} {} is not ultrabubble", x, y);
                // println!("\t{} {} is not ultrabubble", p_x, chain_vx);
                return false;
            }
        }

        // let p_y = self.projected_node(y);

        let mut visited: FnvHashSet<u64> = FnvHashSet::default();
        // visited.insert(p_x);

        if self
            .graph
            .graph
            .neighbors(chain_vx)
            .filter(|&n| n != p_x)
            .count()
            == 0
        {
            if let Some(is_ultrabubble) = labels.get(&(x, y, chain_vx)) {
                return *is_ultrabubble;
            } else {
                return false;
            }
        }

        let mut stack: Vec<(u64, u64)> = Vec::new();
        visited.insert(chain_vx);
        stack.push((chain_vx, p_x));

        let mut last_chain = chain_vx;

        // if self.graph.graph.neighbors(

        //         for (_, adj, _) in self.graph.graph.edges(current) {

        // println!("\nultrabubble {} {} - {}", x, y, chain_vx);
        // println!("projected {}", p_x);
        while let Some((prev, current)) = stack.pop() {
            // println!(" at node {}", current);
            if !visited.contains(&current) {
                let a = prev.min(current);
                let b = prev.max(current);
                if let Some(is_ultrabubble) = labels.get(&(a, b, last_chain)) {
                    // println!("at chain label {}\t{}", current, is_ultrabubble);
                    if !is_ultrabubble {
                        // labels.insert(chain_vx, false);
                        return false;
                    }
                }

                if self.graph.is_chain_vertex(current) {
                    last_chain = current;
                }
                visited.insert(current);

                for (_, adj, _) in self.graph.graph.edges(current) {
                    if !visited.contains(&adj) {
                        stack.push((current, adj));
                    }
                }
            }
        }

        true
    }
}

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

        biedged_to_cactus::contract_simple_cycles(
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

    pub fn find_bridge_pairs(&self) -> FnvHashSet<(u64, u64)> {
        let mut bridge_pairs: FnvHashSet<(u64, u64)> = FnvHashSet::default();
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
                            bridge_pairs.insert((x_, y_));
                        }
                    }
                }
            }
        }

        bridge_pairs
    }
}
