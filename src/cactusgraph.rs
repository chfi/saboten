use fnv::{FnvHashMap, FnvHashSet};
use petgraph::prelude::*;

use crate::{
    biedged_to_cactus,
    biedgedgraph::{
        end_to_black_edge, opposite_vertex, BiedgedGraph, BiedgedWeight,
    },
    netgraph::NetGraph,
    projection::Projection,
    ultrabubble::{Snarl, Ultrabubble},
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
    /*


    */

    pub fn black_edge_cycle(&self, x: u64) -> Option<&Vec<usize>> {
        let (l, r) = end_to_black_edge(x);
        let p_l = self.projection.find(l);
        let p_r = self.projection.find(r);
        let cycles = self.cycle_map.get(&(p_r, p_l))?;
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
            biedged_to_cactus::build_cactus_tree(&mut graph, &cycles);

        Self {
            original_graph: cactus_graph.original_graph,
            graph,
            cycle_chain_map,
            chain_vertices,
            cactus_graph,
        }
    }

    pub fn black_edge_chain_vertex(&self, b: u64) -> Option<u64> {
        let (l, r) = end_to_black_edge(b);
        let p_l = self.projected_node(l);
        let p_r = self.projected_node(r);
        let chain_vx = self.cycle_chain_map.get(&(p_r, p_l))?;
        Some(*chain_vx)
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
                            let c_a = self.black_edge_chain_vertex(a);
                            let c_b = self.black_edge_chain_vertex(b);
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
            .iter()
            .filter_map(|&n| {
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
            path,
        })
    }

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

    pub fn is_chain_pair_ultrabubble(
        &self,
        labels: &mut FnvHashMap<(u64, u64), bool>,
        x: u64,
        y: u64,
        chain_vx: u64,
    ) -> Option<Vec<(u64, u64)>> {
        if !self.cactus_graph.is_chain_pair(x, y) {
            return None;
        }

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

pub fn chain_pair_ultrabubble_labels(
    cactus_tree: &CactusTree<'_>,
    chain_pairs: &FnvHashSet<Snarl>,
) -> FnvHashMap<(u64, u64), bool> {
    let mut chain_edge_labels = FnvHashMap::default();

    let mut net_graphs: FnvHashMap<(u64, u64), NetGraph> =
        FnvHashMap::default();

    for &snarl in chain_pairs.iter() {
        if let Snarl::ChainPair { x, y } = snarl {
            let net = cactus_tree.projected_node(x);
            let chain = cactus_tree.black_edge_chain_vertex(x).unwrap();
            let key = (net, chain);

            if let Some(is_ultrabubble) = chain_edge_labels.get(&key) {
            } else {
                let net_graph = if let Some(net_graph) =
                    net_graphs.get(&(net, chain))
                {
                    net_graph
                } else {
                    let net_graph = cactus_tree.build_net_graph(x, y).unwrap();
                    net_graphs.insert(key, net_graph);
                    net_graphs.get(&key).unwrap()
                };

                let result = net_graph.is_ultrabubble();

                chain_edge_labels.insert(key, result);
            }
        }
    }

    chain_edge_labels
}

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

pub fn bridge_pair_ultrabubbles(
    cactus_tree: &CactusTree<'_>,
    bridge_pairs: &FnvHashSet<Snarl>,
    chain_edge_labels: &FnvHashMap<(u64, u64), bool>,
) -> FnvHashMap<(u64, u64), Vec<(u64, u64)>> {
    let mut bridge_pair_ultrabubbles = FnvHashMap::default();

    for &snarl in bridge_pairs.iter() {
        if let Snarl::BridgePair { x, y } = snarl {
            let net_graph = cactus_tree.build_net_graph(x, y).unwrap();

            if net_graph.is_ultrabubble() {
                let contained_chain_pairs = cactus_tree
                    .is_bridge_pair_ultrabubble(
                        &chain_edge_labels,
                        x,
                        y,
                        &net_graph.path,
                    );

                if let Some(children) = contained_chain_pairs {
                    bridge_pair_ultrabubbles.insert((x, y), children);
                }
            }
        }
    }

    bridge_pair_ultrabubbles
}

pub fn find_ultrabubbles(
    cactus_tree: &CactusTree<'_>,
    bridge_forest: &BridgeForest<'_>,
) -> FnvHashMap<(u64, u64), Vec<(u64, u64)>> {
    let chain_pairs = cactus_tree.find_chain_pairs();
    let bridge_pairs = bridge_forest.find_bridge_pairs();

    let mut chain_edge_labels =
        chain_pair_ultrabubble_labels(cactus_tree, &chain_pairs);

    let chain_ultrabubbles = chain_pair_contained_ultrabubbles(
        cactus_tree,
        &chain_pairs,
        &mut chain_edge_labels,
    );

    let bridge_ultrabubbles = bridge_pair_ultrabubbles(
        cactus_tree,
        &bridge_pairs,
        &chain_edge_labels,
    );

    let ultrabubbles = chain_ultrabubbles
        .into_iter()
        .chain(bridge_ultrabubbles.into_iter())
        .collect::<FnvHashMap<(u64, u64), Vec<(u64, u64)>>>();

    ultrabubbles
}
