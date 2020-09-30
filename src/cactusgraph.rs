use petgraph::prelude::*;

use crate::{
    biedged_to_cactus,
    biedgedgraph::{BiedgedGraph, BiedgedWeight},
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

        CactusGraph {
            original_graph: biedged_graph,
            graph,
            projection,
            cycles,
        }
    }
}

pub struct CactusTree<'a> {
    pub original_graph: &'a BiedgedGraph,
    pub graph: BiedgedGraph,
    pub projection: Projection,
    pub cycles: Vec<Vec<(u64, u64)>>,
    pub chain_vertices: Vec<(u64, usize)>,
}

impl_biedged_wrapper!(CactusTree<'a>);

impl<'a> CactusTree<'a> {
    pub fn from_cactus_graph(cactus_graph: &'_ CactusGraph<'a>) -> Self {
        let mut graph = cactus_graph.graph.clone();
        let projection = cactus_graph.projection.clone();

        let cycles = cactus_graph.cycles.clone();

        let chain_vertices =
            biedged_to_cactus::build_cactus_tree(&mut graph, &cycles);

        Self {
            original_graph: cactus_graph.original_graph,
            graph,
            projection,
            cycles,
            chain_vertices,
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

    // pub fn build_net_graph(
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
        let mut projection = cactus_graph.projection.clone();

        biedged_to_cactus::contract_simple_cycles(
            &mut graph,
            &cactus_graph.cycles,
            &mut projection,
        );

        Self {
            original_graph: cactus_graph.original_graph,
            graph,
            projection,
        }
    }
}
