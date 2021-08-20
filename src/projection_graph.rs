use handlegraph::{disjoint::*, packedgraph::PackedGraph};

use std::sync::Arc;

#[derive(Clone)]
pub struct ProjectedPackedGraph {
    graph: Arc<PackedGraph>,
    // projection: Projection
}

pub struct Projection {
    pub size: usize,
    disj: DisjointSets,
    inverse: Vec<u64>,
}
