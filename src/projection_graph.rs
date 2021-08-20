use handlegraph::{
    disjoint::*,
    handle::{Edge, Handle},
    handlegraph::{HandleGraph, IntoEdges, IntoHandles},
    packedgraph::PackedGraph,
};

use std::sync::Arc;

pub struct BiedgedGraph {
    graph: Arc<PackedGraph>,
    projection: Projection,

    has_gray_edges: bool,
}

impl BiedgedGraph {
    pub fn new(graph: PackedGraph) -> Self {
        let graph = Arc::new(graph);
        let projection = Projection::new(graph.node_count());

        let has_gray_edges = true;

        Self {
            graph,
            projection,
            has_gray_edges,
        }
    }

    pub fn black_edges_iter(&self) -> impl Iterator<Item = Handle> + '_ {
        self.graph.handles()
    }

    pub fn gray_edges_iter(
        &self,
    ) -> Option<impl Iterator<Item = (BNode, BNode)> + '_> {
        if self.has_gray_edges {
            // TODO this has to be the correct mapping
            let iter = self
                .graph
                .edges()
                .map(|Edge(a, b)| (BNode::from(a), BNode::from(b)));
            todo!();
            Some(iter)
        } else {
            None
        }
    }

    pub fn contract_gray_edges(&mut self) {
        if self.has_gray_edges {
            let gray_edges = self.gray_edges_iter().unwrap();

            for (a, b) in gray_edges {
                self.contract_gray_edge(a, b);
            }

            self.has_gray_edges = false;
        }
    }

    fn contract_gray_edge(&self, x: BNode, y: BNode) {
        self.projection.unite(x, y);
    }

    fn merge_vertices(&self, mut vertices: impl Iterator<Item = BNode>) {
        let mut prev = if let Some(first) = vertices.next() {
            first
        } else {
            return;
        };

        for cur in vertices {
            self.projection.unite(prev, cur);
            prev = cur;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BNode(u64);

impl From<Handle> for BNode {
    fn from(h: Handle) -> Self {
        let id = h.id().0;
        let ix = (id - 1) / 2;
        if h.is_reverse() {
            BNode(ix)
        } else {
            BNode(ix + 1)
        }
    }
}

pub struct Projection {
    pub size: usize,
    disj: DisjointSets,
}

impl Projection {
    pub fn new(node_count: usize) -> Self {
        let size = node_count;

        let disj = DisjointSets::new(size);

        Self { size, disj }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn find_handle(&self, handle: Handle) -> BNode {
        let bn = BNode::from(handle);
        let projected = self.disj.find(bn.0);
        BNode(projected)
    }

    pub fn find_biedged_node(&self, node: BNode) -> BNode {
        BNode(self.disj.find(node.0))
    }

    pub fn same(&self, x: BNode, y: BNode) -> bool {
        self.disj.same(x.0, y.0)
    }

    pub fn unite(&self, x: BNode, y: BNode) -> BNode {
        BNode(self.disj.unite(x.0, y.0))
    }
}
