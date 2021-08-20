use handlegraph::{disjoint::*, handle::Handle, packedgraph::PackedGraph};

use std::sync::Arc;

pub struct ProjectedPackedGraph {
    graph: Arc<PackedGraph>,
    projection: Projection,
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
