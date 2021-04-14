use log::{debug, trace};

use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Biedged {}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cactus {}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bridge {}

pub trait GraphType {}

impl GraphType for Biedged {}
impl GraphType for Cactus {}
impl GraphType for Bridge {}

/// A node index for a biedged graph of the specified type
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node<G> {
    _graph: std::marker::PhantomData<G>,
    id: u64,
}

impl<G> Node<G> {
    #[inline]
    pub fn new_for<H>(id: u64) -> Node<H> {
        Node {
            _graph: std::marker::PhantomData,
            id,
        }
    }

    #[inline]
    pub fn new(id: u64) -> Self {
        Node {
            _graph: std::marker::PhantomData,
            id,
        }
    }

    #[inline]
    pub fn map_graph_type<H>(&self) -> Node<H> {
        Node {
            _graph: std::marker::PhantomData,
            id: self.id,
        }
    }

    /// Derive the node IDs for a black edge in a biedged graph, given
    /// a node ID in a GFA graph
    #[inline]
    pub fn from_gfa_id(id: u64) -> (Self, Self) {
        let left = id * 2;
        let right = left + 1;

        (Self::new(left), Self::new(right))
    }

    /// Derive the original GFA ID for the provided black edge node ID
    #[inline]
    pub fn to_gfa_id(&self) -> u64 {
        self.id / 2
    }

    /// Derive the pair of node IDs in the black edge defined by this node
    #[inline]
    pub fn black_edge(&self) -> (Self, Self) {
        let left = self.id & !1;
        let right = left + 1;

        (Self::new(left), Self::new(right))
    }

    /// Return the opposite node
    #[inline]
    pub fn opposite(&self) -> Self {
        Self {
            id: self.id ^ 1,
            ..*self
        }
    }

    #[inline]
    pub fn is_left(&self) -> bool {
        self.id & 1 == 0
    }

    #[inline]
    pub fn is_right(&self) -> bool {
        self.id & 1 != 0
    }
}
