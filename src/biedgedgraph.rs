use petgraph::prelude::*;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use gfa::gfa::{Orientation, GFA};

use crate::projection::{id_to_black_edge, Projection};

use log::{debug, trace};

/// To make a petgraph Graph(Map) into a multigraph, we track the
/// number of black and gray edges between two nodes by using this
/// struct as the edge weight type.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BiedgedWeight {
    pub black: usize,
    pub gray: usize,
}

impl BiedgedWeight {
    /// An empty weight has zero edges of either color.
    #[inline]
    pub fn empty() -> Self {
        Default::default()
    }

    /// Construct a new edge weight with the provided edge counts.
    #[inline]
    pub fn new(black: usize, gray: usize) -> Self {
        BiedgedWeight { black, gray }
    }

    /// Construct a new edge weight with the provided black count,
    /// with gray set to zero.
    #[inline]
    pub fn black(black: usize) -> Self {
        BiedgedWeight { black, gray: 0 }
    }

    /// Construct a new edge weight with the provided gray count,
    /// with black set to zero.
    #[inline]
    pub fn gray(gray: usize) -> Self {
        BiedgedWeight { black: 0, gray }
    }
}

/// Adding two BiedgedWeights adds their corresponding edges, which
/// makes it easy to move black edges when contracting gray edges.
impl Add for BiedgedWeight {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            black: self.black + other.black,
            gray: self.gray + other.gray,
        }
    }
}

impl AddAssign for BiedgedWeight {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.black += other.black;
        self.gray += other.gray;
    }
}

impl Sub for BiedgedWeight {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            black: self.black - other.black,
            gray: self.gray - other.gray,
        }
    }
}

impl SubAssign for BiedgedWeight {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.black -= other.black;
        self.gray -= other.gray;
    }
}

/// A biedged graph is a graph with two types of edges: black edges
/// and gray edges, such that each vertex is incident with at most one
/// black edge.

/// To simplify differentiating between net vertices and chain
/// vertices in the cactus graph, all chain vertices have an index
/// higher than the original vertices. This also makes it easier to
/// track the projections.
#[derive(Default, Clone)]
pub struct BiedgedGraph {
    pub graph: UnGraphMap<u64, BiedgedWeight>,
    pub max_net_vertex: u64,
    pub max_chain_vertex: u64,
    pub id_offset: u64,
}

impl BiedgedGraph {
    pub fn shrink_to_fit(&mut self) {
        let (node_count, node_cap) = self.node_count_capacity();
        let (edge_count, edge_cap) = self.edge_count_capacity();

        if node_count == node_cap && edge_count == edge_cap {
            return;
        }

        trace!(
            "shrink_to_fit - node count & cap: {} | {}",
            node_count,
            node_cap
        );
        trace!(
            "shrink_to_fit - edge count & cap: {} | {}",
            edge_count,
            edge_cap
        );

        let mut new_graph: UnGraphMap<u64, BiedgedWeight> =
            UnGraphMap::with_capacity(node_count, edge_count);

        for (a, b, &w) in self.graph.all_edges() {
            new_graph.add_edge(a, b, w);
        }

        let node_count = new_graph.node_count();
        let edge_count = new_graph.edge_count();
        let (node_cap, edge_cap) = new_graph.capacity();

        trace!(
            "shrink_to_fit - new node count & cap: {} | {}",
            node_count,
            node_cap
        );
        trace!(
            "shrink_to_fit - new edge count & cap: {} | {}",
            edge_count,
            edge_cap
        );

        std::mem::swap(&mut self.graph, &mut new_graph);
    }

    pub fn shrink_into(self) -> BiedgedGraph {
        let (node_count, node_cap) = self.node_count_capacity();
        let (edge_count, edge_cap) = self.edge_count_capacity();

        if node_count == node_cap && edge_count == edge_cap {
            return self;
        }

        trace!(
            "shrink_into - node count & cap: {} | {}",
            node_count,
            node_cap
        );
        trace!(
            "shrink_into - edge count & cap: {} | {}",
            edge_count,
            edge_cap
        );

        let mut new_graph: UnGraphMap<u64, BiedgedWeight> =
            UnGraphMap::with_capacity(node_count, edge_count);

        for (a, b, &w) in self.graph.all_edges() {
            new_graph.add_edge(a, b, w);
        }

        let node_count = new_graph.node_count();
        let edge_count = new_graph.edge_count();
        let (node_cap, edge_cap) = new_graph.capacity();

        trace!(
            "shrink_into - new node count & cap: {} | {}",
            node_count,
            node_cap
        );
        trace!(
            "shrink_into - new edge count & cap: {} | {}",
            edge_count,
            edge_cap
        );

        BiedgedGraph {
            graph: new_graph,
            ..self
        }
    }

    pub fn shrink_clone(&self) -> BiedgedGraph {
        let (node_count, node_cap) = self.node_count_capacity();
        let (edge_count, edge_cap) = self.edge_count_capacity();

        if node_count == node_cap && edge_count == edge_cap {
            return self.clone();
        }

        trace!(
            "shrink_clone - node count & cap: {} | {}",
            node_count,
            node_cap
        );
        trace!(
            "shrink_clone - edge count & cap: {} | {}",
            edge_count,
            edge_cap
        );

        let mut new_graph: UnGraphMap<u64, BiedgedWeight> =
            UnGraphMap::with_capacity(node_count, edge_count);

        for (a, b, &w) in self.graph.all_edges() {
            new_graph.add_edge(a, b, w);
        }

        let node_count = new_graph.node_count();
        let edge_count = new_graph.edge_count();
        let (node_cap, edge_cap) = new_graph.capacity();

        trace!(
            "shrink_clone - new node count & cap: {} | {}",
            node_count,
            node_cap
        );
        trace!(
            "shrink_clone - new edge count & cap: {} | {}",
            edge_count,
            edge_cap
        );

        BiedgedGraph {
            graph: new_graph,
            ..*self
        }
    }

    /// Create an empty biedged graph
    #[inline]
    pub fn new() -> BiedgedGraph {
        Default::default()
    }

    /// Adds a chain vertex, ensuring that it has an index higher than
    /// any net vertex. Returns the new vertex identifier.
    #[inline]
    pub fn add_chain_vertex(&mut self) -> u64 {
        self.max_chain_vertex += 1;
        let id = self.max_chain_vertex;
        self.graph.add_node(id);
        id
    }

    #[inline]
    pub fn is_chain_vertex(&self, n: u64) -> bool {
        n > self.max_net_vertex
    }

    #[inline]
    pub fn is_net_vertex(&self, n: u64) -> bool {
        n <= self.max_net_vertex
    }

    /// Convenience method for getting the projection of a node,
    /// taking the possibility of chain vertices into account
    #[inline]
    pub fn projected_node(&self, projection: &Projection, n: u64) -> u64 {
        if n <= self.max_net_vertex {
            projection.find(n)
        } else {
            n
        }
    }

    pub fn from_directed_edges<I>(i: I) -> Option<BiedgedGraph>
    where
        I: IntoIterator<Item = (u64, u64)>,
    {
        use Orientation::Forward as F;
        let iter = i.into_iter().map(|(a, b)| (a, F, b, F));
        Self::from_bidirected_edges(iter)
    }

    /// Consume an iterator of directed edges to produce a biedged
    /// graph. The edges should be tuples of the form (from, to),
    /// where the elements are node IDs, and each node ID must be in
    /// the range 0..N, where N is the number of nodes.
    pub fn from_bidirected_edges<I>(i: I) -> Option<BiedgedGraph>
    where
        I: IntoIterator<Item = (u64, Orientation, u64, Orientation)>,
    {
        use Orientation::*;

        let mut min_node_id = std::u64::MAX;
        let mut max_node_id = std::u64::MIN;

        let mut graph: UnGraphMap<u64, BiedgedWeight> = UnGraphMap::new();

        for (a, a_o, b, b_o) in i {
            min_node_id = min_node_id.min(a.min(b));
            max_node_id = max_node_id.max(a.max(b));

            let (a_l, a_r) = id_to_black_edge(a);
            let (b_l, b_r) = id_to_black_edge(b);

            if !graph.contains_edge(a_l, a_r) {
                graph.add_edge(a_l, a_r, BiedgedWeight::black(1));
            }

            if !graph.contains_edge(b_l, b_r) {
                graph.add_edge(b_l, b_r, BiedgedWeight::black(1));
            }

            let (left, right) = match (a_o, b_o) {
                (Forward, Forward) => (a_r, b_l),
                (Backward, Backward) => (b_r, a_l),
                (Forward, Backward) => (a_r, b_r),
                (Backward, Forward) => (a_l, b_l),
            };

            if let Some(w) = graph.edge_weight_mut(left, right) {
                *w += BiedgedWeight::gray(1);
            } else {
                graph.add_edge(left, right, BiedgedWeight::gray(1));
            }
        }

        let max_net_vertex = (max_node_id + 1) * 2;
        let max_chain_vertex = max_net_vertex;

        assert_eq!(min_node_id, 0);
        assert_eq!(max_net_vertex, graph.node_count() as u64);

        Some(BiedgedGraph {
            graph,
            max_net_vertex,
            max_chain_vertex,
            id_offset: 0,
        })
    }

    /// Construct a biedged graph from a GFA.
    pub fn from_gfa(gfa: &GFA<usize, ()>) -> BiedgedGraph {
        debug!(
            "building BiedgedGraph from GFA with {} nodes, {} edges",
            gfa.segments.len(),
            gfa.links.len()
        );

        let segs_len = gfa.segments.len();
        let links_len = gfa.links.len();

        let mut be_graph: UnGraphMap<u64, BiedgedWeight> =
            UnGraphMap::with_capacity(segs_len * 2, segs_len + links_len);

        let mut max_seg_id = 0;
        let mut min_seg_id = std::usize::MAX;

        let mut min_node_id = std::u64::MAX;
        let mut max_node_id = 0;

        for segment in gfa.segments.iter() {
            // let (left, right) = id_to_black_edge(segment.name as u64);

            max_seg_id = segment.name.max(max_seg_id);
            min_seg_id = segment.name.min(min_seg_id);

            // be_graph.add_node(left);
            // be_graph.add_node(right);
            // be_graph.add_edge(left, right, BiedgedWeight::black(1));
        }

        // let id_offset = min_seg_id as u64;
        let id_offset = if min_seg_id == 0 {
            0
        } else {
            (min_seg_id - 1) as u64
        };

        for segment in gfa.segments.iter() {
            let seg_id = segment.name - min_seg_id;
            let (left, right) = id_to_black_edge(seg_id as u64);

            // max_node_id = max_node_id.max(segment.name);
            // max_seg_id = segment.name.max(max_seg_id);
            // min_seg_id = segment.name.min(min_seg_id);

            // let left = left - id_offset;
            // let right = right - id_offset;

            max_node_id = max_node_id.max(left).max(right);
            min_node_id = min_node_id.min(left).min(right);

            be_graph.add_node(left);
            be_graph.add_node(right);
            be_graph.add_edge(left, right, BiedgedWeight::black(1));
        }

        use Orientation::*;

        for link in gfa.links.iter() {
            let from_o = link.from_orient;
            let to_o = link.to_orient;

            let from_seg = (link.from_segment as u64) - id_offset;
            let to_seg = (link.to_segment as u64) - id_offset;

            let from = id_to_black_edge(from_seg);
            let to = id_to_black_edge(to_seg);

            // let from = id_to_black_edge(link.from_segment as u64);
            // let to = id_to_black_edge(link.to_segment as u64);

            let (left, right) = match (from_o, to_o) {
                (Forward, Forward) => (from.1, to.0),
                (Backward, Backward) => (to.1, from.0),
                (Forward, Backward) => (from.1, to.1),
                (Backward, Forward) => (from.0, to.0),
            };

            if let Some(w) = be_graph.edge_weight_mut(left, right) {
                *w += BiedgedWeight::gray(1);
            } else {
                be_graph.add_edge(left, right, BiedgedWeight::gray(1));
            }
        }

        // let max_node_id = max_node_id as u64;
        let max_seg_id = max_seg_id as u64;
        let min_seg_id = min_seg_id as u64;
        let max_net_vertex = max_node_id - min_node_id;

        println!("min_node_id: {}\tmax_node_id: {}", min_node_id, max_node_id);

        let max_chain_vertex = max_net_vertex;

        let (node_cap, edge_cap) = be_graph.capacity();

        trace!("BiedgedGraph with {} nodes, {} edges, capacity: {} nodes, {} edges",
               be_graph.node_count(), be_graph.edge_count(), node_cap, edge_cap);

        BiedgedGraph {
            graph: be_graph,
            max_net_vertex,
            max_chain_vertex,
            id_offset,
        }
    }

    /// Add the node with the given id to the graph
    #[inline]
    pub fn add_node(&mut self, id: u64) -> u64 {
        self.graph.add_node(id)
    }

    /// Add an edge with the provided edge weight. If a corresponding
    /// edge already exists in the graph, the edge weights are added.
    #[inline]
    pub fn add_edge(&mut self, from: u64, to: u64, weight: BiedgedWeight) {
        if let Some(old) = self.graph.edge_weight_mut(from, to) {
            *old += weight;
        } else {
            self.graph.add_edge(from, to, weight);
        }
    }

    /// Returns an iterator over the gray edges in the graph, where
    /// the first two elements in the tuple are the `from` and `to`
    /// nodes, and the third is the weight containing the number of
    /// gray and black edges between the two nodes.
    #[inline]
    pub fn gray_edges(
        &self,
    ) -> impl Iterator<Item = (u64, u64, &BiedgedWeight)> {
        self.graph.all_edges().filter(|(_, _, w)| w.gray > 0)
    }

    /// Convenience method for looping through all gray edges while
    /// mutating the graph
    #[inline]
    pub fn next_gray_edge(&self) -> Option<(u64, u64)> {
        self.graph
            .all_edges()
            .find(|(_, _, w)| w.gray > 0)
            .map(|x| (x.0, x.1))
    }

    /// Returns an iterator over the black edges in the graph, where
    /// the first two elements in the tuple are the `from` and `to`
    /// nodes, and the third is the weight containing the number of
    /// gray and black edges between the two nodes.
    #[inline]
    pub fn black_edges(
        &self,
    ) -> impl Iterator<Item = (u64, u64, &BiedgedWeight)> {
        self.graph.all_edges().filter(|(_, _, w)| w.black > 0)
    }

    /// Produces the sum of the gray edges in the graph, counted using
    /// the edge weights.
    #[inline]
    pub fn gray_edge_count(&self) -> usize {
        self.gray_edges().map(|(_, _, w)| w.gray).sum()
    }

    /// Produces the sum of the black edges in the graph, counted using
    /// the edge weights.
    #[inline]
    pub fn black_edge_count(&self) -> usize {
        self.black_edges().map(|(_, _, w)| w.black).sum()
    }

    /// Remove a single black edge between two nodes, if any exists.
    /// If the nodes share more than one black edge, their
    /// corresponding edge weight is decremented, but they will still
    /// have an edge in the graph.
    #[inline]
    pub fn remove_one_black_edge(&mut self, a: u64, b: u64) -> Option<usize> {
        use std::cmp::Ordering;

        let weight = self.graph.edge_weight_mut(a, b)?;

        match weight.black.cmp(&1) {
            Ordering::Greater => {
                weight.black -= 1;
                Some(weight.black)
            }
            Ordering::Equal => {
                self.graph.remove_edge(a, b);
                Some(0)
            }
            Ordering::Less => None,
        }
    }

    /// Merge two vertices into one, such that all the edges incident
    /// to the provided nodes are moved to be incident to the merged
    /// vertex.
    ///
    /// Returns the index of the resulting vertex, or None if either
    /// of the provided vertices were not present in the graph.
    #[inline]
    pub fn merge_vertices(
        &mut self,
        from: u64,
        to: u64,
        projection: &mut Projection,
    ) -> Option<u64> {
        projection.union(from, to);
        let (from, to) = projection.kept_pair(from, to);
        if !self.graph.contains_node(from) || !self.graph.contains_node(to) {
            return None;
        }

        // Retrieve the edges of the node we're removing
        let to_edges: Vec<(u64, u64, BiedgedWeight)> = self
            .graph
            .edges(to)
            .filter(|(_, node, _)| node != &from && node != &to)
            .map(|(a, b, w)| (a, b, *w))
            .collect();

        self.graph.remove_node(to);

        // add the edges that were removed with the deleted node
        for (_, other, w) in to_edges {
            self.add_edge(from, other, w);
        }

        Some(from)
    }

    /// Contract a (gray) edge between two vertices.
    #[inline]
    pub fn contract_edge(
        &mut self,
        left: u64,
        right: u64,
        projection: &mut Projection,
    ) -> Option<u64> {
        projection.union(left, right);
        let (from, to) = projection.kept_pair(left, right);

        let weight = self.graph.edge_weight(from, to).copied()?;
        let other_self_weight = self.graph.edge_weight(to, to).copied();

        // Retrieve the edges of the node we're removing
        let to_edges: Vec<(u64, u64, BiedgedWeight)> = self
            .graph
            .edges(to)
            .filter(|(_, node, _)| node != &from && node != &to)
            .map(|(a, b, w)| (a, b, *w))
            .collect();

        self.graph.remove_node(to);

        // add the edges that were removed with the deleted node
        for (_, other, w) in to_edges {
            self.add_edge(from, other, w);
        }

        if weight.black > 0 {
            let mut new_weight = BiedgedWeight::black(weight.black);
            if from != to {
                let other_black =
                    other_self_weight.map(|w| w.black).unwrap_or_default();
                new_weight.black += other_black
            }
            self.add_edge(from, from, new_weight);
        }

        Some(from)
    }

    pub(crate) fn edge_count_capacity(&self) -> (usize, usize) {
        let count = self.graph.edge_count();
        let (_, cap) = self.graph.capacity();
        (count, cap)
    }

    pub(crate) fn node_count_capacity(&self) -> (usize, usize) {
        let count = self.graph.node_count();
        let (cap, _) = self.graph.capacity();
        (count, cap)
    }
}

// ----------------------------------- TESTS -------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn example_graph_2() -> BiedgedGraph {
        let edges = vec![
            (0, 1),
            (0, 13),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
            (7, 8),
            (7, 12),
            (8, 9),
            (8, 10),
            (9, 11),
            (10, 11),
            (11, 12),
            (12, 13),
        ];

        let graph = BiedgedGraph::from_directed_edges(edges).unwrap();

        graph
    }

    #[test]
    fn test_add_node() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        assert!(graph.graph.contains_node(10));
        assert!(graph.graph.node_count() == 1);
    }

    #[test]
    fn test_add_edge() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(0);
        graph.add_node(1);
        graph.add_node(2);

        graph.add_edge(0, 1, BiedgedWeight::black(1));
        assert!(graph.graph.contains_edge(0, 1));

        assert_eq!(graph.black_edge_count(), 1);
        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 0 }),
            graph.graph.edge_weight(0, 1)
        );

        graph.add_edge(1, 2, BiedgedWeight::gray(1));
        assert!(graph.graph.contains_edge(1, 2));
        assert_eq!(graph.gray_edge_count(), 1);

        assert_eq!(
            Some(&BiedgedWeight { black: 0, gray: 1 }),
            graph.graph.edge_weight(1, 2)
        );

        graph.add_edge(1, 2, BiedgedWeight::black(1));

        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 1 }),
            graph.graph.edge_weight(1, 2)
        );
    }

    #[test]
    fn contract_one_edge() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(0);
        graph.add_node(1);
        graph.add_node(2);
        graph.add_edge(0, 1, BiedgedWeight::black(1));
        graph.add_edge(0, 2, BiedgedWeight::gray(1));
        graph.add_edge(1, 2, BiedgedWeight::black(1));

        graph.max_net_vertex = graph.graph.node_count() as u64;

        let mut proj = Projection::new_for_biedged_graph(&graph);

        assert_eq!(None, graph.graph.edge_weight(0, 0));
        assert_eq!(
            Some(&BiedgedWeight { black: 0, gray: 1 }),
            graph.graph.edge_weight(0, 2)
        );
        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 0 }),
            graph.graph.edge_weight(0, 1)
        );
        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 0 }),
            graph.graph.edge_weight(1, 2)
        );

        graph.contract_edge(0, 1, &mut proj);

        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 0 }),
            graph.graph.edge_weight(0, 0)
        );
        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 1 }),
            graph.graph.edge_weight(0, 2)
        );
        assert_eq!(None, graph.graph.edge_weight(0, 1));
        assert_eq!(None, graph.graph.edge_weight(1, 2));

        assert!(graph.graph.contains_node(0));
        assert!(graph.graph.contains_node(2));
        assert!(!graph.graph.contains_node(1));

        assert!(graph.graph.edge_count() == 2);

        assert_eq!(graph.black_edge_count(), 2);
        assert_eq!(graph.gray_edge_count(), 1);

        assert!(proj.equiv(0, 1));

        for i in 2..=3 {
            assert!(!proj.equiv(0, i as u64));
        }
    }

    #[test]
    fn contract_multiple_edges() {
        let edges =
            vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (3, 0)];

        let mut graph = BiedgedGraph::from_directed_edges(edges).unwrap();
        let mut proj = Projection::new_for_biedged_graph(&graph);

        graph.contract_edge(1, 2, &mut proj);
        let (x, y) = proj.kept_pair(1, 2);

        // One of the two nodes were deleted
        assert!(graph.graph.contains_node(x));
        assert!(!graph.graph.contains_node(y));

        graph.contract_edge(4, 1, &mut proj);

        let (x_, y_) = proj.kept_pair(4, 1);

        // The kept node must be the same in both cases, as one node
        // was included in both contractions
        assert_eq!(x, x_);

        assert!(graph.graph.contains_node(x_));
        assert!(!graph.graph.contains_node(y_));
        assert!(!graph.graph.contains_node(y));

        let first_union: Vec<u64> = vec![1, 2, 4];

        // All combinations of contracted edges have the same projection
        assert!(proj.equiv(1, 2));
        assert!(proj.equiv(1, 4));
        assert!(proj.equiv(2, 4));

        let edges_vec = |g: &BiedgedGraph, x: u64| {
            g.graph
                .edges(x)
                .map(|(a, b, w)| (a, b, w.black, w.gray))
                .collect::<Vec<_>>()
        };

        let x = proj.find(4);
        let edges = edges_vec(&graph, x);

        assert_eq!(edges, vec![(1, 0, 1, 0), (1, 3, 1, 0), (1, 5, 1, 0)]);

        graph.contract_edge(7, 8, &mut proj);
        graph.contract_edge(0, 7, &mut proj);

        let second_union: Vec<u64> = vec![0, 7, 8];

        assert!(proj.equiv(0, 7));
        assert!(proj.equiv(7, 8));
        assert!(proj.equiv(0, 8));

        let x = proj.find(7);
        let edges = edges_vec(&graph, x);

        assert_eq!(
            edges,
            vec![(7, 6, 1, 0), (7, 9, 1, 0), (7, 10, 0, 1), (7, 1, 1, 0)]
        );

        graph.contract_edge(0, 1, &mut proj);

        let (x_2, y_2) = proj.kept_pair(8, 4);

        assert_eq!(x, x_2);

        assert!(graph.graph.contains_node(x_2));
        assert!(!graph.graph.contains_node(y));
        assert!(!graph.graph.contains_node(y_));
        assert!(!graph.graph.contains_node(y_2));

        // Now all nodes in the contracted edges have been unified
        for (a, b) in first_union.iter().zip(second_union.iter()) {
            let x = proj.find(*a);
            let y = proj.find(*b);
            assert_eq!(x, y);
        }
    }

    #[test]
    fn merge_two_vertices() {
        let edges =
            vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (3, 0)];

        let mut graph = BiedgedGraph::from_directed_edges(edges).unwrap();
        let mut proj = Projection::new_for_biedged_graph(&graph);

        graph.merge_vertices(7, 8, &mut proj);
        graph.merge_vertices(7, 9, &mut proj);

        let (x, _y) = proj.kept_pair(7, 9);

        let edges_vec = |g: &BiedgedGraph, x: u64| {
            g.graph
                .edges(x)
                .map(|(a, b, w)| (a, b, w.black, w.gray))
                .collect::<Vec<_>>()
        };

        graph.merge_vertices(0, 7, &mut proj);
        graph.merge_vertices(1, 7, &mut proj);

        let edges = edges_vec(&graph, x);

        assert_eq!(
            edges,
            vec![(7, 6, 1, 0), (7, 10, 0, 1), (7, 4, 0, 1), (7, 2, 0, 1)]
        );

        let merged: Vec<u64> = vec![0, 1, 7, 8, 9];

        for i in merged {
            let x = proj.find(i);
            if i == x {
                assert!(graph.graph.contains_node(i));
            } else {
                assert!(!graph.graph.contains_node(i));
            }
        }
    }
}
