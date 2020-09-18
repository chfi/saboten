use petgraph::prelude::*;

use handlegraph::{
    handle::{Direction, *},
    handlegraph::*,
    hashgraph::*,
};

use gfa::{gfa::GFA, parser::GFAParser};

use bstr::BString;

use std::{
    collections::{HashSet, VecDeque},
    fs::File,
    io::Write,
    ops::{Add, AddAssign, Sub, SubAssign},
    path::PathBuf,
};

/// To make a petgraph Graph(Map) into a multigraph, we track the
/// number of black and gray edges between two nodes by using this
/// struct as the edge weight type
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BiedgedWeight {
    pub black: usize,
    pub gray: usize,
}

impl BiedgedWeight {
    /// An empty weight has zero edges of either color.
    pub fn empty() -> Self {
        Default::default()
    }

    /// Construct a new edge weight with the provided edge counts.
    pub fn new(black: usize, gray: usize) -> Self {
        BiedgedWeight { black, gray }
    }

    /// Construct a new edge weight with the provided black count,
    /// with gray set to zero.
    pub fn black(black: usize) -> Self {
        BiedgedWeight { black, gray: 0 }
    }

    /// Construct a new edge weight with the provided gray count,
    /// with black set to zero.
    pub fn gray(gray: usize) -> Self {
        BiedgedWeight { black: 0, gray }
    }

    /// Extract the corresponding field based on the provided edge
    /// type variant.
    pub fn extract(&self, edge_type: BiedgedEdgeType) -> usize {
        use BiedgedEdgeType::*;
        match edge_type {
            Black => self.black,
            Gray => self.gray,
        }
    }
}

impl Add for BiedgedWeight {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            black: self.black + other.black,
            gray: self.gray + other.gray,
        }
    }
}

impl AddAssign for BiedgedWeight {
    fn add_assign(&mut self, other: Self) {
        self.black += other.black;
        self.gray += other.gray;
    }
}

impl Sub for BiedgedWeight {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            black: self.black - other.black,
            gray: self.gray - other.gray,
        }
    }
}

impl SubAssign for BiedgedWeight {
    fn sub_assign(&mut self, other: Self) {
        self.black -= other.black;
        self.gray -= other.gray;
    }
}

/// A biedged graph is a graph with two types of edges: black edges and gray edges, such that each vertex is
/// incident with at most one black edge. More information can be found in:
/// Superbubbles, Ultrabubbles, and Cacti by BENEDICT PATEN et al.
#[derive(Default)]
pub struct BiedgedGraph {
    pub(crate) graph: UnGraphMap<u64, BiedgedWeight>,
}

impl BiedgedGraph {
    /// Returns an iterator over the gray edges in the graph, where
    /// the first two elements in the tuple are the `from` and `to`
    /// nodes, and the third is the weight containing the number of
    /// gray and black edges between the two nodes.
    pub fn gray_edges(
        &self,
    ) -> impl Iterator<Item = (u64, u64, &BiedgedWeight)> {
        self.graph.all_edges().filter(|(_, _, w)| w.gray > 0)
    }

    /// Returns an iterator over the black edges in the graph, where
    /// the first two elements in the tuple are the `from` and `to`
    /// nodes, and the third is the weight containing the number of
    /// gray and black edges between the two nodes.
    pub fn black_edges(
        &self,
    ) -> impl Iterator<Item = (u64, u64, &BiedgedWeight)> {
        self.graph.all_edges().filter(|(_, _, w)| w.black > 0)
    }

    /// Produces the sum of the gray edges in the graph, counted using
    /// the edge weights.
    pub fn gray_edge_count(&self) -> usize {
        self.gray_edges().map(|(_, _, w)| w.gray).sum()
    }

    /// Produces the sum of the black edges in the graph, counted using
    /// the edge weights.
    pub fn black_edge_count(&self) -> usize {
        self.black_edges().map(|(_, _, w)| w.black).sum()
    }

    /// Produces the sum of all edges in the graph, counted using the
    /// edge weights. Note that black and gray edges are summed
    /// together.
    pub fn edges_count(&self) -> usize {
        self.graph
            .all_edges()
            .map(|(_, _, w)| w.black + w.gray)
            .sum()
    }
}

/// A node in the biedged graph
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiedgedNode {
    pub id: u64,
}
// NOTE: this is a struct and not a typedef because I plan on adding more fields to it

/// An enum that represent the two possible types of edges: Black and Gray
#[derive(Clone, Copy, PartialEq)]
pub enum BiedgedEdgeType {
    Black,
    Gray,
}
/// An edge in a biedged graph
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiedgedEdge {
    pub from: u64,
    pub to: u64,
}

/// Traits

// The NodeFunctions trait includes functions necessary to handle nodes, while maintaining
// coherence between self.graph and self.nodes
pub trait NodeFunctions {
    // Node functions
    fn add_node(&mut self, id: u64) -> u64;

    fn remove_node(&mut self, id: u64) -> bool;

    fn get_adjacent_nodes(&self, id: u64) -> Vec<u64>;

    fn get_adjacent_nodes_by_color(
        &self,
        id: u64,
        edge_type: BiedgedEdgeType,
    ) -> Vec<u64>;

    fn get_nodes(&self) -> Vec<BiedgedNode>;
}

// The EdgeFunctions trait includes functions necessary to handle edges, while maintaining
// coherence between self.graph and self.black_edges/self.gray_edges
pub trait EdgeFunctions {
    // Edge functions
    fn add_edge(
        &mut self,
        from: u64,
        to: u64,
        edge_type: BiedgedEdgeType,
    ) -> Option<BiedgedEdge>;

    fn remove_edge(&mut self, from: u64, to: u64) -> Option<BiedgedEdge>;

    fn remove_edges_incident_to_node(&mut self, id: u64) -> Vec<BiedgedEdge>;

    fn contract_edge(&mut self, from: u64, to: u64);
}

impl NodeFunctions for BiedgedGraph {
    /// Add the node with the given id to the graph
    fn add_node(&mut self, id: u64) -> u64 {
        self.graph.add_node(id)
    }

    /// Remove the node with the given id, and all its incident edges
    fn remove_node(&mut self, id: u64) -> bool {
        self.graph.remove_node(id)
    }

    /// Returns all the adjacents nodes to a node with a given id,
    /// that is all the nodes with an edge from/to the given id.
    fn get_adjacent_nodes(&self, id: u64) -> Vec<u64> {
        self.graph.neighbors(id).collect()
    }

    /// Returns the adjacents nodes to a node with a given id,
    /// connected with a specific edge_type (i.e. either
    /// BiedgedEdge::Black or BiedgedEdge::Gray)
    fn get_adjacent_nodes_by_color(
        &self,
        id: u64,
        edge_type: BiedgedEdgeType,
    ) -> Vec<u64> {
        let mut adj_nodes: Vec<u64> = Vec::new();
        if self.graph.contains_node(id) {
            self.graph
                .edges(id)
                .filter(|(_, _, w)| w.extract(edge_type) != 0)
                .for_each(|(from, to, _)| {
                    if from == id {
                        adj_nodes.push(to);
                    } else {
                        adj_nodes.push(from);
                    }
                });
        }
        adj_nodes
    }

    /// Return all the nodes in the graph
    fn get_nodes(&self) -> Vec<BiedgedNode> {
        self.graph.nodes().map(|id| BiedgedNode { id }).collect()
    }
}

impl EdgeFunctions for BiedgedGraph {
    // TODO: think which string to use
    /// Add an edge between the the two nodes specified by from and to. Returns None
    /// if at least one between from and to does not exist.
    fn add_edge(
        &mut self,
        from: u64,
        to: u64,
        edge_type: BiedgedEdgeType,
    ) -> Option<BiedgedEdge> {
        if self.graph.contains_node(from) && self.graph.contains_node(to) {
            let edge_to_add = BiedgedEdge { from, to };
            let mut new_weight = self
                .graph
                .edge_weight(from, to)
                .copied()
                .unwrap_or_default();
            match edge_type {
                BiedgedEdgeType::Black => {
                    new_weight += BiedgedWeight::black(1);
                }
                BiedgedEdgeType::Gray => {
                    new_weight += BiedgedWeight::gray(1);
                }
            }
            self.graph.add_edge(from, to, new_weight);
            Some(edge_to_add)
        } else {
            None
        }
    }

    /// Remove an edge between the the two nodes specified by from and to. Returns None
    /// if the edge between from and to does not exist.
    fn remove_edge(&mut self, from: u64, to: u64) -> Option<BiedgedEdge> {
        let edge_to_remove = BiedgedEdge { from, to };
        if self.graph.contains_edge(from, to) {
            self.graph.remove_edge(from, to);
            Some(edge_to_remove)
        } else {
            None
        }
    }

    /// Remove all the edges incident to the node with the given id, while leaving the node
    /// itself intact.
    fn remove_edges_incident_to_node(&mut self, id: u64) -> Vec<BiedgedEdge> {
        let edges: Vec<_> = self
            .graph
            .neighbors(id)
            .map(|n| BiedgedEdge { from: id, to: n })
            .collect();
        for edge in &edges {
            self.graph.remove_edge(edge.from, edge.to);
        }
        edges
    }

    /// Contract a given edge. For more information on edge contraction go to:
    /// https://en.wikipedia.org/wiki/Edge_contraction
    fn contract_edge(&mut self, from: u64, to: u64) {
        let mut adjacent_nodes_by_black_edge: Vec<u64> = Vec::new();
        let mut adjacent_nodes_by_gray_edge: Vec<u64> = Vec::new();

        let mut first_node_adjacent_nodes_black: Vec<u64> =
            self.get_adjacent_nodes_by_color(from, BiedgedEdgeType::Black);
        let mut first_node_adjacent_nodes_gray: Vec<u64> =
            self.get_adjacent_nodes_by_color(from, BiedgedEdgeType::Gray);

        let mut second_node_adjacent_nodes_black: Vec<u64> =
            self.get_adjacent_nodes_by_color(to, BiedgedEdgeType::Black);
        let mut second_node_adjacent_nodes_gray: Vec<u64> =
            self.get_adjacent_nodes_by_color(to, BiedgedEdgeType::Gray);

        adjacent_nodes_by_black_edge
            .append(&mut first_node_adjacent_nodes_black);
        adjacent_nodes_by_black_edge
            .append(&mut second_node_adjacent_nodes_black);

        adjacent_nodes_by_gray_edge.append(&mut first_node_adjacent_nodes_gray);
        adjacent_nodes_by_gray_edge
            .append(&mut second_node_adjacent_nodes_gray);

        // All adjacent edges will also be removed
        self.remove_node(from);
        self.remove_node(to);

        let added_node = self.add_node(from);

        for adj_node in adjacent_nodes_by_black_edge {
            self.add_edge(added_node, adj_node, BiedgedEdgeType::Black);
        }

        for adj_node in adjacent_nodes_by_gray_edge {
            if adj_node != from && adj_node != to {
                self.add_edge(added_node, adj_node, BiedgedEdgeType::Gray);
            }
        }
    }
}

impl BiedgedGraph {
    /// Create a new "empty" biedged graph
    pub fn new() -> BiedgedGraph {
        Default::default()
    }

    /// Create a biedged graph from a Handlegraph
    pub fn from_handlegraph<T: HandleGraph>(graph: &T) -> BiedgedGraph {
        let mut biedged: UnGraphMap<u64, BiedgedWeight> = UnGraphMap::new();

        // Create queue
        // NOTE: this is a Queue based implementation, this was done
        // in order not to get a stack overflow
        let mut q: VecDeque<NodeId> = VecDeque::new();

        // Start from the node with the lowest id
        // will probably always be 1, but this is safer
        let node_id = graph.min_node_id();

        // Store which nodes have already been visited
        let mut visited_nodes: HashSet<NodeId> = HashSet::new();

        // Insert first value
        q.push_back(node_id);

        // Store black and grey edges
        let mut black_edges: Vec<BiedgedEdge> = Vec::new();
        let mut gray_edges: Vec<BiedgedEdge> = Vec::new();
        // Store nodes
        let mut nodes: Vec<BiedgedNode> = Vec::new();

        while let Some(curr_node) = q.pop_front() {
            if visited_nodes.contains(&curr_node) {
                continue;
            }

            let current_handle = Handle::pack(curr_node, false);
            let left_id: u64 = current_handle.as_integer();
            let right_id: u64 = current_handle.flip().as_integer();

            // For each node in the Handlegraph, there will be two nodes in the biedged graph
            // each representing one of the two sides
            let node_1 = biedged.add_node(left_id);
            let node_2 = biedged.add_node(right_id);

            // The two nodes are connected
            // let id_edge = format!("B: {}", current_handle.unpack_number());
            biedged.add_edge(node_1, node_2, BiedgedWeight::empty());

            // Add nodes to vec
            nodes.push(BiedgedNode { id: left_id });
            nodes.push(BiedgedNode { id: right_id });

            // Add edge to black edges
            black_edges.push(BiedgedEdge {
                from: node_1,
                to: node_2,
            });

            // Look for neighbors in the Handlegraph, add edges in the biedged graph
            for neighbor in
                graph.handle_edges_iter(current_handle, Direction::Right)
            {
                // Add first node for neighbor
                let neighbor_node_biedged =
                    biedged.add_node(neighbor.as_integer());

                // Add edge from neighbor to
                // let id_edge = format!("G: {}->{}", curr_node, neighbor.id());
                biedged.add_edge(
                    node_2,
                    neighbor_node_biedged,
                    BiedgedWeight::empty(),
                );

                // Add edge to gray edges
                gray_edges.push(BiedgedEdge {
                    from: node_2,
                    to: neighbor_node_biedged,
                });

                // Add to queue
                q.push_back(neighbor.id());
            }

            visited_nodes.insert(curr_node);
        }

        BiedgedGraph { graph: biedged }
    }

    /// Convert a GFA to a biedged graph if file exists
    /// otherwise return None
    pub fn from_gfa_file(path: &PathBuf) -> Option<BiedgedGraph> {
        let parser = GFAParser::new();
        let gfa: GFA<BString, ()> = parser.parse_file(path).ok()?;
        let graph = HashGraph::from_gfa(&gfa);
        Some(BiedgedGraph::from_handlegraph(&graph))
    }

    /// Print the biedged graph to a .dot file. This file can then be used by
    /// various tools (i.e. Graphviz) to produce a graphical representation of the graph
    /// (i.e. dot -Tpng graph.dot -o graph.png)
    pub fn output_dot<T: Write>(&self, mut t: T) -> std::io::Result<()> {
        use petgraph::dot::{Config, Dot};

        // let mut f = File::create(path).unwrap();
        // let output = format!("{}", Dot::with_config(&graph.graph, &[Config::EdgeNoLabel]));
        let output = format!(
            "{:?}",
            Dot::with_config(&self.graph, &[Config::NodeNoLabel])
        );
        t.write_all(&output.as_bytes())?;
        Ok(())
    }
}

// ----------------------------------- TESTS -------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        assert!(graph.graph.contains_node(10));
        assert!(graph.get_nodes().len() == 1);
        assert!(*graph.get_nodes().get(0).unwrap() == BiedgedNode { id: 10 });
    }

    #[test]
    fn test_get_nodes() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.add_node(20);
        graph.add_node(30);

        assert!(graph.get_nodes().len() == 3);
        assert!(*graph.get_nodes().get(0).unwrap() == BiedgedNode { id: 10 });
        assert!(*graph.get_nodes().get(1).unwrap() == BiedgedNode { id: 20 });
        assert!(*graph.get_nodes().get(2).unwrap() == BiedgedNode { id: 30 });
    }

    #[test]
    fn test_get_adjacent_nodes() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.add_node(20);
        graph.add_node(30);

        graph.add_edge(10, 20, BiedgedEdgeType::Black);
        graph.add_edge(10, 30, BiedgedEdgeType::Gray);

        let adjacent_nodes = graph.get_adjacent_nodes(10);
        assert!(adjacent_nodes.len() == 2);
        assert!(adjacent_nodes.contains(&20));
        assert!(adjacent_nodes.contains(&30));

        // Check if node can be either starting or ending
        graph.add_node(0);
        graph.add_edge(0, 10, BiedgedEdgeType::Black);
        let adjacent_nodes = graph.get_adjacent_nodes(10);
        assert!(adjacent_nodes.len() == 3);
        assert!(adjacent_nodes.contains(&0));
    }

    #[test]
    fn test_add_edge() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.add_node(20);
        graph.add_node(30);

        graph.add_edge(10, 20, BiedgedEdgeType::Black);
        assert!(graph.graph.contains_edge(10, 20));

        assert_eq!(graph.black_edge_count(), 1);
        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 0 }),
            graph.graph.edge_weight(10, 20)
        );

        graph.add_edge(20, 30, BiedgedEdgeType::Gray);
        assert!(graph.graph.contains_edge(20, 30));
        assert_eq!(graph.gray_edge_count(), 1);

        assert_eq!(
            Some(&BiedgedWeight { black: 0, gray: 1 }),
            graph.graph.edge_weight(20, 30)
        );

        graph.add_edge(20, 30, BiedgedEdgeType::Black);

        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 1 }),
            graph.graph.edge_weight(20, 30)
        );
    }

    #[test]
    fn test_remove_edge() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.add_node(20);
        graph.add_edge(10, 20, BiedgedEdgeType::Black);

        graph.remove_edge(10, 20);
        assert!(!graph.graph.contains_edge(10, 20));
        assert_eq!(graph.black_edge_count(), 0);
    }

    #[test]
    fn test_remove_node_and_adjacent_edges() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.add_node(20);
        graph.add_node(30);
        graph.add_edge(10, 20, BiedgedEdgeType::Black);
        graph.add_edge(10, 30, BiedgedEdgeType::Gray);

        graph.remove_node(10);

        assert!(!graph.graph.contains_edge(10, 20));
        assert!(!graph.graph.contains_edge(10, 30));

        assert_eq!(graph.black_edge_count(), 0);
        assert_eq!(graph.gray_edge_count(), 0);

        assert_eq!(None, graph.graph.edge_weight(10, 20));
        assert_eq!(None, graph.graph.edge_weight(10, 30));
    }

    #[test]
    fn test_remove_edges_adjacent_to_node() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.add_node(20);
        graph.add_node(30);
        graph.add_edge(10, 20, BiedgedEdgeType::Black);
        graph.add_edge(10, 30, BiedgedEdgeType::Gray);

        graph.remove_edges_incident_to_node(10);

        assert!(graph.graph.contains_node(10));
        assert!(graph.get_nodes().contains(&BiedgedNode { id: 10 }));

        assert!(!graph.graph.contains_edge(10, 20));
        assert!(!graph.graph.contains_edge(10, 30));
        assert_eq!(graph.black_edge_count(), 0);
        assert_eq!(graph.gray_edge_count(), 0);
        assert!(!graph
            .black_edges()
            .find(|(from, to, _)| from == &10 && to == &20)
            .is_some());
        assert!(!graph
            .gray_edges()
            .find(|(from, to, _)| from == &10 && to == &30)
            .is_some());
    }

    #[test]
    fn test_contract_edge() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.add_node(20);
        graph.add_node(30);
        graph.add_edge(10, 20, BiedgedEdgeType::Black);
        graph.add_edge(10, 30, BiedgedEdgeType::Gray);
        graph.add_edge(20, 30, BiedgedEdgeType::Black);

        assert_eq!(None, graph.graph.edge_weight(10, 10));
        assert_eq!(
            Some(&BiedgedWeight { black: 0, gray: 1 }),
            graph.graph.edge_weight(10, 30)
        );
        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 0 }),
            graph.graph.edge_weight(10, 20)
        );
        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 0 }),
            graph.graph.edge_weight(20, 30)
        );

        graph.contract_edge(10, 20);

        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 0 }),
            graph.graph.edge_weight(10, 10)
        );
        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 1 }),
            graph.graph.edge_weight(10, 30)
        );
        assert_eq!(None, graph.graph.edge_weight(10, 20));
        assert_eq!(None, graph.graph.edge_weight(20, 30));

        assert!(graph.graph.contains_node(10));
        assert!(graph.graph.contains_node(30));
        assert!(!graph.graph.contains_node(20));

        assert!(graph.graph.edge_count() == 2);

        assert_eq!(graph.black_edge_count(), 2);
        assert_eq!(graph.gray_edge_count(), 1);
    }
}
