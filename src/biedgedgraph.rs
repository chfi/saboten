use petgraph::{
    dot::{Config, Dot},
    prelude::*,
};

use handlegraph::{
    handle::{Direction, *},
    handlegraph::*,
    hashgraph::*,
};

use gfa::parser::parse_gfa;

use std::{
    collections::{HashSet, VecDeque},
    fs::File,
    io::Write,
    path::PathBuf,
};

/// A biedged graph is a graph with two types of edges: black edges and gray edges, such that each vertex is
/// incident with at most one black edge. More information can be found in:
/// Superbubbles, Ultrabubbles, and Cacti by BENEDICT PATEN et al.

/// This implementation is basically a wrapper over Petgraph. This was necessary since Petgraph does not provide
/// ways to handle edge coloring, and also some other functions necessary to obtain a Cactus Graph.
#[derive(Default)]
pub struct BiedgedGraph {
    // The actual graph implementation, backed by Petgraph. The nodes have an id of type u64,
    // the edges can include non-empty Strings
    graph: UnGraphMap<u64, String>,

    // A Vec containing all the gray edges
    black_edges: Vec<BiedgedEdge>,

    // A Vec containing all the black edges
    gray_edges: Vec<BiedgedEdge>,

    // A Vec containing all nodes in the graph
    nodes: Vec<BiedgedNode>,
}
// NOTE: nodes from the nodes Vec are in a 1:1 relationship with nodes in the graph (i.e. if a node
// can be found by self.nodes.contains(id), it will be present exactly once in self.graph).
// However, this is not true with respect to the edges (i.e. self.black_edges could contain the same
// edge more than once, however in self.graph it will be present only once.) This is due to Petgraph
// not supporting multiple edges between the same nodes.

/// A node in the biedged graph
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiedgedNode {
    pub id: u64,
}
// NOTE: this is a struct and not a typedef because I plan on adding more fields to it

/// An enum that represent the two possible types of edges: Black and Gray
#[derive(PartialEq)]
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
    fn add_node(&mut self, id: u64) -> Option<u64>;
    fn remove_node(&mut self, id: u64) -> Option<u64>;
    fn remove_nodes_incident_with_edge(&mut self, from: u64, to: u64) -> Option<Vec<BiedgedEdge>>;

    fn get_adjacent_nodes(&self, id: u64) -> Option<Vec<u64>>;
    fn get_adjacent_nodes_by_edge_type(
        &self,
        id: u64,
        edge_type: BiedgedEdgeType,
    ) -> Option<Vec<u64>>;

    fn get_nodes(&self) -> &Vec<BiedgedNode>;
    //fn get_nodes_mut(&mut self) -> &mut Vec<BiedgedNode>;
}

// The EdgeFunctions trait includes functions necessary to handle edges, while maintaining
// coherence between self.graph and self.black_edges/self.gray_edges
pub trait EdgeFunctions {
    // Edge functions
    fn add_edge(&mut self, from: u64, to: u64, edge_type: BiedgedEdgeType) -> Option<BiedgedEdge>;
    fn remove_edge(&mut self, from: u64, to: u64) -> Option<BiedgedEdge>;
    fn remove_edges_incident_to_node(&mut self, id: u64) -> Option<Vec<BiedgedEdge>>;
    fn contract_edge(&mut self, from: u64, to: u64);

    fn edges_count(&self) -> usize;
    //fn get_edges(&self) -> &Vec<BiedgedEdge>;

    // Immutable getter/setter
    fn get_gray_edges(&self) -> &Vec<BiedgedEdge>;
    fn get_black_edges(&self) -> &Vec<BiedgedEdge>;

    // Mutable getter/setters
    fn get_gray_edges_mut(&mut self) -> &mut Vec<BiedgedEdge>;
    //fn get_black_edges_mut(&mut self) -> &mut Vec<BiedgedEdge>;
}

impl NodeFunctions for BiedgedGraph {
    /// Add the node with the given id to the graph
    fn add_node(&mut self, id: u64) -> Option<u64> {
        self.nodes.push(BiedgedNode { id: id });
        Some(self.graph.add_node(id))
    }

    /// Remove the node with the given id, and all its incident edges
    fn remove_node(&mut self, id: u64) -> Option<u64> {
        if self.graph.contains_node(id) {
            let _ = self.graph.remove_node(id);
            self.nodes.retain(|x| x.id != id);

            // Remove all incident edges from Vecs
            self.black_edges.retain(|x| !(x.from == id || x.to == id));
            self.gray_edges.retain(|x| !(x.from == id || x.to == id));

            Some(id)
        } else {
            None
        }
    }

    /// Remove the two nodes at the ends of the given edge
    fn remove_nodes_incident_with_edge(&mut self, from: u64, to: u64) -> Option<Vec<BiedgedEdge>> {
        let edge: &BiedgedEdge = &BiedgedEdge { from: from, to: to };
        let mut removed_edges: Vec<BiedgedEdge> = Vec::new();
        if self.black_edges.contains(edge) {
            self.remove_node(from);
            self.remove_node(to);
            removed_edges = self
                .black_edges
                .iter()
                .filter(|x| *x == edge)
                .map(|x| *x)
                .collect();
            self.black_edges.retain(|x| !(x == edge));
            Some(removed_edges)
        } else if self.gray_edges.contains(edge) {
            self.remove_node(from);
            self.remove_node(to);
            removed_edges = self
                .gray_edges
                .iter()
                .filter(|x| *x == edge)
                .map(|x| *x)
                .collect();
            self.gray_edges.retain(|x| !(x == edge));
            Some(removed_edges)
        } else {
            None
        }
    }

    /// Returns all the adjacents nodes to a node with a given id, that is all the nodes
    /// with an edge from/to the given id.
    fn get_adjacent_nodes(&self, id: u64) -> Option<Vec<u64>> {
        if self.graph.contains_node(id) {
            let adjacent_nodes: Vec<u64> = self.graph.edges(id).map(|x| x.1).collect();
            Some(adjacent_nodes)
        } else {
            None
        }
    }

    /// Returns the adjacents nodes to a node with a given id, connected with a specific
    /// edge_type (i.e. either BiedgedEdge::Black or BiedgedEdge::Gray)
    fn get_adjacent_nodes_by_edge_type(
        &self,
        id: u64,
        edge_type: BiedgedEdgeType,
    ) -> Option<Vec<u64>> {
        if self.graph.contains_node(id) {
            let mut adj_nodes: Vec<u64> = Vec::new();

            if edge_type == BiedgedEdgeType::Black {
                let adj_black_edges: Vec<&BiedgedEdge> = self
                    .black_edges
                    .iter()
                    .filter(|x| x.from == id || x.to == id)
                    .collect();

                for edge in adj_black_edges {
                    if edge.from == id {
                        adj_nodes.push(edge.to);
                    } else {
                        adj_nodes.push(edge.from);
                    }
                }
            } else if edge_type == BiedgedEdgeType::Gray {
                let adj_gray_edges: Vec<&BiedgedEdge> = self
                    .gray_edges
                    .iter()
                    .filter(|x| x.from == id || x.to == id)
                    .collect();

                for edge in adj_gray_edges {
                    if edge.from == id {
                        adj_nodes.push(edge.to);
                    } else {
                        adj_nodes.push(edge.from);
                    }
                }
            }

            Some(adj_nodes)
        } else {
            None
        }
    }

    /// Return all the nodes in the graph
    fn get_nodes(&self) -> &Vec<BiedgedNode> {
        self.nodes.as_ref()
    }

    // fn get_nodes_mut(&mut self) -> &mut Vec<BiedgedNode> {
    //     self.nodes.as_mut()
    // }
}

impl EdgeFunctions for BiedgedGraph {
    // TODO: think which string to use
    /// Add an edge between the the two nodes specified by from and to. Returns None
    /// if at least one between from and to does not exist.
    fn add_edge(&mut self, from: u64, to: u64, edge_type: BiedgedEdgeType) -> Option<BiedgedEdge> {
        if self.graph.contains_node(from) && self.graph.contains_node(to) {
            let edge_to_add = BiedgedEdge { from, to };
            self.graph.add_edge(from, to, String::from(""));
            match edge_type {
                BiedgedEdgeType::Black => self.black_edges.push(edge_to_add),
                BiedgedEdgeType::Gray => self.gray_edges.push(edge_to_add),
            }
            Some(edge_to_add)
        } else {
            None
        }
    }

    /// Remove an edge between the the two nodes specified by from and to. Returns None
    /// if the edge between from and to does not exist.
    fn remove_edge(&mut self, from: u64, to: u64) -> Option<BiedgedEdge> {
        let edge_to_remove = BiedgedEdge { from: from, to: to };

        if self.graph.contains_edge(from, to) {
            self.graph.remove_edge(from, to);
            if self.black_edges.contains(&edge_to_remove) {
                self.black_edges.retain(|edge| edge != &edge_to_remove);
                Some(edge_to_remove)
            } else if self.gray_edges.contains(&edge_to_remove) {
                self.gray_edges.retain(|edge| edge != &edge_to_remove);
                Some(edge_to_remove)
            } else {
                panic!("this should be impossible!");
            }
        } else {
            None
        }
    }

    /// Remove all the edges incident to the node with the given id, while leaving the node
    /// itself intact.
    fn remove_edges_incident_to_node(&mut self, id: u64) -> Option<Vec<BiedgedEdge>> {
        if self.nodes.contains(&BiedgedNode { id: id }) && self.graph.contains_node(id) {
            let mut incident_edges: Vec<BiedgedEdge> = Vec::new();
            let mut black_edges: Vec<BiedgedEdge> = self
                .black_edges
                .iter()
                .filter(|x| x.from == id || x.to == id)
                .map(|x| *x)
                .collect();
            let mut gray_edges: Vec<BiedgedEdge> = self
                .gray_edges
                .iter()
                .filter(|x| x.from == id || x.to == id)
                .map(|x| *x)
                .collect();

            self.black_edges.retain(|x| !(x.from == id || x.to == id));
            self.gray_edges.retain(|x| !(x.from == id || x.to == id));

            incident_edges.append(&mut black_edges);
            incident_edges.append(&mut gray_edges);

            for edge in &incident_edges {
                self.graph.remove_edge(edge.from, edge.to);
            }

            Some(incident_edges)
        } else {
            None
        }
    }

    /// Contract a given edge. For more information on edge contraction go to:
    /// https://en.wikipedia.org/wiki/Edge_contraction
    fn contract_edge(&mut self, from: u64, to: u64) {
        let mut adjacent_nodes_by_black_edge: Vec<u64> = Vec::new();
        let mut adjacent_nodes_by_gray_edge: Vec<u64> = Vec::new();

        let mut first_node_adjacent_nodes_black: Vec<u64> = self
            .get_adjacent_nodes_by_edge_type(from, BiedgedEdgeType::Black)
            .unwrap();
        let mut first_node_adjacent_nodes_gray: Vec<u64> = self
            .get_adjacent_nodes_by_edge_type(from, BiedgedEdgeType::Gray)
            .unwrap();

        let mut second_node_adjacent_nodes_black: Vec<u64> = self
            .get_adjacent_nodes_by_edge_type(to, BiedgedEdgeType::Black)
            .unwrap();
        let mut second_node_adjacent_nodes_gray: Vec<u64> = self
            .get_adjacent_nodes_by_edge_type(to, BiedgedEdgeType::Gray)
            .unwrap();

        adjacent_nodes_by_black_edge.append(&mut first_node_adjacent_nodes_black);
        adjacent_nodes_by_black_edge.append(&mut second_node_adjacent_nodes_black);

        adjacent_nodes_by_gray_edge.append(&mut first_node_adjacent_nodes_gray);
        adjacent_nodes_by_gray_edge.append(&mut second_node_adjacent_nodes_gray);

        self.remove_node(from).unwrap();
        self.remove_node(to).unwrap();
        // All adjacent edges will also be removed

        //TODO: decide which id to use
        let added_node = self.add_node(from).unwrap();

        for adj_node in adjacent_nodes_by_black_edge {
            if adj_node != from && adj_node != to {
                self.add_edge(added_node, adj_node, BiedgedEdgeType::Black);
            }
        }

        for adj_node in adjacent_nodes_by_gray_edge {
            if adj_node != from && adj_node != to {
                self.add_edge(added_node, adj_node, BiedgedEdgeType::Gray);
            }
        }
    }

    /// Returns the number of edges in the graph
    fn edges_count(&self) -> usize {
        self.get_black_edges().len() + self.get_gray_edges().len()
    }
    // fn get_edges(&self) -> &Vec<BiedgedEdge> {
    //     &Vec::from_iter(self.black_edges.into_iter().chain(self.gray_edges.into_iter()))
    // }

    /// Return all the gray edges in the graph
    fn get_gray_edges(&self) -> &Vec<BiedgedEdge> {
        self.gray_edges.as_ref()
    }
    /// Return all the black edges in the graph
    fn get_black_edges(&self) -> &Vec<BiedgedEdge> {
        self.black_edges.as_ref()
    }

    fn get_gray_edges_mut(&mut self) -> &mut Vec<BiedgedEdge> {
        self.gray_edges.as_mut()
    }
    // fn get_black_edges_mut(&mut self) -> &mut Vec<BiedgedEdge> {
    //     self.black_edges.as_mut()
    // }
}

impl BiedgedGraph {
    /// Create a new "empty" biedged graph
    pub fn new() -> BiedgedGraph {
        Default::default()
    }

    /// Create a biedged graph from a Handlegraph
    pub fn handlegraph_to_biedged_graph(graph: &HashGraph) -> BiedgedGraph {
        let mut biedged: UnGraphMap<u64, String> = UnGraphMap::new();

        // Create queue
        // NOTE: this is a Queue based implementation, this was done
        // in order not to get a stack overflow
        let mut q: VecDeque<NodeId> = VecDeque::new();

        // Start from the node with the lowest id
        // will probably always be 1, but this is safer
        let node_id = graph.min_id;

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
            let id_edge = format!("B: {}", current_handle.unpack_number());
            biedged.add_edge(node_1, node_2, id_edge);

            // Add nodes to vec
            nodes.push(BiedgedNode { id: left_id });
            nodes.push(BiedgedNode { id: right_id });

            // Add edge to black edges
            black_edges.push(BiedgedEdge {
                from: node_1,
                to: node_2,
            });

            // Look for neighbors in the Handlegraph, add edges in the biedged graph
            for neighbor in handle_edges_iter(graph, current_handle, Direction::Right) {
                // Add first node for neighbor
                let neighbor_node_biedged = biedged.add_node(neighbor.as_integer());

                // Add edge from neighbor to
                let id_edge = format!("G: {}->{}", curr_node, neighbor.id());
                biedged.add_edge(node_2, neighbor_node_biedged, id_edge);

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

        // Create Biedged graph
        let biedged_graph = BiedgedGraph {
            graph: biedged,
            black_edges: black_edges,
            gray_edges: gray_edges,
            nodes: nodes,
        };

        biedged_graph
    }

    /// Convert a GFA to a biedged graph if file exists
    /// otherwise return None
    pub fn gfa_to_biedged_graph(path: &PathBuf) -> Option<BiedgedGraph> {
        if let Some(gfa) = parse_gfa(path) {
            let graph = HashGraph::from_gfa(&gfa);
            Some(BiedgedGraph::handlegraph_to_biedged_graph(&graph))
        } else {
            None
        }
    }

    /// Print the biedged graph to a .dot file. This file can then be used by
    /// various tools (i.e. Graphviz) to produce a graphical representation of the graph
    /// (i.e. dot -Tpng graph.dot -o graph.png)
    pub fn biedged_to_dot(&self, path: &PathBuf) -> std::io::Result<()> {
        let mut f = File::create(path).unwrap();
        //let output = format!("{}", Dot::with_config(&graph.graph, &[Config::EdgeNoLabel]));
        let output = format!("{}", Dot::with_config(&self.graph, &[Config::NodeNoLabel]));
        f.write_all(&output.as_bytes())?;
        Ok(())
    }
}

// ----------------------------------- TESTS -------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::biedged_to_cactus::contract_all_gray_edges;
    //use handlegraph::mutablehandlegraph::MutableHandleGraph;

    #[test]
    fn test_add_node() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        assert!(graph.graph.contains_node(10));
        assert!(graph.get_nodes().len() == 1);
        assert!(*graph.get_nodes().get(0).unwrap() == BiedgedNode { id: 10 });
    }

    #[test]
    fn test_remove_node() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.remove_node(10);
        assert!(!graph.graph.contains_node(10));
        assert!(graph.get_nodes().len() == 0);
        assert!(graph.get_nodes().get(0) == None);
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

        let adjacent_nodes = graph.get_adjacent_nodes(10).unwrap();
        assert!(adjacent_nodes.len() == 2);
        assert!(adjacent_nodes.contains(&20));
        assert!(adjacent_nodes.contains(&30));

        // Check if node can be either starting or ending
        graph.add_node(0);
        graph.add_edge(0, 10, BiedgedEdgeType::Black);
        let adjacent_nodes = graph.get_adjacent_nodes(10).unwrap();
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
        assert!(graph.get_black_edges().len() == 1);
        assert!(graph
            .get_black_edges()
            .contains(&BiedgedEdge { from: 10, to: 20 }));

        graph.add_edge(20, 30, BiedgedEdgeType::Gray);
        assert!(graph.graph.contains_edge(20, 30));
        assert!(graph.get_gray_edges().len() == 1);
        assert!(graph
            .get_gray_edges()
            .contains(&BiedgedEdge { from: 20, to: 30 }));
    }

    #[test]
    fn test_remove_edge() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.add_node(20);
        graph.add_edge(10, 20, BiedgedEdgeType::Black);

        graph.remove_edge(10, 20);
        assert!(!graph.graph.contains_edge(10, 20));
        assert!(graph.get_black_edges().len() == 0);
        assert!(!graph
            .get_black_edges()
            .contains(&BiedgedEdge { from: 10, to: 20 }));
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
        assert!(graph.get_black_edges().len() == 0);
        assert!(graph.get_gray_edges().len() == 0);
        assert!(!graph
            .get_black_edges()
            .contains(&BiedgedEdge { from: 10, to: 20 }));
        assert!(!graph
            .get_gray_edges()
            .contains(&BiedgedEdge { from: 10, to: 30 }));
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

        assert! {graph.graph.contains_node(10)};
        assert!(graph.get_nodes().contains(&BiedgedNode { id: 10 }));

        assert!(!graph.graph.contains_edge(10, 20));
        assert!(!graph.graph.contains_edge(10, 30));
        assert!(graph.get_black_edges().len() == 0);
        assert!(graph.get_gray_edges().len() == 0);
        assert!(!graph
            .get_black_edges()
            .contains(&BiedgedEdge { from: 10, to: 20 }));
        assert!(!graph
            .get_gray_edges()
            .contains(&BiedgedEdge { from: 10, to: 30 }));
    }

    #[test]
    fn test_remove_nodes_incident_with_edge() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.add_node(20);
        graph.add_node(30);
        graph.add_edge(10, 20, BiedgedEdgeType::Black);
        graph.add_edge(10, 30, BiedgedEdgeType::Gray);

        graph.remove_nodes_incident_with_edge(10, 20);

        assert! {!graph.graph.contains_node(10)};
        assert! {!graph.graph.contains_node(20)};
        assert!(!graph.get_nodes().contains(&BiedgedNode { id: 10 }));
        assert!(!graph.get_nodes().contains(&BiedgedNode { id: 20 }));
        assert!(!graph.graph.contains_edge(10, 20));
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

        graph.contract_edge(10, 20);

        assert!(graph.graph.contains_node(10));
        assert!(!graph.get_nodes().contains(&BiedgedNode { id: 20 }));
        assert!(!graph.graph.contains_node(20));
        assert!(graph.graph.edge_count() == 1);
    }

    #[test]
    fn test_contract_all_gray_edges() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();

        //First Handlegraph node
        graph.add_node(10);
        graph.add_node(11);
        graph.add_edge(10, 11, BiedgedEdgeType::Black);

        //Second Handlegraph node
        graph.add_node(20);
        graph.add_node(21);
        graph.add_edge(20, 21, BiedgedEdgeType::Black);

        //Third Handlegraph node
        graph.add_node(30);
        graph.add_node(31);
        graph.add_edge(30, 31, BiedgedEdgeType::Black);

        //Forth Handlegraph node
        graph.add_node(40);
        graph.add_node(41);
        graph.add_edge(40, 41, BiedgedEdgeType::Black);

        //Add Handlegraph edges
        graph.add_edge(11, 20, BiedgedEdgeType::Gray);
        graph.add_edge(11, 30, BiedgedEdgeType::Gray);
        graph.add_edge(21, 40, BiedgedEdgeType::Gray);
        graph.add_edge(31, 40, BiedgedEdgeType::Gray);

        contract_all_gray_edges(&mut graph);

        println!(
            "{:#?}",
            Dot::with_config(&graph.graph, &[Config::NodeNoLabel])
        );
        println!("Nodes: {:#?}", graph.get_nodes());
        println!("Gray_edges {:#?}", graph.get_gray_edges());
        println!("Black_edges {:#?}", graph.get_black_edges());

        assert!(graph.get_nodes().len() == 4);
        assert!(graph.get_black_edges().len() == 4);

        // NOTE: petgraph does not actually support multiple edges between two given nodes
        // however, they are allowed in Biedged Graphs. For this reason it is better to use
        // the count_edges function provided by the EdgeFunctions trait.
        assert!(graph.graph.edge_count() == 3);
    }
}
