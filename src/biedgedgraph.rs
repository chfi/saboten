use petgraph::prelude::*;

use handlegraph::{
    handle::{Direction, *},
    handlegraph::*,
    hashgraph::*,
};

use gfa::{
    gfa::{Header, Link, Orientation, Segment, GFA},
    parser::GFAParser,
};

use bstr::BString;

use std::{
    collections::{HashMap, HashSet, VecDeque},
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
    pub graph: UnGraphMap<u64, BiedgedWeight>,
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
    pub fn edge_color_count(&self) -> usize {
        self.graph
            .all_edges()
            .map(|(_, _, w)| w.black + w.gray)
            .sum()
    }

    pub fn contract_edge(&mut self, left: u64, right: u64) -> Option<()> {
        // We'll always remove the node with a higher ID, for consistency
        let from = left.min(right);
        let to = left.max(right);

        let weight = self.graph.edge_weight(from, to).copied()?;

        // Retrieve the edges of the node we're removing
        let to_edges: Vec<(u64, u64, BiedgedWeight)> = self
            .graph
            .edges(to)
            .filter(|(_, node, _)| node != &from)
            .map(|(a, b, w)| (a, b, *w))
            .collect();

        self.graph.remove_node(to);

        // add the edges that were removed with the deleted node
        for (_, other, w) in to_edges {
            if let Some(old_weight) = self.graph.edge_weight_mut(from, other) {
                *old_weight += w;
            } else {
                self.graph.add_edge(from, other, w);
            }
        }

        if weight.black > 0 {
            let new_weight = BiedgedWeight::black(weight.black);
            if let Some(self_weight) = self.graph.edge_weight_mut(from, from) {
                *self_weight += new_weight;
            } else {
                self.graph.add_edge(from, from, new_weight);
            }
        }

        if weight.gray > 1 {
            let new_weight = BiedgedWeight::gray(weight.gray - 1);
            if let Some(self_weight) = self.graph.edge_weight_mut(from, from) {
                *self_weight += new_weight;
            } else {
                self.graph.add_edge(from, from, new_weight);
            }
        }

        Some(())
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
}

impl BiedgedGraph {
    /// Create a new "empty" biedged graph
    pub fn new() -> BiedgedGraph {
        Default::default()
    }

    pub fn sorted_edges(&self) -> Vec<(u64, u64, usize, usize)> {
        let mut black_edges = self
            .black_edges()
            .map(|x| (x.0, x.1, x.2.black, x.2.gray))
            .collect::<Vec<_>>();
        let mut gray_edges = self
            .gray_edges()
            .map(|x| (x.0, x.1, x.2.black, x.2.gray))
            .collect::<Vec<_>>();

        black_edges.sort();
        gray_edges.sort();
        black_edges.extend(gray_edges);
        black_edges
    }

    pub fn example() -> BiedgedGraph {
        let mut be_graph: UnGraphMap<u64, BiedgedWeight> = UnGraphMap::new();

        // let black_edges =
        //     vec![(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)];

        let black_edges =
            vec![(0, 10), (1, 11), (2, 12), (3, 13), (4, 14), (5, 15)];

        let gray_edges = vec![
            (10, 1),
            (10, 2),
            (11, 3),
            (12, 3),
            (13, 4),
            (13, 5),
            (13, 0),
        ];

        for (a, b) in black_edges {
            be_graph.add_edge(a, b, BiedgedWeight::black(1));
        }

        for (a, b) in gray_edges {
            be_graph.add_edge(a, b, BiedgedWeight::gray(1));
        }

        BiedgedGraph { graph: be_graph }
    }

    /// Create a biedged graph from a Handlegraph
    /// apparently kinda broken!
    pub fn from_handlegraph<T: HandleGraph>(graph: &T) -> Self {
        let mut be_graph: UnGraphMap<u64, BiedgedWeight> = UnGraphMap::new();

        // Add the nodes
        for handle in graph.handles_iter() {
            let left_id = handle.unpack_number();
            let right_id = std::u64::MAX - left_id;
            be_graph.add_node(left_id);
            be_graph.add_node(right_id);
            be_graph.add_edge(left_id, right_id, BiedgedWeight::black(1));
        }

        // Add the edges
        for Edge(from, to) in graph.edges_iter() {
            let from_id = std::u64::MAX - from.unpack_number();
            let to_id = to.unpack_number();
            be_graph.add_edge(from_id, to_id, BiedgedWeight::gray(1));
        }

        BiedgedGraph { graph: be_graph }
    }

    pub fn from_gfa(gfa: &GFA<usize, ()>) -> BiedgedGraph {
        let mut be_graph: UnGraphMap<u64, BiedgedWeight> = UnGraphMap::new();

        for segment in gfa.segments.iter() {
            let name = segment.name as u64;
            let left_id = name;
            let right_id = std::u64::MAX - left_id;
            be_graph.add_node(left_id);
            be_graph.add_node(right_id);
            be_graph.add_edge(left_id, right_id, BiedgedWeight::black(1));
        }

        for link in gfa.links.iter() {
            let from_id = std::u64::MAX - link.from_segment as u64;
            let to_id = link.to_segment as u64;
            be_graph.add_edge(from_id, to_id, BiedgedWeight::gray(1));
        }

        BiedgedGraph { graph: be_graph }
    }

    /// Convert a GFA to a biedged graph if file exists
    /// otherwise return None
    pub fn from_gfa_file(path: &PathBuf) -> Option<BiedgedGraph> {
        let parser = GFAParser::new();
        let gfa: GFA<usize, ()> = parser.parse_file(path).ok()?;
        Some(BiedgedGraph::from_gfa(&gfa))
    }

    pub fn to_gfa_usize(&self) -> GFA<usize, ()> {
        let mut segments = Vec::new();
        let mut links = Vec::new();

        for id in self.graph.nodes() {
            let name = id as usize;
            segments.push(Segment {
                name,
                sequence: BString::from("*"),
                optional: (),
            });
        }

        for (f, t, _) in self.black_edges() {
            links.push(Link {
                from_segment: f as usize,
                from_orient: Orientation::Forward,
                to_segment: t as usize,
                to_orient: Orientation::Forward,
                overlap: BString::from("0M"),
                optional: (),
            });
        }
        /*
        let black = self.black_edges();
        for (f, _t, _w) in black {
            let id = f as usize;
            let seg = Segment {
                name: id,
                sequence: BString::from("*"),
                optional: (),
            };
            segments.push(seg);
        }

        let gray = self.gray_edges();
        for (f, t, _w) in gray {
            let link = Link {
                from_segment: f as usize,
                from_orient: Orientation::Forward,
                to_segment: t as usize,
                to_orient: Orientation::Forward,
                overlap: BString::from(""),
                optional: (),
            };
            links.push(link);
        }

        segments.sort_by(|a, b| a.name.cmp(&b.name));
        segments.dedup_by(|a, b| a.name == b.name);
        links.sort_by(|f, t| f.from_segment.cmp(&t.from_segment));
        */

        GFA {
            header: Header {
                version: Some("1.0".into()),
                optional: (),
            },
            segments,
            links,
            containments: vec![],
            paths: vec![],
        }
    }

    pub fn to_gfa_bstring(&self) -> GFA<BString, ()> {
        let mut segments = Vec::new();
        let mut links = Vec::new();

        for id in self.graph.nodes() {
            let name = BString::from(id.to_string());
            segments.push(Segment {
                name,
                sequence: BString::from("*"),
                optional: (),
            });
        }

        for (f, t, w) in self.black_edges() {
            links.push(Link {
                from_segment: BString::from(f.to_string()),
                from_orient: Orientation::Forward,
                to_segment: BString::from(t.to_string()),
                to_orient: Orientation::Forward,
                overlap: BString::from("0M"),
                // overlap: BString::from(w.black.to_string()),
                optional: (),
            });
        }

        segments.sort_by(|a, b| a.name.cmp(&b.name));
        segments.dedup_by(|a, b| a.name == b.name);
        links.sort_by(|f, t| f.from_segment.cmp(&t.from_segment));

        /*
        for (f, t, _) in self.gray_edges() {
            links.push(Link {
                from_segment: BString::from(f.to_string()),
                from_orient: Orientation::Forward,
                to_segment: BString::from(t.to_string()),
                to_orient: Orientation::Forward,
                overlap: BString::from("0M"),
                optional: (),
            });
        }
        let black = self.black_edges();
        for (f, _t, _w) in black {
            let id = f as usize;
            let seg = Segment {
                name: id,
                sequence: BString::from("*"),
                optional: (),
            };
            segments.push(seg);
        }

        let gray = self.gray_edges();
        for (f, t, _w) in gray {
            let link = Link {
                from_segment: f as usize,
                from_orient: Orientation::Forward,
                to_segment: t as usize,
                to_orient: Orientation::Forward,
                overlap: BString::from(""),
                optional: (),
            };
            links.push(link);
        }

        segments.sort_by(|a, b| a.name.cmp(&b.name));
        segments.dedup_by(|a, b| a.name == b.name);
        links.sort_by(|f, t| f.from_segment.cmp(&t.from_segment));
        */

        GFA {
            header: Header {
                version: Some("1.0".into()),
                optional: (),
            },
            segments,
            links,
            containments: vec![],
            paths: vec![],
        }
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
