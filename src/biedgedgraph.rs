use petgraph::prelude::*;

use gfa::{
    gfa::{name_conversion::NameMap, Header, Link, Orientation, Segment, GFA},
    parser::GFAParser,
};

use bstr::BString;

use std::{
    collections::BTreeMap,
    io::Write,
    ops::{Add, AddAssign, Sub, SubAssign},
    path::PathBuf,
};

pub fn split_node_id(n: u64) -> (u64, u64) {
    let left = n;
    let right = std::u64::MAX - left;
    (left, right)
}

pub fn recover_node_id(n: u64) -> u64 {
    if n < std::u64::MAX / 2 {
        n
    } else {
        std::u64::MAX - n
    }
}

pub fn find_projection(proj_map: &BTreeMap<u64, u64>, mut node: u64) -> u64 {
    while let Some(&next) = proj_map.get(&node) {
        if node == next {
            break;
        } else {
            node = next;
        }
    }
    node
}

pub fn project_graph_id(
    proj_map: &BTreeMap<u64, u64>,
    seg_id: usize,
) -> (usize, usize) {
    let (left, right) = split_node_id(seg_id as u64);
    let left = find_projection(proj_map, left);
    let right = find_projection(proj_map, right);
    (left as usize, right as usize)
}

pub fn projected_node_id(n: u64) -> String {
    let not_orig = n > std::u64::MAX / 2;
    let id = recover_node_id(n);
    let mut name = id.to_string();
    if not_orig {
        name.push_str("_");
    }
    name
}

pub fn recover_node_name(name_map: &NameMap, n: u64) -> Option<BString> {
    let not_orig = n > std::u64::MAX / 2;
    let id = recover_node_id(n);
    let mut name: BString = name_map.inverse_map_name(id as usize)?.to_owned();
    if not_orig {
        name.push(b'_');
    }
    Some(name)
}

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
    /// Create an empty biedged graph
    pub fn new() -> BiedgedGraph {
        Default::default()
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

    /// Add the node with the given id to the graph
    pub fn add_node(&mut self, id: u64) -> u64 {
        self.graph.add_node(id)
    }

    /// Add an edge with the provided edge weight. If a corresponding
    /// edge already exists in the graph, the edge weights are added.
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

    /// Merge two vertices into one, such that all the edges incident
    /// to the provided nodes are moved to be incident to the merged
    /// vertex.
    ///
    /// Returns the index of the resulting vertex, or None if either
    /// of the provided vertices were not present in the graph.
    pub fn merge_vertices(&mut self, a: u64, b: u64) -> Option<u64> {
        // We'll always remove the node with a higher ID, for consistency
        let from = a.min(b);
        let to = a.max(b);

        if !self.graph.contains_node(a) || !self.graph.contains_node(b) {
            return None;
        }

        if self.graph.contains_edge(a, b) {
            return self.contract_edge(a, b);
        }

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
            self.add_edge(from, other, w);
        }

        Some(from)
    }

    /// Merge any number of vertices, provided as an iterator of node
    /// IDs, into one. Returns the index of the resulting vertex, or
    /// None if the iterator was empty or any of the vertices were not
    /// present in the graph.
    pub fn merge_many_vertices<I>(&mut self, vertices: I) -> Option<u64>
    where
        I: IntoIterator<Item = u64>,
    {
        let mut vertices = vertices.into_iter();
        let head = vertices.next()?;

        // Collect all the nodes to be deleted, along with their
        // corresponding edges
        let mut edges = Vec::new();
        let mut nodes = Vec::new();

        for v in vertices {
            if !self.graph.contains_node(v) {
                return None;
            }
            let v_edges =
                self.graph.edges(v).map(|(_, other, w)| (other, w.clone()));
            edges.extend(v_edges);
            nodes.push(v);
        }

        for n in nodes {
            self.graph.remove_node(n);
        }

        // Insert the
        for (other, w) in edges {
            self.add_edge(head, other, w);
        }

        Some(head)
    }

    pub fn contract_edge(&mut self, left: u64, right: u64) -> Option<u64> {
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
            self.add_edge(from, other, w);
        }

        if weight.black > 0 {
            let new_weight = BiedgedWeight::black(weight.black);
            self.add_edge(from, from, new_weight);
        }

        if weight.gray > 1 {
            let new_weight = BiedgedWeight::gray(weight.gray - 1);
            self.add_edge(from, from, new_weight);
        }

        Some(from)
    }
}

impl BiedgedGraph {
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
                optional: (),
            });
        }

        segments.sort_by(|a, b| a.name.cmp(&b.name));
        segments.dedup_by(|a, b| a.name == b.name);
        links.sort_by(|f, t| f.from_segment.cmp(&t.from_segment));

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
        assert!(graph.graph.node_count() == 1);
    }

    #[test]
    fn test_add_edge() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.add_node(20);
        graph.add_node(30);

        graph.add_edge(10, 20, BiedgedWeight::black(1));
        assert!(graph.graph.contains_edge(10, 20));

        assert_eq!(graph.black_edge_count(), 1);
        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 0 }),
            graph.graph.edge_weight(10, 20)
        );

        graph.add_edge(20, 30, BiedgedWeight::gray(1));
        assert!(graph.graph.contains_edge(20, 30));
        assert_eq!(graph.gray_edge_count(), 1);

        assert_eq!(
            Some(&BiedgedWeight { black: 0, gray: 1 }),
            graph.graph.edge_weight(20, 30)
        );

        graph.add_edge(20, 30, BiedgedWeight::black(1));

        assert_eq!(
            Some(&BiedgedWeight { black: 1, gray: 1 }),
            graph.graph.edge_weight(20, 30)
        );
    }

    #[test]
    fn test_contract_edge() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();
        graph.add_node(10);
        graph.add_node(20);
        graph.add_node(30);
        graph.add_edge(10, 20, BiedgedWeight::black(1));
        graph.add_edge(10, 30, BiedgedWeight::gray(1));
        graph.add_edge(20, 30, BiedgedWeight::black(1));

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
