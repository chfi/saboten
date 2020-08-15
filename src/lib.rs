use petgraph::prelude::*;
use petgraph::dot::{Dot, Config};

use handlegraph::hashgraph::*;
use handlegraph::handle::*;
use handlegraph::handlegraph::*;
use handlegraph::handle::Direction;

use std::collections::VecDeque;
use std::collections::HashSet;
use std::path::PathBuf;
use std::fs::File;
use std::io::Write;
use std::io;

use gfa::parser::parse_gfa;

// Traits
trait NodeFunctions {
    // Node functions
    fn add_node(&mut self, id: u64) -> Option<u64>;
    fn remove_node(&mut self, id: u64) -> Option<u64>;
    fn remove_nodes_incident_with_edge(&mut self, edge: &BiedgedEdge) -> Option<Vec<BiedgedEdge>>;

    fn get_adjacent_nodes(&self, id: u64) -> Option<Vec<u64>>;

    fn get_nodes(&self) -> &Vec<BiedgedNode>;
    fn get_nodes_mut(&mut self) -> &mut Vec<BiedgedNode>;
}

trait EdgeFunctions {
    // Edge functions
    fn add_edge(&mut self, from: u64, to: u64, edge_type: BiedgedEdgeType) -> Option<BiedgedEdge>;
    fn remove_edge(&mut self, from: u64, to: u64) -> Option<BiedgedEdge>;
    fn remove_edges_incident_to_node(&mut self, node: &BiedgedNode) -> Option<Vec<BiedgedEdge>>;
    fn contract_edge(&mut self, edge: &BiedgedEdge);
    
    // Immutable getter/setter
    fn get_gray_edges(&self) -> &Vec<BiedgedEdge>;
    fn get_black_edges(&self) -> &Vec<BiedgedEdge>;

    // Mutable getter/setters
    fn get_gray_edges_mut(&mut self) -> &mut Vec<BiedgedEdge>;
    fn get_black_edges_mut(&mut self) -> &mut Vec<BiedgedEdge>;
}

impl NodeFunctions for BiedgedGraph {
    fn add_node(&mut self, id: u64) -> Option<u64> {
        Some(self.graph.add_node(id))
    }

    fn remove_node(&mut self, id: u64) -> Option<u64> {
        if self.graph.contains_node(id) {
            self.graph.remove_node(id);

            // Remove all incident edges from Vecs
            self.black_edges.retain(|x| !(x.from == id || x.to == id));
            self.gray_edges.retain(|x| !(x.from == id || x.to == id));

            Some(id)
        } else {
            None
        }
    }

    //TODO: needs testing
    fn remove_nodes_incident_with_edge(&mut self, edge: &BiedgedEdge) -> Option<Vec<BiedgedEdge>> {
        let mut removed_edges : Vec<BiedgedEdge> = Vec::new();
        if self.black_edges.contains(edge) {
            self.remove_node(edge.from);
            self.remove_node(edge.to);
            removed_edges = self.black_edges.iter().filter(|x| *x == edge).map(|x| *x).collect();
            self.black_edges.retain(|x| !(x == edge));
            Some(removed_edges)
        } else if self.gray_edges.contains(edge) {
            self.remove_node(edge.from);
            self.remove_node(edge.to);
            removed_edges = self.gray_edges.iter().filter(|x| *x == edge).map(|x| *x).collect();
            self.gray_edges.retain(|x| !(x == edge));
            Some(removed_edges)
        } else {
            None
        }
    }

    fn get_adjacent_nodes(&self, id: u64) -> Option<Vec<u64>> {

        if self.graph.contains_node(id) {
            let adjacent_nodes : Vec<u64> = self.graph.edges(id).map(|x| x.0).collect();
            Some(adjacent_nodes)
        } else {
            None
        }
    }

    fn get_nodes(&self) -> &Vec<BiedgedNode> {
        self.nodes.as_ref()
    }

    fn get_nodes_mut(&mut self) -> &mut Vec<BiedgedNode> {
        self.nodes.as_mut()
    }
}

impl EdgeFunctions for BiedgedGraph {

    // TODO: think which string to use
    fn add_edge(&mut self, from: u64, to: u64, edge_type: BiedgedEdgeType) -> Option<BiedgedEdge> {
        
        let edge_to_add = BiedgedEdge{from: from, to: to, };

        if edge_type == BiedgedEdgeType::Black {
            self.graph.add_edge(from, to, String::from(""));
            self.black_edges.push(BiedgedEdge{from: from, to: to});
            Some(edge_to_add)
        } else if edge_type == BiedgedEdgeType::Gray {
            self.graph.add_edge(from, to, String::from(""));
            self.gray_edges.push(BiedgedEdge{from: from, to: to});
            Some(edge_to_add)
        } else {
            None
        }
    }
    
    fn remove_edge(&mut self, from: u64, to: u64) -> Option<BiedgedEdge> {
        
        let edge_to_remove = BiedgedEdge{from: from, to: to};

        if self.graph.contains_edge(from, to) && self.black_edges.contains(&edge_to_remove) {
            self.graph.remove_edge(from, to);
            self.black_edges.iter().position(|x| *x == edge_to_remove).map(|e| self.black_edges.remove(e));
            Some(edge_to_remove)
        } else if self.graph.contains_edge(from, to) && self.gray_edges.contains(&edge_to_remove) {
            self.graph.remove_edge(from, to);
            self.gray_edges.iter().position(|x| *x == edge_to_remove).map(|e| self.gray_edges.remove(e));
            Some(edge_to_remove)
        } else {
            None
        }
    }

    fn remove_edges_incident_to_node(&mut self, node: &BiedgedNode) -> Option<Vec<BiedgedEdge>> {
        
        if self.nodes.contains(node) {
            self.graph.remove_node(node.id); // edges should be removed automatically in petgraph
            let mut incident_edges : Vec<BiedgedEdge> = Vec::new();
            let mut black_edges : Vec<BiedgedEdge> = self.black_edges.iter().filter(|x| x.from == node.id || x.to == node.id).map(|x| *x).collect();
            let mut gray_edges : Vec<BiedgedEdge> = self.gray_edges.iter().filter(|x| x.from == node.id || x.to == node.id).map(|x| *x).collect();
            incident_edges.append(&mut black_edges);
            incident_edges.append(&mut gray_edges);
            
            Some(incident_edges)
        } else {
            None
        }

    }

    fn contract_edge(&mut self, edge: &BiedgedEdge) {
        let mut adjacent_nodes : Vec<u64> = Vec::new();
        let mut first_node_adjacent_nodes : Vec<u64> = self.get_adjacent_nodes(edge.from).unwrap();
        let mut second_node_adjacent_nodes : Vec<u64> = self.get_adjacent_nodes(edge.to).unwrap();
        adjacent_nodes.append(&mut first_node_adjacent_nodes);
        adjacent_nodes.append(&mut second_node_adjacent_nodes);

        self.remove_node(edge.from);
        self.remove_node(edge.to);
        // All adjacent edges will also be removed

        //TODO: decide which id to use
        let added_node = self.add_node(100).unwrap();

        for adj_node in adjacent_nodes {
            self.add_edge(added_node, adj_node, BiedgedEdgeType::Gray);
        }

    }

    fn get_gray_edges(&self) -> &Vec<BiedgedEdge> {
        self.gray_edges.as_ref()
    }
    fn get_black_edges(&self) -> &Vec<BiedgedEdge> {
        self.black_edges.as_ref()
    }

    fn get_gray_edges_mut(&mut self) -> &mut Vec<BiedgedEdge> {
        self.gray_edges.as_mut()
    }
    fn get_black_edges_mut(&mut self) -> &mut Vec<BiedgedEdge> {
        self.black_edges.as_mut()
    }
}

#[derive(PartialEq)]
enum BiedgedEdgeType {
    Black,
    Gray,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BiedgedEdge {
    from : u64,
    to : u64,
}
#[derive(Debug, Clone, Copy)]
struct BiedgedEdgeGray {
    from : u64,
    to : u64,
    handlegraph_edge : (u64,u64)
}
#[derive(Debug, Clone, Copy)]
struct BiedgedEdgeBlack {
    from : u64,
    to : u64,
    handlegraph_node : u64
}

/// Biedged graph class
pub struct BiedgedGraph {
    graph : UnGraphMap::<u64, String>,
    black_edges : Vec<BiedgedEdge>,
    gray_edges : Vec<BiedgedEdge>,
    nodes : Vec<BiedgedNode>
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BiedgedNode {
    id : u64
}


/// Convert a GFA to a biedged graph if file exists
/// otherwise return None
pub fn gfa_to_biedged_graph(path : &PathBuf) -> Option<BiedgedGraph> {
    if let Some(gfa) = parse_gfa(path) {
        let graph = HashGraph::from_gfa(&gfa);
        Some(handlegraph_to_biedged_graph(&graph))
    } else {
        None
    }
}

/// Convert a handlegraph to a biedged graph
pub fn handlegraph_to_biedged_graph(graph: &HashGraph) -> BiedgedGraph {
    let mut biedged : UnGraphMap::<u64, String> = UnGraphMap::new();

    // Create queue
    // NOTE: this is a Queue based implementation, this was done
    // in order not to get a stack overflow
    let mut q: VecDeque<NodeId> = VecDeque::new();

    // Start from the node with the lowest id
    // will probably always be 1, but this is safer
    let node_id = graph.min_id;

    // Store which nodes have already been visited
    let mut visited_nodes : HashSet<NodeId> = HashSet::new();

    // Insert first value
    q.push_back(node_id);

    // Store black and grey edges
    let mut black_edges : Vec<BiedgedEdge> = Vec::new();
    let mut gray_edges : Vec<BiedgedEdge> = Vec::new();
    // Store nodes
    let mut nodes : Vec<BiedgedNode> = Vec::new();

    while let Some(curr_node) = q.pop_front() {

        if visited_nodes.contains(&curr_node) {
            continue;
        }

        let current_handle = Handle::pack(curr_node, false);
        let left_id : u64 = current_handle.as_integer();
        let right_id : u64 = current_handle.flip().as_integer();

        // For each node in the Handlegraph, there will be two nodes in the biedged graph
        // each representing one of the two sides
        let node_1 = biedged.add_node(left_id);
        let node_2 = biedged.add_node(right_id);
        
        // The two nodes are connected
        let id_edge = format!("B: {}",current_handle.unpack_number());
        biedged.add_edge(node_1, node_2, id_edge);

        // Add nodes to vec
        nodes.push(BiedgedNode{id:left_id});
        nodes.push(BiedgedNode{id:right_id});

        // Add edge to black edges
        black_edges.push(BiedgedEdge{from: node_1, to: node_2});

        // Look for neighbors in the Handlegraph, add edges in the biedged graph
        for neighbor in handle_edges_iter(graph, current_handle, Direction::Right) {
            
            // Add first node for neighbor
            let neighbor_node_biedged = biedged.add_node(neighbor.as_integer());

            // Add edge from neighbor to 
            let id_edge = format!("G: {}->{}",curr_node, neighbor.id());
            biedged.add_edge(node_2, neighbor_node_biedged, id_edge);

            // Add edge to gray edges
            gray_edges.push(BiedgedEdge{from: node_2, to: neighbor_node_biedged});

            // Add to queue
            q.push_back(neighbor.id());
        }

        visited_nodes.insert(curr_node);
    }

    // Create Biedged graph
    let biedged_graph = BiedgedGraph {
        graph : biedged,
        black_edges : black_edges,
        gray_edges : gray_edges,
        nodes : nodes
    };

    biedged_graph
}

/// STEP 1: Contract edges
fn contract_edges(biedged_graph : &mut BiedgedGraph) {

    // Create queue
    // NOTE: this is a Queue based implementation, this was done
    // in order not to get a stack overflow
    let mut q: VecDeque<BiedgedEdge> = VecDeque::new();

    // Add all gray edges to queue
    biedged_graph.gray_edges.iter().for_each(|x| q.push_back(*x));

    while let Some(edge) = q.pop_front() {
        
        let start_node = edge.from;
        let end_node = edge.to;

        // Store adjacent nodes
        let mut adjacency_nodes : Vec<u64> = Vec::new();
        let adj_1 : Vec<u64> = biedged_graph.graph.edges(start_node).map(|x| x.0).collect();
        let adj_2 : Vec<u64> = biedged_graph.graph.edges(end_node).map(|x| x.0).collect();
        adjacency_nodes.extend(adj_1.iter());
        adjacency_nodes.extend(adj_2.iter());
        adjacency_nodes.dedup();
        
        // Remove existing nodes, edges will also be removed
        biedged_graph.graph.remove_node(start_node);
        biedged_graph.graph.remove_node(end_node);

        //let to_remove_black : Vec<&BiedgedEdge> = biedged_graph.black_edges.iter().filter(|x| x.from == start_node || x.to == start_node).collect();
        //let to_remove_gray : Vec<&BiedgedEdge> = biedged_graph.gray_edges.iter().filter(|x| x.from == end_node || x.to == end_node).collect();
        //biedged_graph.black_edges.retain(|x| !(x.from == start_node || x.to == start_node));
        //biedged_graph.gray_edges.retain(|x| !(x.from == start_node || x.to == start_node));
        

        // Add new node
        let node = biedged_graph.graph.add_node(start_node);

        for adj_node in adjacency_nodes {
            biedged_graph.graph.add_edge(node, adj_node, format!("New edge"));
            //biedged_graph.gray_edges.push(BiedgedEdge{from: node, to: adj_node});
            //q.push_back(BiedgedEdge{from: node, to: adj_node});
        }

        // Update gray edges in queue with new node
        for gray_edge in q.iter_mut() {
            if gray_edge.from == start_node {
                gray_edge.from = node;
            } else if gray_edge.to == end_node {
                gray_edge.to = node;
            }
        }


        // Update gray edges in graph with new node
        for gray_edge in biedged_graph.gray_edges.iter_mut() {
            if gray_edge.from == start_node {
                gray_edge.from = node;
            } else if gray_edge.to == end_node {
                gray_edge.to = node;
            }
        }

    }

}

/// Print the biedged graph to a .dot file. This file can then be used by
/// various tools (i.e. Graphviz) to produce a graphical representation of the graph
/// (i.e. dot -Tpng graph.dot -o graph.png)
fn biedged_to_dot(graph : &BiedgedGraph, path : &PathBuf) -> std::io::Result<()> {
    let mut f = File::create(path).unwrap();
    //let output = format!("{}", Dot::with_config(&graph.graph, &[Config::EdgeNoLabel]));
    let output = format!("{}", Dot::with_config(&graph.graph, &[Config::NodeNoLabel]));
    f.write_all(&output.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use handlegraph::mutablehandlegraph::MutableHandleGraph;

    #[test]
    fn it_works() {
        let path = PathBuf::from("./input/samplePath3.gfa");
        let graph = HashGraph::from_gfa(&parse_gfa(&path).unwrap());
        let biedged = gfa_to_biedged_graph(&path).unwrap();
        //println!("{:?}", Dot::with_config(&biedged.graph, &[Config::EdgeNoLabel]));
        //println!("{:?}", graph);
    }

    #[test]
    fn it_works_2() {
        let mut graph = HashGraph::new();
        let h1 = graph.append_handle("a");
        let h2 = graph.append_handle("c");
        graph.create_edge(&Edge(h1,h2));
        let biedged = handlegraph_to_biedged_graph(&graph);
        println!("{:#?}", Dot::with_config(&biedged.graph, &[Config::NodeNoLabel]));
        println!("Black edges: {:#?}", biedged.black_edges);
        println!("Gray edges: {:#?}", biedged.gray_edges);
        //println!("Biedged {:#?}",Dot::new(&biedged));

        println!();

        println!("{:?}",graph);
        println!("{:?}",h1.as_integer());
        println!("{:?}",h1.flip().as_integer());
        
        println!();
        //NodeId
        println!("{:?}",h1.id());
        //NodeId as u64
        println!("{:?}",h1.unpack_number());
        //NodeId does not change
        println!("{:?}",h1.flip().unpack_number());

        // println!("");
        // println!("{:?}",h2.as_integer());
        // println!("{:?}",h2.flip().as_integer());
        // println!("{:?}",h2.id());
        // println!("{:?}",h2.unpack_number()); 
    }

    #[test]
    fn it_works_3() {
        let mut graph = HashGraph::new();
        let h1 = graph.append_handle("a");
        let h2 = graph.append_handle("c");
        let h3 = graph.append_handle("t");
        let h4 = graph.append_handle("g");
        
        graph.create_edge(&Edge(h1,h2));
        graph.create_edge(&Edge(h2,h4));
        graph.create_edge(&Edge(h1,h3));
        graph.create_edge(&Edge(h3,h4));
        
        let mut biedged = handlegraph_to_biedged_graph(&graph);
        biedged_to_dot(&biedged, &PathBuf::from("original.dot"));
        // println!("{:#?}", Dot::with_config(&biedged.graph, &[Config::NodeNoLabel]));
        contract_edges(&mut biedged);
        biedged_to_dot(&biedged, &PathBuf::from("modified.dot"));
        // println!("Modified \n {:#?}", Dot::with_config(&biedged.graph, &[Config::NodeNoLabel]));

        // // Print to file
        // let mut f = File::create("example1.dot").unwrap();
        // let output = format!("{}", Dot::with_config(&biedged.graph, &[Config::EdgeNoLabel]));
        // f.write_all(&output.as_bytes());
    }
}
