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
trait GraphInternals {
    // Node functions
    fn add_node(&mut self, id: u64);
    fn remove_node(&mut self, id: u64);

    // Edge functions
    fn add_edge(&mut self, from: u64, to: u64, edge_type: BiedgedEdge);
    fn remove_edge(&mut self, from: u64, to: u64);
    fn remove_edge_with_type(&mut self, from: u64, to: u64, edge_type: BiedgedEdge);
    
    fn get_gray_edges(&self) -> Vec<BiedgedEdge>;
    fn get_black_edges(&self) -> Vec<BiedgedEdge>;
}

enum BiedgedEdgeType {
    Black,
    Gray,
}
// An edge of the biedged graph
#[derive(Debug)]
struct BiedgedEdge {
    from : u64,
    to : u64,
    //edge_type : BiedgedEdgeType
}

/// Biedged graph class
struct BiedgedGraph {
    graph : UnGraphMap::<u64, String>,
    black_edges : Vec<BiedgedEdge>,
    gray_edges : Vec<BiedgedEdge>
}

//TODO: continue OOP


/// Convert a GFA to a biedged graph if file exists
/// otherwise return None
fn gfa_to_biedged_graph(path : &PathBuf) -> Option<BiedgedGraph> {
    if let Some(gfa) = parse_gfa(path) {
        let graph = HashGraph::from_gfa(&gfa);
        Some(handlegraph_to_biedged_graph(&graph))
    } else {
        None
    }
}

/// Convert a handlegraph to a biedged graph
fn handlegraph_to_biedged_graph(graph: &HashGraph) -> BiedgedGraph {
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

    //Store black and grey edges
    let mut black_edges : Vec<BiedgedEdge> = Vec::new();
    let mut gray_edges : Vec<BiedgedEdge> = Vec::new();

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
        gray_edges : gray_edges
    };

    biedged_graph
}

/// STEP 1: Contract edges
fn contract_edges(biedged_graph : &mut BiedgedGraph) {
    for edge in &biedged_graph.gray_edges {

        let start_node = edge.from;
        let end_node = edge.to;

        // Store adjacent nodes
        let mut adjacency_nodes : Vec<u64> = Vec::new();
        let adj_1 : Vec<u64> = biedged_graph.graph.edges(start_node).map(|x| x.0).collect();
        let adj_2 : Vec<u64> = biedged_graph.graph.edges(end_node).map(|x| x.0).collect();
        adjacency_nodes.extend(adj_1.iter());
        adjacency_nodes.extend(adj_2.iter());
        
        // Remove existing nodes, edges will also be removed
        biedged_graph.graph.remove_node(start_node);
        biedged_graph.graph.remove_node(end_node);

        // Add new node
        let node = biedged_graph.graph.add_node(start_node);

        for adj_node in adjacency_nodes {
            biedged_graph.graph.add_edge(node, adj_node, format!("New edge"));
        }

        // Remove the edge from the graph
        //biedged_graph.graph.remove_edge(edge.from, edge.to);
    }

    //biedged_graph.gray_edges.remove(0);
    //assert!(biedged_graph.gray_edges.is_empty());
}

/// Print the biedged graph to a .dot file. This file can then be used by
/// various tools (i.e. Graphviz) to produce a graphical representation of the graph
/// i.e. dot -Tpng graph.dot -o graph.png
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
        println!("{:?}", Dot::with_config(&biedged.graph, &[Config::EdgeNoLabel]));
        println!("{:?}", graph);
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
