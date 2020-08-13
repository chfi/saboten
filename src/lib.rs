use petgraph::prelude::*;
use petgraph::dot::{Dot, Config};

use handlegraph::hashgraph::*;
use handlegraph::handle::*;
use handlegraph::handlegraph::*;
use handlegraph::handle::Direction;

use std::collections::VecDeque;
use std::collections::HashSet;
use std::path::PathBuf;

use gfa::parser::parse_gfa;

/// Convert a GFA to a biedged graph if file exists
/// otherwise return None
fn gfa_to_biedged_graph(path : &PathBuf) -> Option<Graph::<String, String>> {
    if let Some(gfa) = parse_gfa(path) {
        let graph = HashGraph::from_gfa(&gfa);
        Some(handlegraph_to_biedged_graph(&graph))
    } else {
        None
    }
}

/// Convert a handlegraph to a biedged graph
fn handlegraph_to_biedged_graph(graph: &HashGraph) -> Graph::<String, String> {
    let mut biedged : Graph::<String, String> = Graph::new();

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

    while let Some(curr_node) = q.pop_front() {

        if visited_nodes.contains(&curr_node) {
            continue;
        }

        // For each node in the Handlegraph, there will be two nodes in the biedged graph
        // each representing one of the two sides
        let id_1 = format!("{}.{}",curr_node,1);
        let node_1 = biedged.add_node(id_1);
        
        let id_2 = format!("{}.{}",curr_node,2);
        let node_2 = biedged.add_node(id_2);
        
        // The two nodes are connected
        let id_edge = format!("{}",curr_node);
        biedged.add_edge(node_1, node_2, id_edge);

        // Look for neighbors in the Handlegraph, add edges in the biedged graph
        let current_handle = Handle::pack(curr_node, false);
        for neighbor in handle_edges_iter(graph, current_handle, Direction::Right) {
            
            // Add first node for neighbor
            let id_neighbor = format!("{}.{}",neighbor.id(),1);
            let neighbor_node_biedged = biedged.add_node(id_neighbor);

            // Add edge from neighbor to 
            let id_edge = format!("{}->{}",curr_node, neighbor.id());
            biedged.add_edge(node_2, neighbor_node_biedged, id_edge);

            // Add to queue
            q.push_back(neighbor.id());
        }

        visited_nodes.insert(curr_node);
    }

    biedged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let path = PathBuf::from("./input/samplePath3.gfa");
        let biedged = gfa_to_biedged_graph(&path).unwrap();
        println!("{:?}", Dot::with_config(&biedged, &[Config::EdgeNoLabel]));
    }
}
