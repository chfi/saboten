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
fn gfa_to_biedged_graph(path : &PathBuf) -> Option<UnGraphMap::<u64, String>> {
    if let Some(gfa) = parse_gfa(path) {
        let graph = HashGraph::from_gfa(&gfa);
        Some(handlegraph_to_biedged_graph(&graph))
    } else {
        None
    }
}

/// Convert a handlegraph to a biedged graph
fn handlegraph_to_biedged_graph(graph: &HashGraph) -> UnGraphMap::<u64, String> {
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

        // Look for neighbors in the Handlegraph, add edges in the biedged graph
        for neighbor in handle_edges_iter(graph, current_handle, Direction::Right) {
            
            // Add first node for neighbor
            let neighbor_node_biedged = biedged.add_node(neighbor.as_integer());

            // Add edge from neighbor to 
            let id_edge = format!("G:{}->{}",curr_node, neighbor.id());
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
    use handlegraph::mutablehandlegraph::MutableHandleGraph;

    #[test]
    fn it_works() {
        let path = PathBuf::from("./input/samplePath3.gfa");
        let graph = HashGraph::from_gfa(&parse_gfa(&path).unwrap());
        let biedged = gfa_to_biedged_graph(&path).unwrap();
        println!("{:?}", Dot::with_config(&biedged, &[Config::EdgeNoLabel]));
        println!("{:?}", graph);
    }

    #[test]
    fn it_works_2() {
        let mut graph = HashGraph::new();
        let h1 = graph.append_handle("a");
        let h2 = graph.append_handle("c");
        graph.create_edge(&Edge(h1,h2));
        let biedged = handlegraph_to_biedged_graph(&graph);
        println!("{:#?}", Dot::with_config(&biedged, &[Config::NodeNoLabel]));
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
}
