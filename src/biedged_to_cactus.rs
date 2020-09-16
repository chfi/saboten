use crate::biedgedgraph::*;

use std::collections::{BTreeMap, HashMap, HashSet};

use three_edge_connected as t_e_c;

use bstr::{BStr, BString};

/// STEP 1: Contract all gray edges
pub fn contract_all_gray_edges(biedged: &mut BiedgedGraph) {
    while biedged.gray_edge_count() > 0 {
        let (from, to, _) = biedged.gray_edges().next().unwrap();
        biedged.contract_edge(from, to);
    }
}

/// STEP 2: Find 3-edge connected components
/// makes use of chfi's rs-3-edge, which can be found at:
/// https://github.com/chfi/rs-3-edge
fn merge_3_connected_components(
    biedged: &mut BiedgedGraph,
    components: &[Vec<usize>],
) {
    for component in components {
        merge_nodes_in_component(biedged, component);
    }
}
fn merge_nodes_in_component(biedged: &mut BiedgedGraph, component: &[usize]) {
    let mut adj_vertices: HashSet<u64> = HashSet::new();

    for node_id in component {
        let node_id = *node_id as u64;
        for node in biedged.get_adjacent_nodes(node_id) {
            if !component.contains(&(node as usize)) {
                adj_vertices.insert(node);
            }
        }
        // Remove all edges incident to nodeId
        biedged.remove_edges_incident_to_node(node_id);
        // Remove node
        biedged.remove_node(node_id);
    }

    // TODO: Decide which nodeId to use
    biedged.add_node(*component.get(0).unwrap() as u64);
    for node in adj_vertices {
        // TODO: keep track if edge was incoming or outcoming
        biedged.add_edge(100, node, BiedgedEdgeType::Black);
    }
}

pub fn find_3_edge_connected_components(biedged: &mut BiedgedGraph) {
    let graph = t_e_c::Graph::from_edges(
        biedged
            .black_edges()
            .map(|(from, to, _)| (from as usize, to as usize)),
    );

    let components = t_e_c::find_components(&graph.graph);

    merge_3_connected_components(biedged, components.as_ref());
}

/// STEP 3: Find loops and contract edges inside them

// Find loops using a DFS
fn find_loops(biedged: &mut BiedgedGraph) -> Vec<Vec<BiedgedEdge>> {
    let mut loops: Vec<Vec<BiedgedEdge>> = Vec::new();
    let mut dfs_stack: Vec<u64> = Vec::new();
    let mut visited_nodes_set: HashSet<u64> = HashSet::new();

    let start_node = biedged.get_nodes().iter().map(|x| x.id).min().unwrap();
    dfs_stack.push(start_node);

    while let Some(id) = dfs_stack.pop() {
        let adj_nodes = biedged.get_adjacent_nodes(id);
        for node in adj_nodes {
            if !visited_nodes_set.contains(&node) {
                dfs_stack.push(node);
            } else {
                // Found loop
                let mut current_component: Vec<BiedgedEdge> = Vec::new();
                current_component.push(BiedgedEdge { from: id, to: node });
                loops.push(current_component);
            }
        }
        visited_nodes_set.insert(id);
    }

    loops
}

fn contract_loop_edges(
    biedged: &mut BiedgedGraph,
    loop_edges: Vec<Vec<BiedgedEdge>>,
) {
    for loop_components in loop_edges {
        for edge in loop_components {
            biedged.contract_edge(edge.from, edge.to);
        }
    }
}

pub fn contract_loops(biedged: &mut BiedgedGraph) {
    let loop_edges: Vec<Vec<BiedgedEdge>>;
    loop_edges = find_loops(biedged);
    contract_loop_edges(biedged, loop_edges);
}

// ----------------------------------- TESTS -------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::biedgedgraph::*;

    fn graph_from_paper() -> BiedgedGraph {
        let mut graph = BiedgedGraph::new();

        // Add nodes
        for i in 1..=18 {
            let n = 10 * i;
            graph.add_node(n);
            graph.add_node(n + 1);
        }

        // Add edges

        // Node a
        graph.add_edge(10, 11, BiedgedEdgeType::Black);

        // Node b
        graph.add_edge(20, 21, BiedgedEdgeType::Black);

        // Node c
        graph.add_edge(30, 31, BiedgedEdgeType::Black);

        // Node d
        graph.add_edge(40, 41, BiedgedEdgeType::Black);

        // Node e
        graph.add_edge(50, 51, BiedgedEdgeType::Black);

        // Node f
        graph.add_edge(60, 61, BiedgedEdgeType::Black);

        // Node g
        graph.add_edge(70, 71, BiedgedEdgeType::Black);

        // Node h
        graph.add_edge(80, 81, BiedgedEdgeType::Black);

        // Node i
        graph.add_edge(90, 91, BiedgedEdgeType::Black);

        // Node j
        graph.add_edge(100, 101, BiedgedEdgeType::Black);

        // Node k
        graph.add_edge(110, 111, BiedgedEdgeType::Black);

        // Node l
        graph.add_edge(120, 121, BiedgedEdgeType::Black);

        // Node m
        graph.add_edge(130, 131, BiedgedEdgeType::Black);

        // Node n
        graph.add_edge(140, 141, BiedgedEdgeType::Black);

        // Node o
        graph.add_edge(150, 151, BiedgedEdgeType::Black);

        // Node p
        graph.add_edge(160, 161, BiedgedEdgeType::Black);

        // Node q
        graph.add_edge(170, 171, BiedgedEdgeType::Black);

        // Node r
        graph.add_edge(180, 181, BiedgedEdgeType::Black);

        // a-b
        graph.add_edge(11, 20, BiedgedEdgeType::Gray);
        // a-c
        graph.add_edge(11, 30, BiedgedEdgeType::Gray);

        // b-d
        graph.add_edge(21, 40, BiedgedEdgeType::Gray);
        // c-d
        graph.add_edge(31, 40, BiedgedEdgeType::Gray);

        // d-e
        graph.add_edge(41, 50, BiedgedEdgeType::Gray);
        // d-f
        graph.add_edge(41, 60, BiedgedEdgeType::Gray);

        // e-g
        graph.add_edge(51, 70, BiedgedEdgeType::Gray);

        // f-g
        graph.add_edge(61, 70, BiedgedEdgeType::Gray);

        // f-h
        graph.add_edge(61, 80, BiedgedEdgeType::Gray);

        // g-k
        graph.add_edge(71, 110, BiedgedEdgeType::Gray);
        // g-l
        graph.add_edge(71, 120, BiedgedEdgeType::Gray);

        // h-i
        graph.add_edge(81, 90, BiedgedEdgeType::Gray);
        // h-j
        graph.add_edge(81, 100, BiedgedEdgeType::Gray);

        // i-j
        graph.add_edge(91, 100, BiedgedEdgeType::Gray);

        // j-l
        graph.add_edge(101, 120, BiedgedEdgeType::Gray);

        // k-l
        graph.add_edge(110, 120, BiedgedEdgeType::Gray);

        // l-m
        graph.add_edge(121, 130, BiedgedEdgeType::Gray);

        // m-n
        graph.add_edge(131, 140, BiedgedEdgeType::Gray);
        // m-o
        graph.add_edge(131, 150, BiedgedEdgeType::Gray);

        // n-p
        graph.add_edge(141, 160, BiedgedEdgeType::Gray);

        // o-p
        graph.add_edge(151, 160, BiedgedEdgeType::Gray);

        // p-m
        graph.add_edge(161, 130, BiedgedEdgeType::Gray);

        // p-q
        graph.add_edge(161, 170, BiedgedEdgeType::Gray);
        // p-r
        graph.add_edge(161, 180, BiedgedEdgeType::Gray);

        graph
    }

    /*
    #[test]
    fn simple_contract_all_gray_edges() {
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

        use petgraph::dot::{Config, Dot};

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

    #[test]
    fn paper_contract_all_gray_edges() {
        let mut graph: BiedgedGraph = graph_from_paper();
        contract_all_gray_edges(&mut graph);

        assert!(graph.get_gray_edges().len() == 0);
        assert!(
            graph.get_black_edges().len() == 18,
            "Expected 18 black edges, is actually {:#?}",
            graph.get_black_edges().len()
        );
    }
    */
}
