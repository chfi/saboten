use crate::biedgedgraph::*;

use handlegraph::hashgraph::*;

use std::collections::HashSet;
use std::path::PathBuf;


use gfa::parser::parse_gfa;

use three_edge_connected::*;
use three_edge_connected::graph::Graph;
use three_edge_connected::state::State;
use std::collections::HashMap;
use std::collections::BTreeMap;

use three_edge_connected::graph::AdjacencyList;
use bstr::{BStr, BString};

/// STEP 1: Contract all gray edges
pub fn contract_all_gray_edges(biedged: &mut BiedgedGraph) {
    while !biedged.get_gray_edges().is_empty() {
        let curr_edge = biedged.get_gray_edges().get(0).unwrap().clone();
        biedged.contract_edge(curr_edge.from, curr_edge.to);
    }
}

/// STEP 2: Find 3-edge connected components
/// makes use of chfi's rs-3-edge, which can be found at:
/// https://github.com/chfi/rs-3-edge

/// Generate a Graph as defined in rs-3-edge from a biedged graph
fn from_biedged_graph(biedged: &mut BiedgedGraph) -> Graph {
    let mut graph: BTreeMap<usize, AdjacencyList> = BTreeMap::new();
    let mut name_map: HashMap<BString, usize> = HashMap::new();
    let mut inv_names = Vec::new();

    let mut get_ix = |name: &BStr| {
        if let Some(ix) = name_map.get(name) {
            *ix
        } else {
            let ix = name_map.len();
            name_map.insert(name.into(), ix);
            inv_names.push(name.into());
            ix
        }
    };

    // Black edges
    for black_edge in biedged.get_black_edges() {
        let from_ix = get_ix(BString::from(format!("{:#?}", black_edge.from)).as_ref());
        let to_ix = get_ix(BString::from(format!("{:#?}", black_edge.to)).as_ref());

        graph.entry(from_ix).or_default().push(to_ix);
        graph.entry(to_ix).or_default().push(from_ix);
    }

    Graph { graph, inv_names }
}

/// Obtain connected components of length greater than 1
fn obtain_complex_components(inv_names: &[BString], components: &[Vec<usize>]) -> Vec<Vec<u64>> {
    let mut complex_components: Vec<Vec<u64>> = Vec::new();
    for component in components {
        let mut current_component: Vec<u64> = Vec::new();
        if component.len() > 1 {
            component.iter().enumerate().for_each(|(i, j)| {
                let temp: String = format!("{}", inv_names[*j]);
                current_component.push(temp.parse::<u64>().unwrap());
            });
            complex_components.push(current_component);
        }
    }
    complex_components
}

fn merge_3_connected_components(biedged: &mut BiedgedGraph, components: &Vec<Vec<u64>>) {
    for component in components {
        merge_nodes_in_component(biedged, component);
    }
}
fn merge_nodes_in_component(biedged: &mut BiedgedGraph, component: &Vec<u64>) {
    let mut adj_vertices: HashSet<u64> = HashSet::new();

    for nodeId in component {
        for node in biedged.get_adjacent_nodes(*nodeId).unwrap() {
            if !component.contains(&node) {
                adj_vertices.insert(node);
            }
        }
        // Remove all edges incident to nodeId
        biedged.remove_edges_incident_to_node(*nodeId);
        // Remove node
        biedged.remove_node(*nodeId);
    }

    // TODO: Decide which nodeId to use
    biedged.add_node(*component.get(0).unwrap());
    for node in adj_vertices {
        // TODO: keep track if edge was incoming or outcoming
        biedged.add_edge(100, node, BiedgedEdgeType::Black);
    }
}

pub fn find_3_edge_connected_components(biedged: &mut BiedgedGraph) {
    let graph = from_biedged_graph(biedged);
    let mut state = State::initialize(&graph.graph);
    algorithm::three_edge_connect(&graph.graph, &mut state);
    let components = obtain_complex_components(&graph.inv_names, state.components());
    merge_3_connected_components(biedged,&components);
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
        let adj_nodes = biedged.get_adjacent_nodes(id).unwrap();
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

fn contract_loop_edges(biedged: &mut BiedgedGraph, loop_edges: Vec<Vec<BiedgedEdge>>) {
    for loop_components in loop_edges {
        for edge in loop_components {
            biedged.contract_edge(edge.from, edge.to);
        }
    }
}

pub fn contract_loops(biedged: &mut BiedgedGraph) {
    let mut loop_edges: Vec<Vec<BiedgedEdge>> = Vec::new();
    loop_edges = find_loops(biedged);
    contract_loop_edges(biedged, loop_edges);
}
