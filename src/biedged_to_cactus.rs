use crate::biedgedgraph::*;

use std::collections::{BTreeMap, HashSet};

use three_edge_connected as t_e_c;

/// STEP 1: Contract all gray edges
pub fn contract_all_gray_edges(
    biedged: &mut BiedgedGraph,
    proj_map: &mut BTreeMap<u64, u64>,
) {
    while biedged.gray_edge_count() > 0 {
        let (from, to, _w) = biedged.gray_edges().next().unwrap();
        let kept = biedged.contract_edge(from, to).unwrap();
        proj_map.insert(from, kept);
        proj_map.insert(to, kept);
    }
}

/// STEP 2: Find 3-edge connected components
pub fn find_3_edge_connected_components(
    biedged: &BiedgedGraph,
) -> Vec<Vec<usize>> {
    let edges = biedged
        .graph
        .all_edges()
        .flat_map(|(a, b, w)| {
            std::iter::repeat((a as usize, b as usize)).take(w.black)
        })
        .collect::<Vec<_>>();

    let graph = t_e_c::Graph::from_edges(edges.into_iter());

    let (components, _) = t_e_c::find_components(&graph.graph);

    let components: Vec<_> =
        components.into_iter().filter(|c| c.len() > 1).collect();

    let components = graph.invert_components(components);

    components
}

// merge the detected components

pub fn merge_components(
    biedged: &mut BiedgedGraph,
    components: Vec<Vec<usize>>,
    proj_map: &mut BTreeMap<u64, u64>,
) {
    for comp in components {
        let mut iter = comp.into_iter();
        let head = iter.next().unwrap() as u64;
        for other in iter {
            let other = other as u64;
            let prj = biedged.merge_vertices(head, other).unwrap();
            proj_map.insert(head, prj);
            proj_map.insert(other, prj);
        }
    }
}

/// STEP 3: Find loops and contract edges inside them

pub fn find_cycles(biedged: &BiedgedGraph) -> Vec<Vec<u64>> {
    let mut cycles = Vec::new();

    cycles
}

// Find loops using a DFS
fn find_loops(biedged: &mut BiedgedGraph) -> Vec<Vec<(u64, u64)>> {
    let mut loops: Vec<Vec<_>> = Vec::new();
    let mut dfs_stack: Vec<u64> = Vec::new();
    let mut visited_nodes_set: HashSet<u64> = HashSet::new();

    let start_node = biedged.graph.nodes().min().unwrap();
    dfs_stack.push(start_node);

    let mut parent = start_node;
    let mut current_loop: Vec<u64> = Vec::new();
    let mut loops_: Vec<Vec<u64>> = Vec::new();
    // let mut current_component: Vec<BiedgedEdge> = Vec::new();
    while let Some(id) = dfs_stack.pop() {
        current_loop.push(id);

        for node in biedged.graph.neighbors(id) {
            if !visited_nodes_set.contains(&node) {
                dfs_stack.push(node);
            }
            // current_component.push(BiedgedEdge { from: id, to: node });
            // else if node != parent
            //     && current_loop.iter().find(|&n| n == &node).is_some()
            else {
                // else {
                // Found loop
                let mut current_component: Vec<_> = Vec::new();
                current_component.push((id, node));
                loops.push(current_component);
                // current_component = Vec::new();

                current_loop.push(node);
                loops_.push(current_loop);
                current_loop = Vec::new();
            }
        }
        parent = id;
        visited_nodes_set.insert(id);
    }
    for each_loop in loops_ {
        // println!("loop length: {}", each_loop.len());
        for node in each_loop {
            print!(" {}", node);
        }
        println!();
    }

    loops
}

pub fn contract_loops(biedged: &mut BiedgedGraph) {
    // let loop_edges: Vec<Vec<BiedgedEdge>>;
    let loop_edges = find_loops(biedged);
    println!("found {} loops", loop_edges.len());
    for each_loop in loop_edges {
        for (from, to) in each_loop {
            // print!(" {}, {}", edge.from, edge.to);
            println!("contracting {}, {}", from, to);
            biedged.contract_edge(from, to);
        }
        // println!();
    }
}

// ----------------------------------- TESTS -------------------------------
#[cfg(test)]
mod tests {
    use super::*;

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
        graph.add_edge(10, 11, BiedgedWeight::black(1));

        // Node b
        graph.add_edge(20, 21, BiedgedWeight::black(1));

        // Node c
        graph.add_edge(30, 31, BiedgedWeight::black(1));

        // Node d
        graph.add_edge(40, 41, BiedgedWeight::black(1));

        // Node e
        graph.add_edge(50, 51, BiedgedWeight::black(1));

        // Node f
        graph.add_edge(60, 61, BiedgedWeight::black(1));

        // Node g
        graph.add_edge(70, 71, BiedgedWeight::black(1));

        // Node h
        graph.add_edge(80, 81, BiedgedWeight::black(1));

        // Node i
        graph.add_edge(90, 91, BiedgedWeight::black(1));

        // Node j
        graph.add_edge(100, 101, BiedgedWeight::black(1));

        // Node k
        graph.add_edge(110, 111, BiedgedWeight::black(1));

        // Node l
        graph.add_edge(120, 121, BiedgedWeight::black(1));

        // Node m
        graph.add_edge(130, 131, BiedgedWeight::black(1));

        // Node n
        graph.add_edge(140, 141, BiedgedWeight::black(1));

        // Node o
        graph.add_edge(150, 151, BiedgedWeight::black(1));

        // Node p
        graph.add_edge(160, 161, BiedgedWeight::black(1));

        // Node q
        graph.add_edge(170, 171, BiedgedWeight::black(1));

        // Node r
        graph.add_edge(180, 181, BiedgedWeight::black(1));

        // a-b
        graph.add_edge(11, 20, BiedgedWeight::gray(1));
        // a-c
        graph.add_edge(11, 30, BiedgedWeight::gray(1));

        // b-d
        graph.add_edge(21, 40, BiedgedWeight::gray(1));
        // c-d
        graph.add_edge(31, 40, BiedgedWeight::gray(1));

        // d-e
        graph.add_edge(41, 50, BiedgedWeight::gray(1));
        // d-f
        graph.add_edge(41, 60, BiedgedWeight::gray(1));

        // e-g
        graph.add_edge(51, 70, BiedgedWeight::gray(1));

        // f-g
        graph.add_edge(61, 70, BiedgedWeight::gray(1));

        // f-h
        graph.add_edge(61, 80, BiedgedWeight::gray(1));

        // g-k
        graph.add_edge(71, 110, BiedgedWeight::gray(1));
        // g-l
        graph.add_edge(71, 120, BiedgedWeight::gray(1));

        // h-i
        graph.add_edge(81, 90, BiedgedWeight::gray(1));
        // h-j
        graph.add_edge(81, 100, BiedgedWeight::gray(1));

        // i-j
        graph.add_edge(91, 100, BiedgedWeight::gray(1));

        // j-l
        graph.add_edge(101, 120, BiedgedWeight::gray(1));

        // k-l
        graph.add_edge(110, 120, BiedgedWeight::gray(1));

        // l-m
        graph.add_edge(121, 130, BiedgedWeight::gray(1));

        // m-n
        graph.add_edge(131, 140, BiedgedWeight::gray(1));
        // m-o
        graph.add_edge(131, 150, BiedgedWeight::gray(1));

        // n-p
        graph.add_edge(141, 160, BiedgedWeight::gray(1));

        // o-p
        graph.add_edge(151, 160, BiedgedWeight::gray(1));

        // p-m
        graph.add_edge(161, 130, BiedgedWeight::gray(1));

        // p-q
        graph.add_edge(161, 170, BiedgedWeight::gray(1));
        // p-r
        graph.add_edge(161, 180, BiedgedWeight::gray(1));

        graph
    }

    #[test]
    fn simple_contract_all_gray_edges() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();

        //First Handlegraph node
        graph.add_node(10);
        graph.add_node(11);
        graph.add_edge(10, 11, BiedgedWeight::black(1));

        //Second Handlegraph node
        graph.add_node(20);
        graph.add_node(21);
        graph.add_edge(20, 21, BiedgedWeight::black(1));

        //Third Handlegraph node
        graph.add_node(30);
        graph.add_node(31);
        graph.add_edge(30, 31, BiedgedWeight::black(1));

        //Forth Handlegraph node
        graph.add_node(40);
        graph.add_node(41);
        graph.add_edge(40, 41, BiedgedWeight::black(1));

        //Add Handlegraph edges
        graph.add_edge(11, 20, BiedgedWeight::gray(1));
        graph.add_edge(11, 30, BiedgedWeight::gray(1));
        graph.add_edge(21, 40, BiedgedWeight::gray(1));
        graph.add_edge(31, 40, BiedgedWeight::gray(1));

        let mut proj_map = BTreeMap::new();
        contract_all_gray_edges(&mut graph, &mut proj_map);

        use petgraph::dot::{Config, Dot};

        println!(
            "{:#?}",
            Dot::with_config(&graph.graph, &[Config::NodeNoLabel])
        );
        // println!("Nodes: {:#?}", graph.get_nodes());
        println!("Gray_edges {:#?}", graph.gray_edges().collect::<Vec<_>>());
        println!("Black_edges {:#?}", graph.black_edges().collect::<Vec<_>>());

        assert!(graph.graph.node_count() == 4);
        assert_eq!(graph.black_edge_count(), 4);

        // NOTE: petgraph does not actually support multiple edges between two given nodes
        // however, they are allowed in Biedged Graphs. For this reason it is better to use
        // the count_edges function provided by the EdgeFunctions trait.
        assert!(graph.graph.edge_count() == 3);
    }

    #[test]
    fn paper_contract_all_gray_edges() {
        let mut graph: BiedgedGraph = graph_from_paper();

        let mut proj_map = BTreeMap::new();
        contract_all_gray_edges(&mut graph, &mut proj_map);

        assert_eq!(graph.gray_edge_count(), 0);
        assert_eq!(
            graph.black_edge_count(),
            18,
            "Expected 18 black edges, is actually {:#?}",
            graph.black_edge_count()
        );
    }

    #[test]
    fn edge_contraction_projection_map() {
        use crate::biedgedgraph::{find_projection, recover_node_name};
        use bstr::BString;
        use gfa::{
            gfa::{name_conversion::NameMap, GFA},
            parser::GFAParser,
        };

        let parser = GFAParser::new();
        let bstr_gfa: GFA<bstr::BString, ()> =
            parser.parse_file("./paper.gfa").unwrap();

        let name_map = NameMap::build_from_gfa(&bstr_gfa);
        let gfa = name_map.gfa_bstring_to_usize(&bstr_gfa, false).unwrap();

        let mut graph = BiedgedGraph::from_gfa(&gfa);

        let mut proj_map = BTreeMap::new();
        contract_all_gray_edges(&mut graph, &mut proj_map);

        let proj_names = bstr_gfa
            .segments
            .iter()
            .map(|s| {
                let orig = name_map.map_name(&s.name).unwrap();
                let orig_name = s.name.to_owned();
                let (l, r) = crate::biedgedgraph::split_node_id(orig as u64);
                let l_end = find_projection(&proj_map, l);
                let r_end = find_projection(&proj_map, r);
                let l_end = recover_node_name(&name_map, l_end).unwrap();
                let r_end = recover_node_name(&name_map, r_end).unwrap();
                (orig_name, (l_end, r_end))
            })
            .collect::<Vec<_>>();

        let expected_names: Vec<_> = vec![
            ("a", ("a", "b")),
            ("b", ("b", "d")),
            ("c", ("b", "d")),
            ("d", ("d", "e")),
            ("e", ("e", "g")),
            ("f", ("e", "g")),
            ("g", ("g", "k")),
            ("h", ("g", "i")),
            ("i", ("i", "i")),
            ("j", ("i", "k")),
            ("k", ("k", "k")),
            ("l", ("k", "m")),
            ("m", ("m", "n")),
            ("n", ("n", "p")),
            ("o", ("n", "p")),
            ("p", ("p", "m")),
            ("q", ("m", "q_")),
            ("r", ("m", "r_")),
        ]
        .into_iter()
        .map(|(a, (l, r))| {
            (BString::from(a), (BString::from(l), BString::from(r)))
        })
        .collect();

        assert_eq!(expected_names, proj_names);
    }
}
