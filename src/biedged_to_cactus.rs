use crate::biedgedgraph::*;

use petgraph::unionfind::UnionFind;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use three_edge_connected as t_e_c;

/// STEP 1: Contract all gray edges
pub fn contract_all_gray_edges(
    biedged: &mut BiedgedGraph,
    union_find: &mut UnionFind<usize>,
) {
    while biedged.gray_edge_count() > 0 {
        let (from, to, _w) = biedged.gray_edges().next().unwrap();
        let kept = biedged.contract_edge(from, to, union_find).unwrap();
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
    union_find: &mut UnionFind<usize>,
) {
    for comp in components {
        let mut iter = comp.into_iter();
        let head = iter.next().unwrap() as u64;
        for other in iter {
            biedged.merge_vertices(head, other as u64, union_find);
        }
    }
}

/// Find the simple cycles in a cactus graph and return them. A cycle
/// is represented as a vector of vertices, with the same start and
/// end vertex.
pub fn find_cycles(biedged: &BiedgedGraph) -> Vec<Vec<u64>> {
    let graph = &biedged.graph;

    let mut visited: BTreeSet<u64> = BTreeSet::new();
    let mut parents: BTreeMap<u64, u64> = BTreeMap::new();

    let mut stack: Vec<u64> = Vec::new();

    let mut cycles = Vec::new();
    let mut cycle_ends: Vec<(u64, u64)> = Vec::new();

    for node in graph.nodes() {
        if !visited.contains(&node) {
            stack.push(node);
            while let Some(current) = stack.pop() {
                if !visited.contains(&current) {
                    visited.insert(current);
                    for (_, adj, weight) in graph.edges(current) {
                        if adj == current {
                            for _ in 0..weight.black {
                                cycles.push(vec![current, current]);
                            }
                        } else {
                            if !visited.contains(&adj) {
                                if weight.black == 2 {
                                    cycles.push(vec![current, adj, current]);
                                }
                                stack.push(adj);
                                parents.insert(adj, current);
                            } else {
                                if parents.get(&current) != Some(&adj) {
                                    cycle_ends.push((adj, current));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (start, end) in cycle_ends {
        let mut cycle: Vec<u64> = vec![end];
        let mut current = end;

        while current != start {
            if let Some(parent) = parents.get(&current) {
                cycle.push(*parent);
                current = *parent;
            }
        }

        cycle.push(end);
        cycles.push(cycle);
    }

    cycles
}

/// STEP 3: Find loops and contract edges inside them

/// Contracts each cycle into a single vertex, updating the projection
/// map accordingly.
pub fn contract_simple_cycles(
    biedged: &mut BiedgedGraph,
    cycles: &[Vec<u64>],
    union_find: &mut UnionFind<usize>,
) {
    for cycle in cycles {
        cycle.windows(2).for_each(|vs| {
            assert!(vs.len() == 2);
            let (from, to) = (vs[0], vs[1]);
            biedged.contract_edge(from, to, union_find);
        });
    }
}

/// Adds a chain vertex for each cycle, with edges to each of the
/// elements in the cycle, and removes the edges within the cycle.
/// Returns the IDs of the new chain vertices as a vector.
pub fn build_cactus_tree(
    biedged: &mut BiedgedGraph,
    cycles: &[Vec<u64>],
) -> Vec<u64> {
    let mut chain_vertices = Vec::with_capacity(cycles.len());

    for cycle in cycles {
        assert!(cycle.len() > 1);

        let chain_vx = biedged.new_node();

        cycle.windows(2).for_each(|vs| {
            assert!(vs.len() == 2);
            let (from, to) = (vs[0], vs[1]);
            biedged.add_edge(to, chain_vx, BiedgedWeight::black(1));
            biedged.remove_one_black_edge(from, to);
        });

        chain_vertices.push(chain_vx);
    }

    chain_vertices
}

pub fn black_edge_cycle(
    vx_proj: &UnionFind<usize>,
    cycle_map: &HashMap<u64, Vec<usize>>,
    x: u64,
) -> Option<Vec<usize>> {
    let (l, r) = end_to_black_edge(x);
    let p_l = vx_proj.find(l as usize) as u64;
    let p_r = vx_proj.find(r as usize) as u64;
    let cyc_l = cycle_map.get(&p_l)?;
    let cyc_r = cycle_map.get(&p_r)?;
    let intersection = cyc_l
        .iter()
        .copied()
        .filter(|c| cyc_r.contains(c))
        .collect();
    Some(intersection)
}

pub fn is_chain_pair(
    vx_proj: &UnionFind<usize>,
    cycle_map: &HashMap<u64, Vec<usize>>,
    x: u64,
    y: u64,
) -> bool {
    if !distinct_vertices(x, y) {
        return false;
    }

    let p_x = vx_proj.find(x as usize);
    let p_y = vx_proj.find(y as usize);

    if p_x != p_y {
        return false;
    }

    let x_cycles = black_edge_cycle(vx_proj, cycle_map, x);
    let y_cycles = black_edge_cycle(vx_proj, cycle_map, y);

    if x_cycles.is_none() || y_cycles.is_none() {
        return false;
    }

    x_cycles == y_cycles
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
        let mut add_node = |n: u64| {
            let (l, r) = id_to_black_edge(n);
            graph.add_node(l);
            graph.add_node(r);
            graph.add_edge(l, r, BiedgedWeight::black(1));
        };

        add_node(0);
        add_node(1);
        add_node(2);
        add_node(3);

        let mut add_edge = |l: u64, r: u64| {
            graph.add_edge(l, r, BiedgedWeight::gray(1));
        };

        add_edge(1, 2);
        add_edge(1, 4);
        add_edge(3, 6);
        add_edge(5, 6);

        let mut proj_map = BTreeMap::new();
        contract_all_gray_edges(&mut graph, &mut proj_map);

        assert_eq!(
            graph.graph.edge_weight(0, 1),
            Some(&BiedgedWeight::black(1))
        );
        assert_eq!(
            graph.graph.edge_weight(1, 3),
            Some(&BiedgedWeight::black(2))
        );
        assert_eq!(
            graph.graph.edge_weight(3, 7),
            Some(&BiedgedWeight::black(1))
        );

        assert_eq!(graph.graph.node_count(), 4);
        assert_eq!(graph.black_edge_count(), 4);
        assert_eq!(graph.gray_edge_count(), 0);
        assert_eq!(graph.graph.edge_count(), 3);
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
        use crate::biedgedgraph::{find_projection, projected_node_name};
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
                let (l, r) = crate::biedgedgraph::id_to_black_edge(orig as u64);
                let l_end = find_projection(&proj_map, l);
                let r_end = find_projection(&proj_map, r);
                let l_end = projected_node_name(&name_map, l_end).unwrap();
                let r_end = projected_node_name(&name_map, r_end).unwrap();
                (orig_name, (l_end, r_end))
            })
            .collect::<Vec<_>>();

        let expected_names: Vec<_> = vec![
            ("a", ("a", "a_")),
            ("b", ("a_", "b_")),
            ("c", ("a_", "b_")),
            ("d", ("b_", "d_")),
            ("e", ("d_", "e_")),
            ("f", ("d_", "e_")),
            ("g", ("e_", "g_")),
            ("h", ("e_", "h_")),
            ("i", ("h_", "h_")),
            ("j", ("h_", "g_")),
            ("k", ("g_", "g_")),
            ("l", ("g_", "l_")),
            ("m", ("l_", "m_")),
            ("n", ("m_", "n_")),
            ("o", ("m_", "n_")),
            ("p", ("n_", "l_")),
            ("q", ("l_", "q_")),
            ("r", ("l_", "r_")),
        ]
        .into_iter()
        .map(|(a, (l, r))| {
            (BString::from(a), (BString::from(l), BString::from(r)))
        })
        .collect();

        assert_eq!(expected_names, proj_names);
    }

    fn example_graph() -> BiedgedGraph {
        let mut graph: BiedgedGraph = BiedgedGraph::new();

        for i in 0..=9 {
            graph.add_node(i);
        }

        /*               -i
                 &     &/
        a--b==c--d==f--h--j
               \ |   \ |
                -e    -g
                 &     &

        & self cycles
        - 1 black edge
        = 2 black edges
                */

        let edges = vec![
            (0, 1),
            (1, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (3, 3),
            (3, 4),
            (4, 4),
            (4, 5),
            (4, 5),
            (5, 6),
            (5, 7),
            (6, 6),
            (7, 6),
            (7, 7),
            (7, 8),
            (7, 9),
        ];

        for (a, b) in edges {
            graph.add_edge(a, b, BiedgedWeight::black(1));
        }

        graph
    }

    #[test]
    fn cycle_detection() {
        let graph = example_graph();

        let cycles = find_cycles(&graph);

        assert_eq!(
            cycles,
            vec![
                vec![1, 2, 1],
                vec![4, 4],
                vec![4, 5, 4],
                vec![7, 7],
                vec![6, 6],
                vec![3, 3],
                vec![6, 7, 5, 6],
                vec![3, 4, 2, 3],
            ]
        );
    }

    #[test]
    fn test_build_cactus_tree() {
        let mut graph = example_graph();

        let cycles = find_cycles(&graph);

        let chains = build_cactus_tree(&mut graph, &cycles);

        assert_eq!(cycles.len(), chains.len());

        for (cv, cycle) in chains.iter().zip(cycles.iter()) {
            let chain_edges =
                graph.graph.edges(*cv).map(|x| x.1).collect::<Vec<_>>();

            assert_eq!(&cycle[1..], &chain_edges[..]);
        }
    }
}
