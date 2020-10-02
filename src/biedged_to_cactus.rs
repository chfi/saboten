use crate::{biedgedgraph::*, projection::Projection};

use fnv::{FnvHashMap, FnvHashSet};

use three_edge_connected as t_e_c;

/// STEP 1: Contract all gray edges
pub(crate) fn contract_all_gray_edges(
    biedged: &mut BiedgedGraph,
    projection: &mut Projection,
) {
    while let Some((from, to)) = biedged.next_gray_edge() {
        biedged.contract_edge(from, to, projection).unwrap();
    }
}

/// STEP 2: Find 3-edge connected components
pub(crate) fn find_3_edge_connected_components(
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

    let components = t_e_c::find_components(&graph.graph);

    let components: Vec<_> =
        components.into_iter().filter(|c| c.len() > 1).collect();

    let components = graph.invert_components(components);

    components
}

// merge the detected components

pub(crate) fn merge_components(
    biedged: &mut BiedgedGraph,
    components: Vec<Vec<usize>>,
    projection: &mut Projection,
) {
    for comp in components {
        let mut iter = comp.into_iter();
        let head = iter.next().unwrap() as u64;
        for other in iter {
            let other = other as u64;
            if biedged.graph.contains_node(head)
                && biedged.graph.contains_node(other)
            {
                if biedged.graph.contains_edge(head, other) {
                    biedged.contract_edge(head, other, projection);
                } else {
                    biedged.merge_vertices(head, other, projection);
                }
            }
        }
    }
}

/// Find the simple cycles in a cactus graph and return them. A cycle
/// is represented as a vector of vertices, with the same start and
/// end vertex.
pub(crate) fn find_cycles(biedged: &BiedgedGraph) -> Vec<Vec<(u64, u64)>> {
    let graph = &biedged.graph;

    let mut visited: FnvHashSet<u64> = FnvHashSet::default();
    let mut parents: FnvHashMap<u64, u64> = FnvHashMap::default();

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
                                cycles.push(vec![(current, current)]);
                            }
                        } else {
                            if !visited.contains(&adj) {
                                if weight.black == 2 {
                                    cycles.push(vec![
                                        (current, adj),
                                        (adj, current),
                                    ]);
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
        let mut cycle: Vec<(u64, u64)> = vec![];
        let mut current = end;

        while current != start {
            if let Some(parent) = parents.get(&current) {
                cycle.push((current, *parent));
                current = *parent;
            }
        }

        cycle.push((start, end));
        cycles.push(cycle);
    }

    cycles
}

/// STEP 3: Find loops and contract edges inside them

/// Contracts each cycle into a single vertex, updating the projection
/// map accordingly.
pub(crate) fn contract_simple_cycles(
    biedged: &mut BiedgedGraph,
    cycles: &[Vec<(u64, u64)>],
    projection: &mut Projection,
) {
    for cycle in cycles {
        for &(from, to) in cycle {
            let from = if biedged.graph.contains_node(from) {
                from
            } else {
                projection.find(from)
            };

            let to = if biedged.graph.contains_node(to) {
                to
            } else {
                projection.find(to)
            };

            biedged.merge_vertices(from, to, projection);
        }
    }
}

/// Adds a chain vertex for each cycle, with edges to each of the
/// elements in the cycle, and removes the edges within the cycle.
/// Returns a vector of tuples, where the first element is the chain
/// vertex ID in the graph, and the second element is the index of the
/// corresponding cycle in the provided cycles vector.
pub(crate) fn build_cactus_tree(
    biedged: &mut BiedgedGraph,
    cycles: &[Vec<(u64, u64)>],
) -> (FnvHashMap<(u64, u64), u64>, FnvHashSet<u64>) {
    let mut cycle_chain_map = FnvHashMap::default();
    let mut chain_vertices = FnvHashSet::default();

    for cycle in cycles.iter() {
        let chain_vx = biedged.add_chain_vertex();

        for (from, to) in cycle {
            cycle_chain_map.insert((*from, *to), chain_vx);
            biedged.add_edge(*to, chain_vx, BiedgedWeight::black(1));
            biedged.remove_one_black_edge(*from, *to);
        }

        chain_vertices.insert(chain_vx);
    }

    (cycle_chain_map, chain_vertices)
}

pub(crate) fn snarl_cactus_tree_path(
    cactus_tree: &BiedgedGraph,
    projection: &Projection,
    x: u64,
    y: u64,
) -> Option<Vec<u64>> {
    let p_x = projection.find(x);
    let p_y = projection.find(y);

    let mut path = Vec::new();

    if p_x == p_y {
        // If {x, y} is a chain pair
        path.push(p_x);
    } else {
        // If {x, y} is not a chain pair
        let mut visited: FnvHashSet<u64> = FnvHashSet::default();
        let mut parents: FnvHashMap<u64, u64> = FnvHashMap::default();

        let mut stack: Vec<u64> = Vec::new();

        stack.push(p_x);

        while let Some(current) = stack.pop() {
            if !current != p_y && !visited.contains(&current) {
                visited.insert(current);

                let current_net_vertex = cactus_tree.is_net_vertex(current);
                let neighbors = cactus_tree.graph.neighbors(current);

                let neighbors = neighbors.filter(|&n| {
                    if current_net_vertex {
                        cactus_tree.is_chain_vertex(n)
                            // && n != prev
                            && n != current
                    } else {
                        cactus_tree.is_net_vertex(n)
                            // && n != prev
                            && n != current
                    }
                });

                for n in neighbors {
                    if !visited.contains(&n) {
                        stack.push(n);
                        parents.insert(n, current);
                    }
                }
            }
        }

        let mut current = p_y;
        let mut path_ = vec![p_y];
        while current != p_x {
            let parent = parents.get(&current)?;
            path_.push(*parent);
            current = *parent;
        }

        path_.reverse();
        path.append(&mut path_);
    }

    Some(path)
}

pub(crate) fn net_graph_black_edge_walk(
    biedged: &BiedgedGraph,
    x: u64,
    y: u64,
) -> bool {
    let start = x;
    let end = y;
    let adj_end = opposite_vertex(y);

    let mut visited: FnvHashSet<u64> = FnvHashSet::default();
    let mut stack: Vec<u64> = Vec::new();

    stack.push(start);

    while let Some(current) = stack.pop() {
        if current == end {
            return true;
        }

        if !visited.contains(&current) {
            visited.insert(current);

            let edges = biedged.graph.edges(current);

            if current == start || current == adj_end {
                for (_, n, w) in edges {
                    if w.black > 0 {
                        if !visited.contains(&n) {
                            stack.push(n);
                        }
                    }
                }
            } else {
                for (_, n, _) in edges {
                    if !visited.contains(&n) && n != end {
                        stack.push(n);
                    }
                }
            }
        }
    }

    false
}

// ----------------------------------- TESTS -------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn graph_from_paper() -> BiedgedGraph {
        let edges = vec![
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (3, 5),
            (4, 6),
            (5, 6),
            (5, 7),
            (6, 10),
            (6, 11),
            (7, 8),
            (7, 9),
            (8, 9),
            (9, 11),
            (10, 11),
            (11, 12),
            (12, 13),
            (12, 14),
            (13, 15),
            (14, 15),
            (15, 16),
            (15, 17),
            (15, 12),
        ];

        BiedgedGraph::from_directed_edges(edges).unwrap()
    }

    #[test]
    fn simple_contract_all_gray_edges() {
        let mut graph: BiedgedGraph = BiedgedGraph::new();

        let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3)];

        let mut graph = BiedgedGraph::from_directed_edges(edges).unwrap();

        let mut proj = Projection::new_for_biedged_graph(&graph);

        contract_all_gray_edges(&mut graph, &mut proj);

        let a = proj.projected(0);
        let b = proj.projected(1);
        let c = proj.projected(3);
        let d = proj.projected(7);

        assert_eq!(
            graph.graph.edge_weight(a, b),
            Some(&BiedgedWeight::black(1))
        );
        assert_eq!(
            graph.graph.edge_weight(c, d),
            Some(&BiedgedWeight::black(1))
        );
        assert_eq!(
            graph.graph.edge_weight(b, c),
            Some(&BiedgedWeight::black(2))
        );

        assert_eq!(graph.graph.node_count(), 4);
        assert_eq!(graph.black_edge_count(), 4);
        assert_eq!(graph.gray_edge_count(), 0);
        assert_eq!(graph.graph.edge_count(), 3);
    }

    #[test]
    fn paper_contract_all_gray_edges() {
        let mut graph: BiedgedGraph = graph_from_paper();

        let mut proj = Projection::new_for_biedged_graph(&graph);
        contract_all_gray_edges(&mut graph, &mut proj);

        assert_eq!(graph.gray_edge_count(), 0);
        assert_eq!(
            graph.black_edge_count(),
            18,
            "Expected 18 black edges, is actually {:#?}",
            graph.black_edge_count()
        );
        assert_eq!(graph.graph.node_count(), 12);

        let inv_map = proj.mut_get_inverse();
    }

    #[test]
    fn edge_contraction_projection() {
        use crate::biedgedgraph::{id_to_black_edge, segment_split_name};
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

        let mut proj = Projection::new_for_biedged_graph(&graph);

        contract_all_gray_edges(&mut graph, &mut proj);

        let proj_names = bstr_gfa
            .segments
            .iter()
            .map(|s| {
                let orig = name_map.map_name(&s.name).unwrap();
                let orig_name = s.name.to_owned();
                let (l, r) = id_to_black_edge(orig as u64);
                let l_end = proj.projected(l);
                let r_end = proj.projected(r);
                let l_end = segment_split_name(&name_map, l_end).unwrap();
                let r_end = segment_split_name(&name_map, r_end).unwrap();
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
            ("g", ("e_", "k_")),
            ("h", ("e_", "h_")),
            ("i", ("h_", "h_")),
            ("j", ("h_", "k_")),
            ("k", ("k_", "k_")),
            ("l", ("k_", "p_")),
            ("m", ("p_", "m_")),
            ("n", ("m_", "n_")),
            ("o", ("m_", "n_")),
            ("p", ("n_", "p_")),
            ("q", ("p_", "q_")),
            ("r", ("p_", "r_")),
        ]
        .into_iter()
        .map(|(a, (l, r))| {
            (BString::from(a), (BString::from(l), BString::from(r)))
        })
        .collect();

        assert_eq!(expected_names, proj_names);
    }

    fn example_graph() -> BiedgedGraph {
        /*               -i
                 &     &/
        a--b==c--e==f--h--j
               \ |   \ |
                -d    -g
                 &     &

        & self cycles
        - 1 black edge
        = 2 black edges
                */

        let mut graph: BiedgedGraph = BiedgedGraph::new();

        for i in 0..=9 {
            graph.add_node(i);
        }

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

        graph.max_net_vertex = (graph.graph.node_count() - 1) as u64;
        graph.max_chain_vertex = graph.max_net_vertex;

        graph
    }

    #[test]
    fn cycle_detection() {
        let graph = example_graph();

        let cycles = find_cycles(&graph);

        assert_eq!(
            cycles,
            vec![
                vec![(1, 2), (2, 1)],
                vec![(4, 4)],
                vec![(4, 5), (5, 4)],
                vec![(7, 7)],
                vec![(6, 6)],
                vec![(3, 3)],
                vec![(6, 7), (7, 5), (5, 6)],
                vec![(3, 4), (4, 2), (2, 3)],
            ]
        );
    }

    #[test]
    fn test_build_cactus_tree() {
        let mut graph = example_graph();

        let cycles = find_cycles(&graph);

        let (cycle_chain_map, chain_vertices) =
            build_cactus_tree(&mut graph, &cycles);

        assert_eq!(cycles.len(), chain_vertices.len());

        for (edge, chain_vx) in cycle_chain_map.iter() {
            let chain_edges = graph
                .graph
                .edges(*chain_vx)
                .map(|x| x.1)
                .collect::<Vec<_>>();

            assert!(chain_edges.contains(&edge.0));
            assert!(chain_edges.contains(&edge.1));
        }
    }
}
