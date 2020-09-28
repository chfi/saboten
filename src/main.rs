use fnv::{FnvHashMap, FnvHashSet};
use std::{collections::HashMap, path::PathBuf};

use bstr::{BString, ByteSlice};

use structopt::StructOpt;

use rs_cactusgraph::{biedged_to_cactus, biedgedgraph::*};

use petgraph::unionfind::UnionFind;

use gfa::{
    gfa::{name_conversion::NameMap, GFA},
    parser::GFAParser,
};

#[derive(StructOpt, Debug)]
struct Opt {
    in_gfa: PathBuf,
}

fn main() {
    let opt = Opt::from_args();

    let mut gfa_name_map = None;

    let gfa = {
        let parser = GFAParser::new();
        let bstr_gfa: GFA<BString, ()> =
            parser.parse_file(&opt.in_gfa).unwrap();

        let name_map = NameMap::build_from_gfa(&bstr_gfa);

        let usize_gfa =
            name_map.gfa_bstring_to_usize(&bstr_gfa, false).unwrap();

        gfa_name_map = Some(name_map);

        usize_gfa
    };

    let mut be_graph = BiedgedGraph::from_gfa(&gfa);

    let orig_graph = be_graph.clone();

    let mut union_find: UnionFind<usize> =
        UnionFind::new(be_graph.graph.node_count());

    println!(" --- Contracting gray edges --- ");
    biedged_to_cactus::contract_all_gray_edges(&mut be_graph, &mut union_find);

    println!(" --- Finding 3EC-components --- ");
    let components =
        biedged_to_cactus::find_3_edge_connected_components(&be_graph);

    println!("  {} components", components.len());
    for c in components.iter() {
        println!("    {:?}", c);
    }

    println!(" --- Merging 3EC-components --- ");
    biedged_to_cactus::merge_components(
        &mut be_graph,
        components,
        &mut union_find,
    );

    let cactus_graph_projections = union_find.clone();

    println!(" --- Finding cycles --- ");
    let cycles = biedged_to_cactus::find_cycles(&be_graph);

    let mut cycle_map: FnvHashMap<(u64, u64), Vec<usize>> =
        FnvHashMap::default();

    for (i, cycle) in cycles.iter().enumerate() {
        for (a, b) in cycle.iter() {
            cycle_map.entry((*a, *b)).or_default().push(i);
        }
        println!("{}\t{:?}", i, cycle);
    }

    let mut bridge_forest = be_graph.clone();

    let mut bridge_forest_projections = union_find.clone();

    println!(" --- Constructing bridge forest --- ");
    biedged_to_cactus::contract_simple_cycles(
        &mut bridge_forest,
        &cycles,
        &mut bridge_forest_projections,
    );

    println!(" --- Constructing cactus tree --- ");

    let mut cactus_tree = be_graph.clone();

    let chain_vertices =
        biedged_to_cactus::build_cactus_tree(&mut cactus_tree, &cycles);

    let mut chain_edges = Vec::new();
    let mut bridge_edges = Vec::new();
    cactus_tree.graph.all_edges().for_each(|(a, b, _)| {
        if biedged_to_cactus::is_chain_edge(
            &cactus_tree,
            &chain_vertices,
            &cactus_graph_projections,
            a,
            b,
        ) {
            chain_edges.push((a, b))
        }
    });

    orig_graph.graph.all_edges().for_each(|(a, b, w)| {
        if w.black > 0
            && biedged_to_cactus::is_bridge_edge(
                &cactus_tree,
                &cactus_graph_projections,
                a,
                b,
            )
        {
            bridge_edges.push((a, b))
        }
    });
    /*
    let (bridge_edges, chain_edges): (Vec<_>, Vec<_>) =
        cactus_tree.graph.all_edges().partition(|(a, b, _)| {
            biedged_to_cactus::is_bridge_edge(
                &cactus_tree,
                &cactus_graph_projections,
                *a,
                *b,
            )
        });
    */

    let mut cactus_graph_inverse: FnvHashMap<usize, Vec<usize>> =
        FnvHashMap::default();

    let cactus_reps = cactus_graph_projections.clone().into_labeling();
    for (i, k) in cactus_reps.iter().enumerate() {
        cactus_graph_inverse.entry(*k).or_default().push(i);
    }

    println!();
    println!("Cactus graph vertex projections");
    for (k, v) in cactus_graph_inverse.iter() {
        println!("{} mapped from {:?}", k, v);
    }

    let mut bridge_forest_inverse: FnvHashMap<usize, Vec<usize>> =
        FnvHashMap::default();

    let bridge_reps = bridge_forest_projections.into_labeling();
    for (i, k) in bridge_reps.iter().enumerate() {
        bridge_forest_inverse.entry(*k).or_default().push(i);
    }

    println!();
    println!("Bridge forest vertex projections");
    for (k, v) in bridge_forest_inverse.iter() {
        println!("{} mapped from {:?}", k, v);
    }

    let name_map = gfa_name_map.unwrap();
    for seg in gfa.segments.iter() {
        let id = seg.name as u64;
        let name = name_map.inverse_map_name(seg.name).unwrap();
        let name = name.to_str().unwrap();
        let (a, b) = id_to_black_edge(id);
        let (x, y) = projected_edge(&cactus_graph_projections, a, b);
        let cyc = cycle_map.get(&(x, y));
        println!(
            "{} projects to edge ({}, {}) = {},\tcycles {:?}",
            name,
            x,
            y,
            be_graph.graph.contains_edge(x, y),
            cyc,
        );
    }

    /*
    let chain_pairs = vec![
        (24, 30),
        (25, 30),
        (24, 25),
        (24, 31),
        (25, 31),
        (30, 31),
        (14, 15),
        (14, 18),
        (14, 19),
        (15, 18),
        (15, 19),
        (18, 19),
    ];

    for (a, b) in chain_pairs.iter() {
        println!(
            "{}, {} is chain pair:\t{}",
            a,
            b,
            biedged_to_cactus::is_chain_pair(
                &cactus_graph_projections,
                &cycle_map,
                *a,
                *b
            )
        );
    }
    */

    // let mut chain_pairs = Vec::new();
    let mut chain_pairs = FnvHashSet::default();

    println!("----- chain edges -----\n");
    for (a, b) in chain_edges.iter() {
        // println!("  {}, {}", a, b);
        let ix = b - cactus_tree.max_net_vertex - 1;
        let chain_vx = chain_vertices[ix as usize];
        let cycle = &cycles[chain_vx.1];
        println!("chain edge {} {}", a, b);
        println!("  {:?}", cycle);

        for (x, y) in cycle.iter() {
            let x = *x as usize;
            let y = *y as usize;
            // let y = *y as usize;
            let orig_xs = cactus_graph_inverse.get(&x).unwrap();
            // let orig_ys = cactus_graph_inverse.get(&y).unwrap();

            if let Some(w) = be_graph.graph.edge_weight(x as u64, y as u64) {
                println!("  -- {}", w.black);
            }

            println!("  orig {:?}", orig_xs);
            let filtered: Vec<_> = orig_xs
                .iter()
                .filter(|&&u| {
                    let (w, v) = end_to_black_edge(u as u64);
                    let w = w as usize;
                    let v = v as usize;
                    println!("  {} - {}, {}", u, w, v);
                    if orig_xs.contains(&w) && orig_xs.contains(&v) {
                        false
                    } else {
                        true
                    }
                })
                .copied()
                .collect();

            // println!("  from {:?}", filtered);
            // println!("   and {:?}", orig_ys);

            // for x_a in orig_xs.iter() {
            // for x_b in orig_xs.iter() {

            for x_a in filtered.iter() {
                for x_b in filtered.iter() {
                    if x_a != x_b {
                        let x_a = *x_a as u64;
                        let x_b = *x_b as u64;
                        let is_chain_pair = biedged_to_cactus::is_chain_pair(
                            &cactus_graph_projections,
                            &cycle_map,
                            x_a,
                            x_b,
                        );
                        if is_chain_pair {
                            let a = x_a.min(x_b);
                            let b = x_a.max(x_b);
                            chain_pairs.insert((a, b));
                            println!("   {} {} -> {}", x_a, x_b, is_chain_pair);
                        }
                    }
                }
            }
            /*
            println!(
                "   {} {} -> {}",
                x,
                y,
                biedged_to_cactus::is_chain_pair(
                    &cactus_graph_projections,
                    &cycle_map,
                    *x,
                    *y
                )
            );
            */
        }
        // println!("  {}, {:?}", a, cycle);

        // let  = cycle.iter().map(|(x, _)| x).collect::<Vec<_>>();

        /*
        println!(
            "{}, {} is chain pair:\t{}",
            a,
            b,
            biedged_to_cactus::is_chain_pair(
                &cactus_graph_projections,
                &cycle_map,
                *a,
                *b
            )
        );
        */
    }

    println!(" --- (possible) chain pairs ---");
    for (a, b) in chain_pairs.iter() {
        println!("  {}, {}", a, b);
    }

    println!("bridge edges");
    for (a, b) in bridge_edges.iter() {
        println!("  {}, {}", a, b);
    }

    println!("max_net_vertex: {}", orig_graph.max_net_vertex);
    println!("max_chain_vertex: {}", orig_graph.max_chain_vertex);

    println!("max_net_vertex: {}", cactus_tree.max_net_vertex);
    println!("max_chain_vertex: {}", cactus_tree.max_chain_vertex);

    /*
    let bridge_edges =

    let bridge_edges = vec![
        (0, 1),
        (1, 6),
        (6, 7),
        (7, 22),
        (22, 23),
        (2, 3),
        (4, 5),
        (32, 33),
        (34, 35),
    ];

    for (a, b) in bridge_edges.iter() {
        println!(
            "{}, {} is bridge edge:\t{}",
            a,
            b,
            biedged_to_cactus::is_bridge_edge(
                &cactus_tree,
                &cactus_graph_projections,
                *a,
                *b
            )
        );
    }
    */

    let pairs = vec![(1, 6), (7, 22), (25, 30), (24, 31), (16, 17)];

    for (a, b) in pairs {
        let path = biedged_to_cactus::snarl_cactus_tree_path(
            &cactus_tree,
            &cactus_graph_projections,
            a,
            b,
        );
        // paths.push(path);
        println!(" pair {}, {}", a, b);
        println!(" -- {:?}", path);
    }

    /*
    let nodes: Vec<_> = gfa
        .segments
        .iter()
        .map(|seg| id_to_black_edge(segment.name as u64))
        .collect();

    */
    /*
        let mut cycle_inverse: HashMap<usize, Vec<usize>> = HashMap::new();

        let cycle_reps = cycle_projections.into_labeling();
        for (i, k) in bridge_reps.iter().enumerate() {
            cycle_inverse.entry(*k).or_default().push(i);
        }

        println!("Cactus graph cycle projections");
        for (k, v) in cycle_inverse.iter() {
            println!("{}
        }
    */
}
