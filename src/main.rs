use fnv::{FnvHashMap, FnvHashSet};
use std::{collections::HashMap, path::PathBuf};

use bstr::{BString, ByteSlice};

use structopt::StructOpt;

use rs_cactusgraph::{
    biedged_to_cactus,
    biedgedgraph::*,
    cactusgraph::{BiedgedWrapper, BridgeForest, CactusGraph, CactusTree},
    netgraph::NetGraph,
    projection::Projection,
};

use gfa::{
    gfa::{name_conversion::NameMap, Orientation, GFA},
    parser::GFAParser,
};

#[derive(StructOpt, Debug)]
struct Opt {
    in_gfa: PathBuf,
}


fn example_graph_2() -> BiedgedGraph {
    let edges = vec![
        (0, 1),
        (0, 13),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (7, 8),
        (7, 12),
        (8, 9),
        (8, 10),
        (9, 11),
        (10, 11),
        (11, 12),
        (12, 13),
    ];

    let graph = BiedgedGraph::from_directed_edges(edges).unwrap();

    graph
}

fn main() {
    /*
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
    */
    let mut be_graph = example_graph_2();

    let mut projection = Projection::new_for_biedged_graph(&be_graph);

    let orig_graph = be_graph.clone();

    let mut cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);

    let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);

    let chain_edges = cactus_tree.chain_edges();

    println!(" --------- cycles");
    for cycle in cactus_graph.cycles.iter() {
        println!("{:?}", cycle);
    }
    println!(" --------- cycles end");

    for (a, (b, c)) in chain_edges.iter() {
        println!("{} - {:?} {}", a, b, c);
    }

    for x in cactus_tree.chain_vertices.iter() {
        println!("  - {:?}", x);
    }

    let chain_pairs = cactus_tree.find_chain_pairs();

    for ((a, b), c) in chain_pairs.iter() {
        let p_x = cactus_tree.projected_node(*a);
        println!("  -- {} -- {}, {} \t {:?}", p_x, a, b, c);
    }

    println!("{} chain pairs", chain_pairs.len());

    let chain_net_graphs: Vec<((u64, u64), NetGraph, Vec<u64>)> = chain_pairs
        .iter()
        .map(|(&(a, b), c)| {
            let net_graph = cactus_tree.build_net_graph(a, b).unwrap();
            println!(" {} {}\t in chain pair {:?}", a, b, c);
            ((a, b), net_graph, c.clone())
        })
        .collect();

    let mut chain_edge_labels: FnvHashMap<(u64, u64, u64), bool> =
        FnvHashMap::default();

    let mut proj_chain_edge_labels: FnvHashMap<(u64, u64, u64), bool> =
        FnvHashMap::default();

    println!("{} net graphs", chain_net_graphs.len());
    for (chain_pair, net_graph, chain) in chain_net_graphs.iter() {
        let result = net_graph.is_ultrabubble();
        for c_vx in chain {
            let p_x = cactus_tree.projected_node(chain_pair.0);
            let a = chain_pair.0;
            let b = chain_pair.1;
            let key = (a, b, *c_vx as u64);
            println!(
                "net graph\t{:?}  \t{} {} {:?} - {}",
                chain_pair, p_x, c_vx, key, result
            );
            chain_edge_labels.insert(key, result);

            let p_a = cactus_tree.projected_node(a);
            let p_b = cactus_tree.projected_node(b);
            let key_2 = (p_a, p_b, *c_vx as u64);
            proj_chain_edge_labels.insert(key_2, result);

            // println!(
            //     "net graph\t{:?}  \t{} {} {:?} - {}",
            //     chain_pair, p_x, c_vx, key_2, result
            // );
        }
    }

    let mut net_chain_map: FnvHashMap<(u64, u64), Vec<(u64, u64, u64)>> =
        FnvHashMap::default();

    let proj_to_pair: FnvHashMap<u64, (u64, u64)> = chain_pairs
        .keys()
        .copied()
        .map(|x| (cactus_tree.projected_node(x.0), x))
        .collect();

    for (pair, chain_vxs) in chain_pairs.iter() {
        let proj = cactus_tree.projected_node(pair.0);
        for chain in chain_vxs.iter() {
            let net_vxs = cactus_tree.chain_vertices.get(chain).unwrap();
            for n in net_vxs.iter() {
                if n != &proj {
                    let (a, b) = proj_to_pair.get(n).unwrap();
                    net_chain_map
                        .entry(*pair)
                        .or_default()
                        .push((*a, *b, *chain));
                }
            }
        }
    }

    println!("max net vertex: {}", cactus_tree.graph.max_net_vertex);

    println!(" {} chain pairs", chain_pairs.len());

    let mut chain_pair_hierarchy: FnvHashMap<(u64, u64), Vec<(u64, u64)>> =
        FnvHashMap::default();

    let mut chain_pair_ultrabubbles = Vec::new();
    for ((x, y), chain) in chain_pairs.iter() {
        println!("\t{}-{}\t{:?}", x, y, chain);
        let mut ultrabubble = true;
        let mut chain = chain.clone();
        chain.dedup();
        for chain_vx in chain.iter() {
            let vx = *chain_vx as u64;

            let children = net_chain_map.get((x, y)).as_ref();
            let is_ultrabubble = cactus_tree.is_chain_pair_ultrabubble(
                &mut chain_edge_labels,
                // &mut proj_chain_edge_labels,
                children,
                &mut chain_pair_hierarchy,
                *x,
                *y,
                vx,
            );

            if !is_ultrabubble {
                ultrabubble = false
            }
        }

        if ultrabubble {
            println!("{} - {} is ultrabubble", x, y);
            chain_pair_ultrabubbles.push((x, y));
        } else {
            println!("{} - {} is NOT ultrabubble", x, y);
        }
    }
}
