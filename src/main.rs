use fnv::FnvHashMap;
use std::path::PathBuf;

use bstr::BString;

use structopt::StructOpt;

use rs_cactusgraph::{
    biedgedgraph::*,
    cactusgraph::{BiedgedWrapper, BridgeForest, CactusGraph, CactusTree},
    netgraph::NetGraph,
};

use gfa::{
    gfa::{name_conversion::NameMap, GFA},
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

    let be_graph = BiedgedGraph::from_gfa(&gfa);
    // let mut be_graph = example_graph_2();

    let orig_graph = be_graph.clone();

    let cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);

    let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);

    let chain_pairs = cactus_tree.find_chain_pairs();

    let chain_net_graphs: Vec<((u64, u64), NetGraph)> = chain_pairs
        .iter()
        .map(|&(a, b)| {
            let net_graph = cactus_tree.build_net_graph(a, b).unwrap();
            ((a, b), net_graph)
        })
        .collect();

    let mut chain_edge_labels: FnvHashMap<(u64, u64), bool> =
        FnvHashMap::default();

    for (chain_pair, net_graph) in chain_net_graphs.iter() {
        let result = net_graph.is_ultrabubble();

        let a = chain_pair.0.min(chain_pair.1);
        let c_x = cactus_tree.black_edge_chain_vertex(a).unwrap();

        let p_a = cactus_tree.projected_node(a);

        let key = (p_a, c_x);
        chain_edge_labels.insert(key, result);
    }

    let mut chain_pair_ultrabubbles = Vec::new();
    for &(x, y) in chain_pairs.iter() {
        let c_x = cactus_tree.black_edge_chain_vertex(x).unwrap();

        let contained_chain_pairs = cactus_tree.is_chain_pair_ultrabubble(
            &mut chain_edge_labels,
            x,
            y,
            c_x,
        );

        if let Some(children) = contained_chain_pairs {
            chain_pair_ultrabubbles.push((x, y, children));
        }
    }

    let bridge_forest = BridgeForest::from_cactus_graph(&cactus_graph);

    let bridge_pairs = bridge_forest.find_bridge_pairs();

    let bridge_net_graphs: Vec<((u64, u64), NetGraph)> = bridge_pairs
        .iter()
        .map(|&(a, b)| {
            let net_graph = cactus_tree.build_net_graph(a, b).unwrap();
            ((a, b), net_graph)
        })
        .collect();

    let mut bridge_pair_labels: FnvHashMap<(u64, u64), bool> =
        FnvHashMap::default();

    println!(" --- Chain Pairs  --- ");
    println!("x\ty\tnet\tchain\tcontains");
    println!();
    for (x, y, contained) in chain_pair_ultrabubbles.iter() {
        let chain = cactus_tree.black_edge_chain_vertex(*x).unwrap();
        let net = cactus_tree.projected_node(*x);
        print!("{}\t{}\t{}\t{}", x, y, net, chain);
        if !contained.is_empty() {
            print!("\t{:?}", contained);
        }
        println!();
    }

    let mut bridge_pair_ultrabubbles = Vec::new();

    println!();
    for (bridge_pair, net_graph) in bridge_net_graphs.iter() {
        let result = net_graph.is_ultrabubble();
        bridge_pair_labels.insert(*bridge_pair, result);
        if result {
            let x = bridge_pair.0;
            let y = bridge_pair.1;

            let contained_chain_pairs = cactus_tree.is_bridge_pair_ultrabubble(
                &chain_edge_labels,
                x,
                y,
                &net_graph.path,
            );

            if let Some(children) = contained_chain_pairs {
                bridge_pair_ultrabubbles.push((x, y, children));
            }
        }
    }

    println!(" --- Bridge Pairs --- ");
    println!("x\ty\tnet\tx edge\ty edge\tcontains");
    println!();
    for (x, y, contained) in bridge_pair_ultrabubbles.iter() {
        let x_black_edge = end_to_black_edge(*x);
        let y_black_edge = end_to_black_edge(*y);
        let net = cactus_tree.projected_node(*x);
        print!(
            "{}\t{}\t{}\t{:?}\t{:?}",
            x, y, net, x_black_edge, y_black_edge
        );
        if !contained.is_empty() {
            print!("\t{:?}", contained);
        }
        println!();
    }
}
