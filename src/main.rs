use fnv::FnvHashMap;
use std::path::PathBuf;

use bstr::BString;

use structopt::StructOpt;

use rs_cactusgraph::{
    biedgedgraph::*,
    cactusgraph,
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

    let bridge_forest = BridgeForest::from_cactus_graph(&cactus_graph);

    let ultrabubbles =
        cactusgraph::find_ultrabubbles(&cactus_tree, &bridge_forest);

    println!("x\ty\tnet\tchain/edges\tcontains");
    println!();
    for ((x, y), contained) in ultrabubbles.iter() {
        let net = cactus_tree.projected_node(*x);

        // if x's black edge maps to a chain vertex, this is a chain pair
        if let Some(chain) = cactus_tree.black_edge_chain_vertex(*x) {
            print!("{}\t{}\t{}\t\t{}", x, y, net, chain);
        } else {
            let x_black_edge = end_to_black_edge(*x);
            let y_black_edge = end_to_black_edge(*y);

            print!(
                "{}\t{}\t{}\t{:?}\t{:?}",
                x, y, net, x_black_edge, y_black_edge
            );
        }

        if !contained.is_empty() {
            print!("\t{:?}", contained);
        }
        println!();
    }
}
