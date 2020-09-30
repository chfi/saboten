use fnv::{FnvHashMap, FnvHashSet};
use std::{collections::HashMap, path::PathBuf};

use bstr::{BString, ByteSlice};

use structopt::StructOpt;

use rs_cactusgraph::{
    biedged_to_cactus,
    biedgedgraph::*,
    cactusgraph::{BiedgedWrapper, BridgeForest, CactusGraph, CactusTree},
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
    let mut projection = Projection::new_for_biedged_graph(&be_graph);

    let orig_graph = be_graph.clone();

    let mut cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);
    cactus_graph.projection.build_inverse();

    let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);
    // cactus_tree.projection.build_inverse();

    let mut bridge_forest = BridgeForest::from_cactus_graph(&cactus_graph);
    bridge_forest.projection.build_inverse();

    // let orig_graph_2 = cactus_tree

    /*
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
    */

    // let mut bridge_edges = Vec::new();

    let chain_edges = cactus_tree.chain_edges();
    /*
    cactus_tree.base_graph().all_edges().for_each(|(a, b, _)| {
        if cactus_tree.is_chain_edge(a, b) {
            println!(" chain edge {} {}", a, b);
            chain_edges.push((a, b));
        }
    });
    */

    /*
    println!("----------");

    orig_graph.graph.all_edges().for_each(|(a, b, w)| {
        if w.black > 0 && cactus_tree.is_bridge_edge(a, b) {
            println!(" bridge edge {} {}", a, b);
            bridge_edges.push((a, b));
        }
    });
    */

    let chain_pairs = cactus_tree.find_chain_pairs();

    for ((a, b), c) in chain_pairs.iter() {
        println!(" -- {}, {} - - {}", a, b, c);
    }

    /*

    println!("bridge edges");
    for (a, b) in bridge_edges.iter() {
        println!("  {}, {}", a, b);
    }

    let mut chain_labels: FnvHashMap<u64, bool> = FnvHashMap::default();

    println!(" --- chain pairs --- ");
    for ((a, b), chain_ix) in chain_pairs.iter() {
        if !chain_labels.contains_key(chain_ix) {
            let net_graph = biedged_to_cactus::build_net_graph(
                &orig_graph,
                &cactus_tree,
                &cactus_graph_projections,
                &cactus_graph_inverse,
                &cycle_map,
                *a,
                *b,
            )
            .unwrap();

            let acyclic = biedged_to_cactus::snarl_is_acyclic(&net_graph, *a);
            let bridgeless =
                biedged_to_cactus::snarl_is_bridgeless(&net_graph, *a, *b);

            chain_labels.insert(**chain_ix, acyclic && bridgeless);

            println!("{}, {} -    acyclic\t{}", a, b, acyclic);
            println!("{}, {} - bridgeless\t{}", a, b, bridgeless);
        }
    }
    */

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
