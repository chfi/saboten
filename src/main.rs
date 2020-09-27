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

    let mut union_find: UnionFind<usize> =
        UnionFind::new(be_graph.graph.node_count());

    println!(" --- Contracting gray edges --- ");
    biedged_to_cactus::contract_all_gray_edges(&mut be_graph, &mut union_find);

    println!(" --- Finding 3EC-components --- ");
    let components =
        biedged_to_cactus::find_3_edge_connected_components(&be_graph);

    /*
    println!("found {} components", components.len());
    for c in components.iter() {
        println!("{:?}", c);
    }
    */

    println!(" --- Merging 3EC-components --- ");
    biedged_to_cactus::merge_components(
        &mut be_graph,
        components,
        &mut union_find,
    );

    let cactus_graph_projections = union_find.clone();

    println!(" --- Finding cycles --- ");
    let cycles = biedged_to_cactus::find_cycles(&be_graph);

    let mut bridge_forest = be_graph.clone();

    let mut bridge_forest_projections = union_find.clone();

    println!(" --- Constructing bridge forest --- ");
    biedged_to_cactus::contract_simple_cycles(
        &mut bridge_forest,
        &cycles,
        &mut bridge_forest_projections,
    );

    println!(" --- Constructing cactus tree --- ");
    biedged_to_cactus::build_cactus_tree(&mut be_graph, &cycles);

    let mut cactus_graph_inverse: HashMap<usize, Vec<usize>> = HashMap::new();

    let cactus_reps = cactus_graph_projections.into_labeling();
    for (i, k) in cactus_reps.iter().enumerate() {
        cactus_graph_inverse.entry(*k).or_default().push(i);
    }

    println!("Cactus graph vertex projections");
    for (k, v) in cactus_graph_inverse.iter() {
        println!("{} mapped from {:?}", k, v);
    }

    let mut bridge_forest_inverse: HashMap<usize, Vec<usize>> = HashMap::new();

    let bridge_reps = bridge_forest_projections.into_labeling();
    for (i, k) in bridge_reps.iter().enumerate() {
        bridge_forest_inverse.entry(*k).or_default().push(i);
    }

    println!("Bridge forest vertex projections");
    for (k, v) in bridge_forest_inverse.iter() {
        println!("{} mapped from {:?}", k, v);
    }
}
