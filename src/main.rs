use std::path::PathBuf;

use structopt::StructOpt;

use rs_cactusgraph::{biedged_to_cactus, biedgedgraph::*};

#[derive(StructOpt, Debug)]
struct Opt {
    in_gfa: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    let mut be_graph = BiedgedGraph::from_gfa_file(&opt.in_gfa)
        .expect("Could not parse provided GFA");

    biedged_to_cactus::contract_all_gray_edges(&mut be_graph);
    biedged_to_cactus::find_3_edge_connected_components(&mut be_graph);
    biedged_to_cactus::contract_loops(&mut be_graph);
}
