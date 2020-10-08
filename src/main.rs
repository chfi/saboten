use std::path::PathBuf;

use structopt::StructOpt;

use rs_cactusgraph::{
    biedgedgraph::*,
    cactusgraph,
    cactusgraph::{BridgeForest, CactusGraph, CactusTree},
    ultrabubble::ChainPair,
};

use gfa::{gfa::GFA, parser::GFAParser};

#[derive(StructOpt, Debug)]
struct Opt {
    in_gfa: PathBuf,
    /// Output JSON
    #[structopt(short, long)]
    json: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    let parser: GFAParser<usize, ()> = GFAParser::new();
    let gfa: GFA<usize, ()> = parser.parse_file(&opt.in_gfa)?;

    let be_graph = BiedgedGraph::from_gfa(&gfa);

    let orig_graph = be_graph.clone();

    let cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);

    let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);

    let bridge_forest = BridgeForest::from_cactus_graph(&cactus_graph);

    let ultrabubbles =
        cactusgraph::find_ultrabubbles_par(&cactus_tree, &bridge_forest);

    let ultrabubbles = cactusgraph::inverse_map_ultrabubbles(ultrabubbles);

    if opt.json {
        let ultrabubble_vec = ultrabubbles
            .iter()
            .map(|(&(x, y), _)| ChainPair { x, y })
            .collect::<Vec<_>>();

        let json = serde_json::to_string(&ultrabubble_vec)?;
        println!("{}", json);
    } else {
        println!("x\ty");
        for ((x, y), _) in ultrabubbles.iter() {
            println!("{}\t{}", x, y);
        }
    }

    Ok(())
}
