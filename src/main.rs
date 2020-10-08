use std::path::PathBuf;

use structopt::StructOpt;

use saboten::{
    biedgedgraph::*,
    cactusgraph,
    cactusgraph::{BridgeForest, CactusGraph, CactusTree},
    ultrabubble::Ultrabubble,
};

use gfa::{gfa::GFA, parser::GFAParser};

#[derive(StructOpt, Debug)]
struct Opt {
    /// Path to input GFA file
    gfa: PathBuf,
    /// Output JSON
    #[structopt(short, long)]
    json: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    let parser: GFAParser<usize, ()> = GFAParser::new();
    let gfa: GFA<usize, ()> = parser.parse_file(&opt.gfa)?;

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
            .map(|(&(x, y), _)| Ultrabubble { start: x, end: y })
            .collect::<Vec<_>>();

        let json = serde_json::to_string(&ultrabubble_vec)?;
        println!("{}", json);
    } else {
        for ((x, y), _) in ultrabubbles.iter() {
            println!("{}\t{}", x, y);
        }
    }

    Ok(())
}
