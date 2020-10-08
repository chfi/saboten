use std::path::PathBuf;

use structopt::StructOpt;

use rs_cactusgraph::{
    biedgedgraph::*,
    cactusgraph,
    cactusgraph::{BiedgedWrapper, BridgeForest, CactusGraph, CactusTree},
    ultrabubble::ChainPair,
};

use gfa::{gfa::GFA, parser::GFAParser};

#[derive(StructOpt, Debug)]
struct Opt {
    in_gfa: PathBuf,
    /// Output JSON
    #[structopt(short)]
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
        println!("x\ty\tnet\tchain/edges\tcontains");
        println!();

        for ((x, y), _) in ultrabubbles.iter() {
            let net = cactus_tree.projected_node(*x);

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
        }
    }

    Ok(())
}
