use std::path::PathBuf;

use structopt::StructOpt;

use saboten::{
    biedgedgraph::*,
    cactusgraph,
    cactusgraph::{BridgeForest, CactusGraph, CactusTree},
    ultrabubble::Ultrabubble,
};

use gfa::{
    gfa::GFA,
    parser::{GFAParser, GFAParserBuilder},
};

use log::info;

#[derive(StructOpt, Debug)]
struct Opt {
    /// Path to input GFA file
    gfa: PathBuf,
    /// Output JSON
    #[structopt(short, long)]
    json: bool,
    /// The number of threads to use. If omitted, Rayon's default will
    /// be used, based on the RAYON_NUM_THREADS environment variable,
    /// or the number of logical CPUs.
    #[structopt(short, long)]
    threads: Option<usize>,
    /// Show no messages.
    #[structopt(long)]
    quiet: bool,
    /// Show info messages.
    #[structopt(long)]
    info: bool,
    /// Show debug messages.
    #[structopt(long)]
    debug: bool,
}

fn init_logger(opt: &Opt) {
    let mut builder = pretty_env_logger::formatted_builder();
    if !opt.quiet {
        let mut log_level = log::LevelFilter::Error;
        if opt.info {
            log_level = log::LevelFilter::Info;
        }
        if opt.debug {
            log_level = log::LevelFilter::Debug;
        }
        builder.filter_level(log_level);
    }

    builder.init();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    init_logger(&opt);

    if let Some(threads) = &opt.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(*threads)
            .build_global()?;
    }

    info!("Parsing GFA");
    let mut parser_builder = GFAParserBuilder::all();
    parser_builder.paths = false;
    parser_builder.containments = false;
    let parser: GFAParser<usize, ()> = parser_builder.build();
    let gfa: GFA<usize, ()> = parser.parse_file(&opt.gfa)?;

    info!("Building biedged graph");
    let be_graph = BiedgedGraph::from_gfa(&gfa);

    let orig_graph = be_graph.clone();

    info!("Building cactus graph");
    let cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);

    info!("Building cactus tree");
    let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);

    info!("Building bridge forest");
    let bridge_forest = BridgeForest::from_cactus_graph(&cactus_graph);

    info!("Finding ultrabubbles");
    let ultrabubbles =
        cactusgraph::find_ultrabubbles(&cactus_tree, &bridge_forest);

    let ultrabubbles = cactusgraph::inverse_map_ultrabubbles(ultrabubbles);

    info!("Found {} ultrabubbles", ultrabubbles.len());

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
