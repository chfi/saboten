use std::{fs::File, io, io::BufReader, path::PathBuf};

use saboten::{
    biedgedgraph::*,
    cactusgraph,
    cactusgraph::{BridgeForest, CactusGraph, CactusTree},
    ultrabubble::Ultrabubble,
};

use gfa::{gfa::GFA, parser::GFAParser};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

static A3105_PATH: &str = "./test/gfas/A-3105.gfa";

fn graph_transformations(c: &mut Criterion) {
    let parser: GFAParser<usize, ()> = GFAParser::new();
    let gfa: GFA<usize, ()> = parser.parse_file(&A3105_PATH).unwrap();
    c.bench_with_input(
        BenchmarkId::new("graph transformations", "A-3015.gfa"),
        &gfa,
        |b, l| {
            b.iter(|| {
                let orig_graph = BiedgedGraph::from_gfa(&gfa);

                let cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);

                let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);

                let bridge_forest =
                    BridgeForest::from_cactus_graph(&cactus_graph);
            });
        },
    );
}

criterion_group!(
    name = transformations;
    config = Criterion::default();
    targets = graph_transformations);

criterion_main!(transformations);
