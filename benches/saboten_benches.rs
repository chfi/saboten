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

                let _cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);

                let _bridge_forest =
                    BridgeForest::from_cactus_graph(&cactus_graph);
            });
        },
    );
}

fn finding_snarls(c: &mut Criterion) {
    let parser: GFAParser<usize, ()> = GFAParser::new();
    let gfa: GFA<usize, ()> = parser.parse_file(&A3105_PATH).unwrap();

    let orig_graph = BiedgedGraph::from_gfa(&gfa);

    let cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);

    let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);

    let bridge_forest = BridgeForest::from_cactus_graph(&cactus_graph);
    c.bench_with_input(
        BenchmarkId::new("finding snarls", "A-3015.gfa"),
        &gfa,
        |b, l| {
            b.iter(|| {
                cactus_tree.find_chain_pairs();

                bridge_forest.find_bridge_pairs();
            });
        },
    );
}
                    &cactus_tree,
                    &bridge_forest,
                );
            });
        },
    );
}

fn finding_ultrabubbles(c: &mut Criterion) {
    let parser: GFAParser<usize, ()> = GFAParser::new();
    let gfa: GFA<usize, ()> = parser.parse_file(&A3105_PATH).unwrap();

    let orig_graph = BiedgedGraph::from_gfa(&gfa);

    let cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);

    let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);

    let bridge_forest = BridgeForest::from_cactus_graph(&cactus_graph);
    c.bench_with_input(
        BenchmarkId::new("finding ultrabubbles", "A-3015.gfa"),
        &gfa,
        |b, l| {
            b.iter(|| {
                cactusgraph::find_ultrabubbles(&cactus_tree, &bridge_forest);
            });
        },
    );
}

criterion_group!(
    name = transformations;
    config = Criterion::default();
    targets = graph_transformations);

criterion_group!(
    name = snarls;
    config = Criterion::default();
    targets = finding_snarls);

criterion_group!(
    name = ultrabubbles;
    config = Criterion::default().sample_size(20);
    targets = finding_ultrabubbles);

criterion_main!(transformations, snarls, ultrabubbles);
