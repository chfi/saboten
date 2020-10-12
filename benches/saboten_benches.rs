use saboten::{
    biedgedgraph::*,
    cactusgraph,
    cactusgraph::{BridgeForest, CactusGraph, CactusTree},
    ultrabubble::Ultrabubble,
};

use fnv::FnvHashSet;

use std::path::PathBuf;

use gfa::{gfa::GFA, parser::GFAParser};

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};

static GFAPATH: &str = "./test/gfas/";

macro_rules! bench_graph_transforms {
    ($name:ident, $gfa:literal) => {
        fn $name(c: &mut Criterion) {
            let parser: GFAParser<usize, ()> = GFAParser::new();
            let mut path = PathBuf::from(GFAPATH);
            path.push($gfa);
            let gfa: GFA<usize, ()> = parser.parse_file(&path).unwrap();
            c.bench_with_input(
                BenchmarkId::new("graph transformations", $gfa),
                &gfa,
                |b, l| {
                    b.iter(|| {
                        let orig_graph = BiedgedGraph::from_gfa(&gfa);

                        let cactus_graph =
                            CactusGraph::from_biedged_graph(&orig_graph);

                        let _cactus_tree =
                            CactusTree::from_cactus_graph(&cactus_graph);

                        let _bridge_forest =
                            BridgeForest::from_cactus_graph(&cactus_graph);
                    });
                },
            );
        }
    };
}

macro_rules! bench_finding_snarls {
    ($name:ident, $gfa:literal) => {
        fn $name(c: &mut Criterion) {
            let parser: GFAParser<usize, ()> = GFAParser::new();
            let mut path = PathBuf::from(GFAPATH);
            path.push($gfa);
            let gfa: GFA<usize, ()> = parser.parse_file(&path).unwrap();

            let orig_graph = BiedgedGraph::from_gfa(&gfa);

            let cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);

            let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);

            let bridge_forest = BridgeForest::from_cactus_graph(&cactus_graph);
            c.bench_with_input(
                BenchmarkId::new("finding snarls", $gfa),
                &gfa,
                |b, l| {
                    b.iter(|| {
                        cactus_tree.find_chain_pairs();

                        bridge_forest.find_bridge_pairs();
                    });
                },
            );
        }
    };
}

macro_rules! bench_label_chain_edges {
    ($name:ident, $gfa:literal) => {
        fn $name(c: &mut Criterion) {
            let parser: GFAParser<usize, ()> = GFAParser::new();
            let mut path = PathBuf::from(GFAPATH);
            path.push($gfa);
            let gfa: GFA<usize, ()> = parser.parse_file(&path).unwrap();

            let orig_graph = BiedgedGraph::from_gfa(&gfa);
            let cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);
            let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);
            let bridge_forest = BridgeForest::from_cactus_graph(&cactus_graph);
            let chain_pairs = cactus_tree.find_chain_pairs();
            let bridge_pairs = bridge_forest.find_bridge_pairs();

            c.bench_with_input(
                BenchmarkId::new("labeling chain edges", $gfa),
                &gfa,
                |b, l| {
                    b.iter(|| {
                        cactusgraph::chain_pair_ultrabubble_labels(
                            &cactus_tree,
                            &chain_pairs,
                        );
                    });
                },
            );
        }
    };
}

macro_rules! bench_finding_ultrabubbles {
    ($name:ident, $gfa:literal) => {
        fn $name(c: &mut Criterion) {
            let parser: GFAParser<usize, ()> = GFAParser::new();
            let mut path = PathBuf::from(GFAPATH);
            path.push($gfa);
            let gfa: GFA<usize, ()> = parser.parse_file(&path).unwrap();

            let orig_graph = BiedgedGraph::from_gfa(&gfa);
            let cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);
            let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);
            let bridge_forest = BridgeForest::from_cactus_graph(&cactus_graph);

            c.bench_with_input(
                BenchmarkId::new("finding ultrabubbles", $gfa),
                &gfa,
                |b, l| {
                    b.iter(|| {
                        cactusgraph::find_ultrabubbles(
                            &cactus_tree,
                            &bridge_forest,
                        );
                    });
                },
            );
        }
    };
}

macro_rules! bench_build_net_graph {
    ($name:ident, $gfa:literal) => {
        fn $name(c: &mut Criterion) {
            let parser: GFAParser<usize, ()> = GFAParser::new();
            let mut path = PathBuf::from(GFAPATH);
            path.push($gfa);
            let gfa: GFA<usize, ()> = parser.parse_file(&path).unwrap();

            let orig_graph = BiedgedGraph::from_gfa(&gfa);
            let cactus_graph = CactusGraph::from_biedged_graph(&orig_graph);
            let cactus_tree = CactusTree::from_cactus_graph(&cactus_graph);

            let chain_pairs = cactus_tree.find_chain_pairs();
            let chain_edges =
                saboten::cactusgraph::chain_edges(&chain_pairs, &cactus_tree);

            let mut x_ys = chain_edges
                .into_iter()
                .map(|(_, xy)| xy)
                .collect::<Vec<_>>();

            x_ys.sort();

            let chunk_size = x_ys.len() / 4;

            let mut group = c.benchmark_group(&format!("net_graphs/{}", $gfa));
            group.sampling_mode(criterion::SamplingMode::Flat);

            for (i, chunk) in x_ys.chunks(chunk_size).enumerate() {
                let chunk_len = chunk.len() as u64;
                group.throughput(Throughput::Elements(chunk_len));
                group.bench_with_input(
                    BenchmarkId::new(&format!("net graph chunk {}", i), $gfa),
                    &chunk,
                    |b, &chunk| {
                        b.iter(|| {
                            for &(x, y) in chunk.iter() {
                                let net_graph =
                                    cactus_tree.build_net_graph(x, y);
                            }
                        });
                    },
                );
            }

            group.finish();
        }
    };
}

bench_graph_transforms!(transform_a3015, "A-3105.gfa");
bench_graph_transforms!(transform_covid, "relabeledSeqs.nopaths.gfa");

bench_finding_snarls!(find_snarls_a3015, "A-3105.gfa");
bench_finding_snarls!(find_snarls_covid, "relabeledSeqs.nopaths.gfa");

bench_label_chain_edges!(label_chain_edges_a3015, "A-3105.gfa");
bench_label_chain_edges!(label_chain_edges_covid, "relabeledSeqs.nopaths.gfa");

bench_finding_ultrabubbles!(find_ultrabubbles_a3015, "A-3105.gfa");
bench_finding_ultrabubbles!(
    find_ultrabubbles_covid,
    "relabeledSeqs.nopaths.gfa"
);

bench_build_net_graph!(build_net_graphs_a3105, "A-3105.gfa");
bench_build_net_graph!(build_net_graphs_covid, "relabeledSeqs.nopaths.gfa");

/*
criterion_group!(
    name = transformations;
    // config = Criterion::default();
    config = Criterion::default().sample_size(10);
    targets = transform_a3015, transform_covid);

criterion_group!(
    name = snarls;
    // config = Criterion::default();
    config = Criterion::default().sample_size(10);
    targets = find_snarls_a3015, find_snarls_covid);

*/
criterion_group!(
    name = labeling;
    config = Criterion::default().sample_size(10);
    // config = Criterion::default().sample_size(20);
    targets = label_chain_edges_a3015,label_chain_edges_covid);

criterion_group!(
    name = ultrabubbles;
    config = Criterion::default().sample_size(10);
    // config = Criterion::default().sample_size(20);
    targets = find_ultrabubbles_a3015, find_ultrabubbles_covid);

criterion_group!(
    name = net_graphs;
    // config = Criterion::default().sample_size(20).measurement_time(std::time::Duration::from_secs(60));
    config = Criterion::default().sample_size(50);
    // config = Criterion::default().sample_size(20);
    targets = build_net_graphs_a3105, build_net_graphs_covid);

// criterion_main!(transformations, snarls, labeling, ultrabubbles);
// criterion_main!(labeling, ultrabubbles);
criterion_main!(net_graphs);
