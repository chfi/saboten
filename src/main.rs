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

    let orig_graph = be_graph.clone();

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

    let mut cycle_map: HashMap<u64, Vec<usize>> = HashMap::new();

    for (i, cycle) in cycles.iter().enumerate() {
        let mut iter = cycle.iter().skip(1);
        for vx in iter {
            cycle_map.entry(*vx).or_default().push(i);
        }
        println!("{}\t{:?}", i, cycle);
    }

    let mut bridge_forest = be_graph.clone();

    let mut bridge_forest_projections = union_find.clone();

    println!(" --- Constructing bridge forest --- ");
    biedged_to_cactus::contract_simple_cycles(
        &mut bridge_forest,
        &cycles,
        &mut bridge_forest_projections,
    );

    println!(" --- Constructing cactus tree --- ");

    let mut cactus_tree = be_graph.clone();

    let chain_vertices =
        biedged_to_cactus::build_cactus_tree(&mut cactus_tree, &cycles);

    let mut cactus_graph_inverse: HashMap<usize, Vec<usize>> = HashMap::new();

    let cactus_reps = cactus_graph_projections.clone().into_labeling();
    for (i, k) in cactus_reps.iter().enumerate() {
        cactus_graph_inverse.entry(*k).or_default().push(i);
    }

    println!();
    println!("Cactus graph vertex projections");
    for (k, v) in cactus_graph_inverse.iter() {
        println!("{} mapped from {:?}", k, v);
    }

    let mut bridge_forest_inverse: HashMap<usize, Vec<usize>> = HashMap::new();

    let bridge_reps = bridge_forest_projections.into_labeling();
    for (i, k) in bridge_reps.iter().enumerate() {
        bridge_forest_inverse.entry(*k).or_default().push(i);
    }

    println!();
    println!("Bridge forest vertex projections");
    for (k, v) in bridge_forest_inverse.iter() {
        println!("{} mapped from {:?}", k, v);
    }

    let name_map = gfa_name_map.unwrap();
    for seg in gfa.segments.iter() {
        let id = seg.name as u64;
        let name = name_map.inverse_map_name(seg.name).unwrap();
        let name = name.to_str().unwrap();
        let (a, b) = id_to_black_edge(id);
        let (x, y) = projected_edge(&cactus_graph_projections, a, b);
        let cyc_x = cycle_map.get(&x);
        let cyc_y = cycle_map.get(&y);
        println!(
            "{} projects to edge ({}, {}) = {},\tcycles {:?}, {:?}",
            name,
            x,
            y,
            be_graph.graph.contains_edge(x, y),
            cyc_x,
            cyc_y
        );
    }

    let chain_pairs = vec![
        (24, 30),
        (25, 30),
        (24, 25),
        (24, 31),
        (25, 31),
        (30, 31),
        (14, 15),
        (14, 18),
        (14, 19),
        (15, 18),
        (15, 19),
        (18, 19),
    ];

    for (a, b) in chain_pairs.iter() {
        println!(
            "{}, {} is chain pair:\t{}",
            a,
            b,
            biedged_to_cactus::is_chain_pair(
                &cactus_graph_projections,
                &cycle_map,
                *a,
                *b
            )
        );
    }

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
