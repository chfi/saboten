use std::{collections::BTreeMap, path::PathBuf};

use bstr::{BString, ByteSlice};

use structopt::StructOpt;

use rs_cactusgraph::{biedged_to_cactus, biedgedgraph::*};

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

        for seg in bstr_gfa.segments.iter() {
            let name = seg.name.to_str().unwrap();
            let id = name_map.map_name(name).unwrap();

            let (l, r) = id_to_black_edge(id as u64);

            println!("{} -> ({},{})", name, l, r);
        }

        gfa_name_map = Some(name_map);

        usize_gfa
    };

    let mut be_graph = BiedgedGraph::from_gfa(&gfa);

    let inv_map = |name_map: &NameMap, s: usize| {
        projected_node_name(name_map, s as u64)
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned()
    };

    let project_gfa_name = move |proj_map: &BTreeMap<u64, u64>,
                                 seg_id: usize| {
        let (left, right) = project_graph_id(proj_map, seg_id);
        if let Some(ref name_map) = gfa_name_map {
            (
                inv_map(name_map, seg_id),
                inv_map(name_map, left),
                inv_map(name_map, right),
            )
        } else {
            (
                projected_node_id(seg_id as u64),
                projected_node_id(left as u64),
                projected_node_id(right as u64),
            )
        }
    };

    let mut projection_map = BTreeMap::new();

    biedged_to_cactus::contract_all_gray_edges(
        &mut be_graph,
        &mut projection_map,
    );

    let components =
        biedged_to_cactus::find_3_edge_connected_components(&be_graph);

    biedged_to_cactus::merge_components(
        &mut be_graph,
        components,
        &mut projection_map,
    );

    let cycles = biedged_to_cactus::find_cycles(&be_graph);

    /*
    println!("nodes -- ");
    for node in be_graph.graph.nodes() {
        println!("{}", node);
    }
    println!("edges -- ");

    for (from, to, w) in be_graph.graph.all_edges() {
        println!("{}\t{}\t{}", from, to, w.black);
    }

    println!("      -- ");
    */

    biedged_to_cactus::contract_simple_cycles(
        &mut be_graph,
        &cycles,
        &mut projection_map,
    );

    for seg in gfa.segments.iter() {
        let (seg, left, right) = project_gfa_name(&projection_map, seg.name);
        println!("{}\t{}\t{}", seg, left, right);
    }
}
