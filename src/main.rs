use std::{io::stdout, path::PathBuf};

use bstr::BString;

use structopt::StructOpt;

use rs_cactusgraph::{biedged_to_cactus, biedgedgraph::*};

use gfa::{
    gfa::{Header, Link, Orientation, Segment, GFA},
    gfa_name_conversion::NameMap,
    writer::{gfa_string, write_gfa},
};

#[derive(StructOpt, Debug)]
struct Opt {
    in_gfa: PathBuf,
}

fn paper_gfa() -> GFA<BString, ()> {
    let segments: Vec<Segment<BString, ()>> = vec![
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
        "o", "p", "q", "r",
    ]
    .into_iter()
    .map(|n| Segment {
        name: BString::from(n),
        sequence: BString::from("*"),
        optional: (),
    })
    .collect();

    let links: Vec<Link<BString, ()>> = vec![
        ("a", "b"),
        ("a", "c"),
        ("b", "d"),
        ("c", "d"),
        ("d", "e"),
        ("d", "f"),
        ("e", "g"),
        ("f", "g"),
        ("f", "h"),
        ("g", "k"),
        ("g", "l"),
        ("h", "i"),
        ("h", "j"),
        ("i", "j"),
        ("j", "l"),
        ("k", "l"),
        ("l", "m"),
        ("m", "n"),
        ("m", "o"),
        ("n", "p"),
        ("o", "p"),
        ("p", "m"),
        ("p", "q"),
        ("p", "r"),
    ]
    .into_iter()
    .map(|(f, t)| Link {
        from_segment: BString::from(f),
        from_orient: Orientation::Forward,
        to_segment: BString::from(t),
        to_orient: Orientation::Forward,
        overlap: BString::from("0M"),
        optional: (),
    })
    .collect();

    GFA {
        header: Default::default(),
        segments,
        links,
        containments: vec![],
        paths: vec![],
    }
}

fn paper_gfa_with_map() -> (GFA<usize, ()>, NameMap) {
    let bstr_gfa = paper_gfa();
    let name_map = NameMap::build_from_gfa(&bstr_gfa);
    let usize_gfa = name_map.gfa_bstring_to_usize(&bstr_gfa, false).unwrap();
    (usize_gfa, name_map)
}

fn main() {
    let (usize_gfa, name_map) = paper_gfa_with_map();

    println!("{} segments", usize_gfa.segments.len());
    println!("{} links", usize_gfa.links.len());

    let mut gfa_str = String::new();
    write_gfa(&usize_gfa, &mut gfa_str);
    println!("{}", gfa_str);

    let mut be_graph = BiedgedGraph::from_gfa(&usize_gfa);

    biedged_to_cactus::contract_all_gray_edges(&mut be_graph);
    biedged_to_cactus::find_3_edge_connected_components(&mut be_graph);
    biedged_to_cactus::contract_loops(&mut be_graph);

    println!("-----------------------------");
    let mut un_gfa = be_graph.to_gfa();

    let mut un_gfa_str = String::new();
    write_gfa(&un_gfa, &mut un_gfa_str);
    println!("{}", un_gfa_str);

    println!("black edge count: {}", be_graph.black_edge_count());
    println!("gray edge count: {}", be_graph.gray_edge_count());

    /*
    let opt = Opt::from_args();
    let mut be_graph = BiedgedGraph::from_gfa_file(&opt.in_gfa)
        .expect("Could not parse provided GFA");

    biedged_to_cactus::contract_all_gray_edges(&mut be_graph);
    biedged_to_cactus::find_3_edge_connected_components(&mut be_graph);
    biedged_to_cactus::contract_loops(&mut be_graph);

    let mut s = stdout();
    be_graph.output_dot(&mut s).unwrap();
    */
}
