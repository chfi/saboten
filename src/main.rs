use std::{
    cmp,
    io::{stdout, Write},
    path::PathBuf,
};

use bstr::BString;

use structopt::StructOpt;

use rs_cactusgraph::{biedged_to_cactus, biedgedgraph::*};

use gfa::{
    gfa::{Header, Link, Orientation, Segment, GFA},
    gfa_name_conversion::NameMap,
    parser::GFAParser,
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
        header: Header {
            version: Some("1.0".into()),
            optional: (),
        },
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
    let opt = Opt::from_args();

    let mut be_graph = {
        if let Some(gfa) = BiedgedGraph::from_gfa_file(&opt.in_gfa) {
            gfa
        } else {
            let parser = GFAParser::new();
            let bstr_gfa: GFA<BString, ()> =
                parser.parse_file(&opt.in_gfa).unwrap();
            let name_map = NameMap::build_from_gfa(&bstr_gfa);
            let usize_gfa =
                name_map.gfa_bstring_to_usize(&bstr_gfa, false).unwrap();
            BiedgedGraph::from_gfa(&usize_gfa)
        }
    };

    biedged_to_cactus::contract_all_gray_edges(&mut be_graph);
    let components =
        biedged_to_cactus::find_3_edge_connected_components(&be_graph);
    biedged_to_cactus::merge_components(&mut be_graph, components);
    // biedged_to_cactus::contract_loops(&mut be_graph);

    // let mut s = stdout();
    // be_graph.output_dot(&mut s).unwrap();

    let mut un_gfa = be_graph.to_gfa_bstring();
    let mut un_gfa_str = String::new();
    write_gfa(&un_gfa, &mut un_gfa_str);
    println!("{}", un_gfa_str);

    // let mut out_file = std::fs::File::create("./test.gfa").unwrap();
    // let _ = writeln!(out_file, "{}", un_gfa_str);

    /*
    let (usize_gfa, name_map) = paper_gfa_with_map();

    let mut be_graph = BiedgedGraph::from_gfa(&usize_gfa);

    biedged_to_cactus::contract_all_gray_edges(&mut be_graph);

    let components =
        biedged_to_cactus::find_3_edge_connected_components(&be_graph);

    biedged_to_cactus::merge_components(&mut be_graph, components);

    println!("\nafter mutations");
    println!("  num nodes: {}", be_graph.graph.node_count());
    println!("  num black edges: {}", be_graph.black_edge_count());
    println!("  num gray edges: {}", be_graph.gray_edge_count());

    let mut un_gfa = be_graph.to_gfa_bstring();

    let mut un_gfa_str = String::new();
    write_gfa(&un_gfa, &mut un_gfa_str);

    let mut out_file = std::fs::File::create("./test.gfa").unwrap();

    let _ = writeln!(out_file, "{}", un_gfa_str);

    let edges_after = be_graph.sorted_edges();

    for (a, b, bl, gr) in edges_after {
        println!("{} - {}; {} {}", a, b, bl, gr);
    }
    */
}
