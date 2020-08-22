use handlegraph::hashgraph::*;
use std::path::PathBuf;
use gfa::parser::parse_gfa;

fn main() {
    let path = PathBuf::from("./input/human__pan.AF0__18.gfa");
    let _graph = HashGraph::from_gfa(&parse_gfa(&path).unwrap());
}