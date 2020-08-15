use petgraph::prelude::*;
use petgraph::dot::{Dot, Config};

use handlegraph::hashgraph::*;
use handlegraph::handle::*;
use handlegraph::handlegraph::*;
use handlegraph::handle::Direction;

use std::collections::VecDeque;
use std::collections::HashSet;
use std::path::PathBuf;
use std::fs::File;
use std::io::Write;
use std::io;

mod lib;
use lib::gfa_to_biedged_graph;

use handlegraph::mutablehandlegraph::MutableHandleGraph;

use gfa::parser::parse_gfa;

fn main() {
    let path = PathBuf::from("./input/human__pan.AF0__18.gfa");
    let graph = HashGraph::from_gfa(&parse_gfa(&path).unwrap());
}