use fnv::{FnvHashMap, FnvHashSet};
use petgraph::prelude::*;

use crate::{
    biedged_to_cactus,
    biedgedgraph::{
        end_to_black_edge, opposite_vertex, BiedgedGraph, BiedgedWeight,
    },
    cactusgraph::CactusTree,
    projection::Projection,
};

#[derive(Clone)]
pub struct NetGraph {
    pub graph: BiedgedGraph,
    pub x: u64,
    pub y: u64,
    pub path: Vec<u64>,
}

#[derive(Debug, PartialEq)]
enum Color {
    Black,
    Gray,
}

impl NetGraph {
    pub fn is_acyclic(&self) -> bool {
        let graph = &self.graph.graph;
        let mut visited: FnvHashSet<u64> = FnvHashSet::default();
        let mut in_path: FnvHashSet<u64> = FnvHashSet::default();

        let other_color = |col: &Color| match col {
            Color::Black => Color::Gray,
            Color::Gray => Color::Black,
        };

        let mut stack: Vec<(Color, u64)> = Vec::new();

        let x = self.x;

        if graph.edges(x).find(|(_, _, w)| w.black > 0).is_some() {
            stack.push((Color::Gray, x));
        } else {
            stack.push((Color::Black, x));
        }
        let mut acyclic = true;

        while let Some((last_color, current)) = stack.pop() {
            if !visited.contains(&current) {
                visited.insert(current);
                in_path.insert(current);

                let edges: Vec<_> = graph
                    .edges(current)
                    .filter(|(_, _, w)| match last_color {
                        Color::Black => w.gray > 0,
                        Color::Gray => w.black > 0,
                    })
                    .collect();

                stack.push((other_color(&last_color), current));
                for (_, adj, _) in edges {
                    if in_path.contains(&adj) {
                        acyclic = false;
                    } else {
                        stack.push((other_color(&last_color), adj));
                    }
                }
            } else if in_path.contains(&current) {
                in_path.remove(&current);
            }
        }

        acyclic
    }

    pub fn is_bridgeless(&self) -> bool {
        for node in self.graph.graph.nodes() {
            if node != self.x && node != self.y {
                if self
                    .graph
                    .graph
                    .edges(node)
                    .find(|(_, _, w)| w.black == 1)
                    .is_none()
                {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_ultrabubble(&self) -> bool {
        self.is_bridgeless() & self.is_acyclic()
    }
}
