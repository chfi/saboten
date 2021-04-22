use rustc_hash::FxHashSet;

use crate::biedgedgraph::BiedgedGraph;
use crate::snarls::{Biedged, Node};

#[derive(Clone)]
pub struct NetGraph {
    pub graph: BiedgedGraph<Biedged>,
    pub x: Node,
    pub y: Node,
    pub path: Vec<Node>,
}

#[derive(Debug, PartialEq)]
enum Color {
    Black,
    Gray,
}

impl Color {
    fn toggle(&self) -> Self {
        match self {
            Color::Black => Color::Gray,
            Color::Gray => Color::Black,
        }
    }
}

impl NetGraph {
    pub fn is_acyclic(&self) -> bool {
        let graph = &self.graph.graph;

        let mut visited: FxHashSet<Node> = FxHashSet::default();
        let mut in_path: FxHashSet<Node> = FxHashSet::default();
        let mut stack: Vec<(Color, Node)> = Vec::new();

        let mut acyclic = true;

        let x = self.x;

        let start_color = if graph.edges(x).any(|(_, _, w)| w.black > 0) {
            Color::Gray
        } else {
            Color::Black
        };

        stack.push((start_color, x));

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

                stack.push((last_color.toggle(), current));
                for (_, adj, _) in edges {
                    if in_path.contains(&adj) {
                        acyclic = false;
                    } else {
                        stack.push((last_color.toggle(), adj));
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
            if node != self.x
                && node != self.y
                && self
                    .graph
                    .graph
                    .edges(node)
                    .find(|(_, _, w)| w.black == 1)
                    .is_none()
            {
                return false;
            }
        }
        true
    }

    pub fn is_ultrabubble(&self) -> bool {
        self.is_bridgeless() && self.is_acyclic()
    }
}
