use rustc_hash::FxHashSet;

use crate::biedgedgraph::BiedgedGraph;
use crate::cactusgraph::{CactusTree, ChainEdges};
use crate::snarls::{Biedged, Node, Snarl, SnarlMap};

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
    pub fn contained_snarls(
        &self,
        cactus_tree: &CactusTree,
        snarl_map: &SnarlMap,
    ) -> FxHashSet<Snarl<()>> {
        let mut res: FxHashSet<Snarl<()>> = FxHashSet::default();

        for node in self.graph.graph.nodes() {
            if node != self.x && node != self.y {
                if let Some(lefts) = snarl_map.lefts.get(&node) {
                    for &left_snarl_ix in lefts.iter() {
                        let snarl =
                            snarl_map.snarls.get(&left_snarl_ix).unwrap();

                        if self.graph.graph.contains_node(snarl.right()) {
                            res.insert(*snarl);
                        }
                    }
                }

                if let Some(rights) = snarl_map.rights.get(&node) {
                    for &right_snarl_ix in rights.iter() {
                        let snarl =
                            snarl_map.snarls.get(&right_snarl_ix).unwrap();

                        if self.graph.graph.contains_node(snarl.left()) {
                            res.insert(*snarl);
                        }
                    }
                }
            }
        }

        res
    }

    /*
    pub fn chain_pair_dfs(
        &self,
        cactus_tree: &CactusTree,
        chain_edges: &ChainEdges,
    ) -> Vec<Node> {
        let (net, chain) = chain_edges
            .biedged_to_chain(&Snarl::<()>::chain_pair(self.x, self.y))
            .unwrap();

        let mut visited: FxHashSet<Node> = FxHashSet::default();
        visited.insert(net);

        let neighbors = cactus_tree.graph.graph.neighbors(net);

        #[derive(Clone, Copy, PartialEq, Eq)]
        struct Step {
            parent: Node,
            step: Node,
        }

        let mut stack: Vec<Step> = Vec::new();

        let mut log: Vec<Step> = Vec::new();

        let mut reachable_chain_pairs: FxHashSet<Snarl<()>> =
            FxHashSet::default();

        for other in neighbors {
            if other != chain && cactus_tree.graph.is_chain_vertex(other) {
                stack.push(Step {
                    parent: net,
                    step: other,
                });
                // stack.push(Step::Chain { chain: other });
            }
        }

        for step in stack.pop() {
            log.push(step);

            visited.insert(step.step);

            /*
            if let Some(chain) = prev_chain {
                if let Some(chain_pairs) =
                    chain_edges.chain_to_biedged(net, chain)
                {
                    reachable_chain_pairs.extend(
                        chain_pairs
                            .iter()
                            .map(|&(x, y)| Snarl::chain_pair(x, y)),
                    );
                }
            }
            */

            let on_net = cactus_tree.graph.is_net_vertex(step.step);
            let neighbors = cactus_tree.graph.graph.neighbors(step.step);

            for other in neighbors {
                if !visited.contains(&other) {
                    if on_net {
                        if cactus_tree.graph.is_chain_vertex(other) {
                            stack.push(Step::Chain { chain: other });
                        }
                    } else {
                        if cactus_tree.graph.is_net_vertex(other) {
                            stack.push(Step::Net {
                                chain: cur_node,
                                net: other,
                            });
                        }
                    }
                }
            }
        }

        reachable_chain_pairs
    }
    */

    /*
    pub fn contained_chain_pairs(
        &self,
        cactus_tree: &crate::cactusgraph::CactusTree,
        chain_edges: &crate::cactusgraph::ChainEdges,
    ) -> FxHashSet<Snarl<()>> {


        #[derive(Clone, Copy)]
        enum Step {
            Net { chain: Node, net: Node },
            Chain { chain: Node },
        }

        if self.path.len() == 1 {
            // chain pair
            let (net, chain) = chain_edges
                .biedged_to_chain(&Snarl::<()>::chain_pair(self.x, self.y))
                .unwrap();

            let mut visited: FxHashSet<Node> = FxHashSet::default();
            visited.insert(net);

            let neighbors = cactus_tree.graph.graph.neighbors(net);

            let mut stack: Vec<Step> = Vec::new();

            let mut reachable_chain_pairs: FxHashSet<Snarl<()>> =
                FxHashSet::default();

            for other in neighbors {
                if other != chain && cactus_tree.graph.is_chain_vertex(other) {
                    stack.push(Step::Chain { chain: other });
                }
            }

            for step in stack.pop() {
                let (prev_chain, cur_node, on_net) = match step {
                    Step::Net { chain, net } => (Some(chain), net, true),
                    Step::Chain { chain } => (None, chain, false),
                };

                visited.insert(cur_node);

                if let Some(chain) = prev_chain {
                    if let Some(chain_pairs) =
                        chain_edges.chain_to_biedged(net, chain)
                    {
                        reachable_chain_pairs.extend(
                            chain_pairs
                                .iter()
                                .map(|&(x, y)| Snarl::chain_pair(x, y)),
                        );
                    }
                }

                let neighbors = cactus_tree.graph.graph.neighbors(cur_node);

                for other in neighbors {
                    if !visited.contains(&other) {
                        if on_net {
                            if cactus_tree.graph.is_chain_vertex(other) {
                                stack.push(Step::Chain { chain: other });
                            }
                        } else {
                            if cactus_tree.graph.is_net_vertex(other) {
                                stack.push(Step::Net {
                                    chain: cur_node,
                                    net: other,
                                });
                            }
                        }
                    }
                }
            }

            reachable_chain_pairs
        } else {
            // bridge pair

            let mut reachable_chain_pairs: FxHashSet<Snarl<()>> =
                FxHashSet::default();

            let mut visited: FxHashSet<Node> = FxHashSet::default();

            let mut parent: FxHashMap<Node, Node> = FxHashMap::default();

            let mut path_set: FxHashSet<Node> = self.path.iter().copied().collect();

            // avoid the snarl bridge edges too
            path_set.insert(self.x.opposite());
            path_set.insert(self.y.opposite());

            let mut stack: Vec<Step> = Vec::new();


            for &path_node in self.path.iter() {
                visited.insert(path_node);

                let neighbors = cactus_tree.graph.graph.neighbors(path_node).filter(|other| {
                    !path_set.contains(other) && !visited.contains(other)
                });

                if cactus_tree.graph.is_net_vertex(path_node) {


                    for other in neighbors {
                        if cactus_tree.graph
                    }
                    /*
                    stack.push(Step::Chain { chain: other });
                    stack.push(Step::Net {
                        chain: cur_node,
                        net: other,
                    });
                    */

                    // stack.push(value)
                } else {
                }

            }

            // visited.insert(net);

            // let neighbors = cactus_tree.graph.graph.neighbors(net);

            unimplemented!();
        }
    }
    */

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
