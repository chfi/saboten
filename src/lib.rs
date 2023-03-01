use roaring::RoaringBitmap;
use sprs::{CsMat, CsMatBase, CsVec};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::Arc,
};
use waragraph_core::graph::{
    matrix::MatGraph,
    spoke::{
        hyper::{Cycle, HyperSpokeGraph, VertexId},
        HubId, SpokeGraph,
    },
    PathIndex,
};
use waragraph_core::graph::{Edge, Node, OrientedNode};

#[derive(Debug, Clone)]
pub struct Saboten {
    cactus_tree: CactusTree,
    vg_adj: CsMat<u8>,
    // chain_edges: Vec<()>,
    // bridge_edges: Vec<()>,
    ultrabubbles: Vec<(OrientedNode, OrientedNode)>,

    // ultrabubble index -> [ultrabubble index]
    contained_ultrabubbles: BTreeMap<usize, Vec<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Snarl {
    from: OrientedNode,
    to: OrientedNode,

    is_acyclic_and_bridgeless: bool,
}

impl Saboten {
    pub fn from_edges(
        node_count: usize,
        edges: impl IntoIterator<Item = Edge>,
    ) -> Self {
        let edges = edges.into_iter().collect::<Vec<_>>();
        let vg_adj = PathIndex::directed_adjacency_matrix(
            node_count,
            edges.iter().copied(),
        );

        let cactus_tree = CactusTree::from_edges(node_count, edges);

        let chain_pairs = cactus_tree.enumerate_chain_pairs();

        // let mut chain_edge_labels: HashMap<usize, bool> = HashMap::new();

        // (ChainVx, NetVx) -> bool
        let mut chain_edge_labels: HashMap<(usize, usize), bool> =
            HashMap::new();

        let mut chain_edge_pairs: HashMap<
            (usize, usize),
            (OrientedNode, OrientedNode),
        > = HashMap::new();

        for &((a, b), chain_ix) in chain_pairs.iter() {
            let chain_edge = if let CacTreeEdge::Chain { net, cycle, .. } =
                cactus_tree.graph.edge[cactus_tree.net_edges + chain_ix]
            {
                (net.ix(), cactus_tree.net_vertices + cycle)
            } else {
                unreachable!();
            };

            if let Some(net_graph) =
                cactus_tree.chain_edge_net_graph(&vg_adj, (a, b), chain_ix)
            {
                // the net graph method returns None if the graph has a bridge
                let is_acyclic = net_graph_is_acyclic(a, &net_graph);
                chain_edge_labels.insert(chain_edge, is_acyclic);
            } else {
                chain_edge_labels.insert(chain_edge, false);
            }

            chain_edge_pairs.insert(chain_edge, (a, b));
        }

        // DFS from each chain edge to check the contained chain pairs
        let mut ultrabubbles: Vec<(OrientedNode, OrientedNode)> = Vec::new();

        let mut ultrabubble_map: HashMap<(usize, usize), usize> =
            HashMap::new();

        // map from ultrabubble index to chain edges
        let mut contained: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();

        for (&chain_edge, &is_valid) in chain_edge_labels.iter() {
            let (net, chain) = chain_edge;

            let mut all_valid = is_valid;

            let mut inner = Vec::new();

            cactus_tree.chain_pair_dfs((chain, net), |from, to| {
                if let Some(contained_valid) =
                    chain_edge_labels.get(&(from, to))
                {
                    if *contained_valid {
                        inner.push((from, to));
                    }
                    all_valid &= contained_valid
                }
            });

            if all_valid {
                let snarl = chain_edge_pairs.get(&chain_edge).unwrap();

                let i = ultrabubbles.len();
                contained.insert(i, inner);
                ultrabubble_map.insert(chain_edge, i);

                ultrabubbles.push(*snarl);
            }
        }

        let mut contained_ultrabubbles: BTreeMap<usize, Vec<usize>> =
            BTreeMap::new();

        for (ub_i, _ultrabubble) in ultrabubbles.iter().enumerate() {
            let entry = contained_ultrabubbles.entry(ub_i).or_default();

            if let Some(contained) = contained.get(&ub_i) {
                contained
                    .iter()
                    .filter_map(|inner_chain_edge| {
                        ultrabubble_map.get(inner_chain_edge)
                    })
                    .for_each(|ub_j| {
                        entry.push(*ub_j);
                    })
            }
        }

        Self {
            cactus_tree,
            vg_adj,
            ultrabubbles,
            contained_ultrabubbles,
        }
    }
}

fn cactus_graph_from_edges(
    node_count: usize,
    edges: impl IntoIterator<Item = Edge>,
) -> HyperSpokeGraph {
    let graph = SpokeGraph::new(node_count, edges);

    let inverted_comps = {
        let seg_hubs = (0..node_count as u32)
            .map(|i| {
                let node = Node::from(i);
                let left = graph.node_endpoint_hub(node.as_reverse());
                let right = graph.node_endpoint_hub(node.as_forward());
                (left, right)
            })
            .filter(|(a, b)| a != b)
            .collect::<Vec<_>>();

        let tec_graph = three_edge_connected::Graph::from_edges(
            seg_hubs.into_iter().map(|(l, r)| (l.ix(), r.ix())),
        );

        let components =
            three_edge_connected::find_components(&tec_graph.graph);

        let inverted = tec_graph.invert_components(components);

        inverted
    };

    let spoke_graph = Arc::new(graph);

    let mut cactus_graph = HyperSpokeGraph::new(spoke_graph);

    for comp in inverted_comps {
        let hubs = comp.into_iter().map(|i| HubId(i as u32));
        cactus_graph.merge_hub_partition(hubs);
    }

    cactus_graph.apply_deletions();

    cactus_graph
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacTreeVx {
    Net { vertex: VertexId },
    Chain { cycle: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacTreeEdge {
    Net {
        segment: Node,
        from: VertexId,
        to: VertexId,
    },
    Chain {
        net: VertexId,
        cycle: usize,
        prev_step: OrientedNode,
        this_step: OrientedNode,
    },
}

#[derive(Debug, Clone)]
pub struct CactusTree {
    pub cactus_graph: HyperSpokeGraph,
    pub graph: MatGraph<CacTreeVx, CacTreeEdge>,

    pub vertex_cycle_map: HashMap<VertexId, BTreeSet<usize>>,
    pub cycles: Vec<Cycle>,

    pub net_vertices: usize,
    pub net_edges: usize,

    pub chain_vertices: usize,
    pub chain_edges: usize,
}

impl CactusTree {
    /// constructs the cactus graph as well
    pub fn from_edges(
        node_count: usize,
        edges: impl IntoIterator<Item = Edge>,
    ) -> Self {
        let cactus_graph = cactus_graph_from_edges(node_count, edges);
        Self::from_cactus_graph(cactus_graph)
    }

    pub fn from_cactus_graph(cactus: HyperSpokeGraph) -> Self {
        let cycles = find_cactus_graph_cycles(&cactus);

        // we have the VertexIds from the cactus graph,
        // plus the chain vertices, one for each cycle

        let net_vx_count = cactus.vertex_count();
        let chain_vx_count = cycles.len();
        let vertex_count = net_vx_count + chain_vx_count;

        let mut vertex: Vec<CacTreeVx> = Vec::new();

        for (vxid, _) in cactus.vertices() {
            vertex.push(CacTreeVx::Net { vertex: vxid });
        }

        let mut edges: Vec<CacTreeEdge> = Vec::new();

        let seg_count = cactus.spoke_graph.max_endpoint.ix() / 2;

        let mut remaining_segments = RoaringBitmap::default();
        remaining_segments.insert_range(0..=seg_count as u32);

        let mut cycle_vertices: Vec<Vec<([OrientedNode; 2], VertexId)>> =
            Vec::new();

        for (cix, cycle) in cycles.iter().enumerate() {
            vertex.push(CacTreeVx::Chain { cycle: cix });

            let mut vertices = Vec::new();

            for (step_ix, step) in cycle.steps.iter().enumerate() {
                let prev_step = if step_ix == 0 {
                    let i = cycle.steps.len() - 1;
                    cycle.steps[i]
                } else {
                    cycle.steps[step_ix - 1]
                };
                remaining_segments.remove(step.node().ix() as u32);
                vertices
                    .push(([prev_step, *step], cactus.endpoint_vertex(*step)));
            }

            cycle_vertices.push(vertices);
        }

        for i in remaining_segments.iter() {
            let node = Node::from(i);

            let from_hub =
                cactus.spoke_graph.node_endpoint_hub(node.as_reverse());
            let from_vx = cactus.hub_vertex_map[from_hub.ix()];

            let to_hub =
                cactus.spoke_graph.node_endpoint_hub(node.as_forward());
            let to_vx = cactus.hub_vertex_map[to_hub.ix()];

            edges.push(CacTreeEdge::Net {
                segment: node,
                from: from_vx,
                to: to_vx,
            });
        }

        let net_edges = edges.len();

        let mut vertex_cycle_map: HashMap<_, BTreeSet<_>> = HashMap::new();

        for (cix, vertices) in cycle_vertices.into_iter().enumerate() {
            for ([prev_step, this_step], vertex) in vertices {
                vertex_cycle_map.entry(vertex).or_default().insert(cix);
                edges.push(CacTreeEdge::Chain {
                    cycle: cix,
                    net: vertex,
                    prev_step,
                    this_step,
                });
            }
        }

        use sprs::TriMat;

        let v_n = vertex.len();
        let e_n = edges.len();

        let mut adj: TriMat<u8> = TriMat::new((v_n, v_n));
        let mut inc: TriMat<u8> = TriMat::new((v_n, e_n));

        edges.sort();
        edges.dedup();

        for (i, edge) in edges.iter().enumerate() {
            match edge {
                CacTreeEdge::Net { from, to, .. } => {
                    adj.add_triplet(from.ix(), to.ix(), 1);
                    adj.add_triplet(to.ix(), from.ix(), 1);
                    inc.add_triplet(from.ix(), i, 1);
                    inc.add_triplet(to.ix(), i, 1);
                }
                CacTreeEdge::Chain { net, cycle, .. } => {
                    let c_i = net_vx_count + *cycle;
                    adj.add_triplet(net.ix(), c_i, 1);
                    adj.add_triplet(c_i, net.ix(), 1);
                    inc.add_triplet(net.ix(), i, 1);
                    inc.add_triplet(c_i, i, 1);
                }
            }
        }

        let adj: CsMat<u8> = adj.to_csc();
        let inc: CsMat<u8> = inc.to_csr();

        let chain_edges = edges.len() - net_edges;

        let graph = MatGraph {
            vertex_count,
            edge_count: edges.len(),
            adj,
            inc,
            vertex,
            edge: edges,
        };

        Self {
            cactus_graph: cactus,
            graph,
            cycles,
            vertex_cycle_map,
            net_vertices: net_vx_count,
            net_edges,
            chain_vertices: chain_vx_count,
            chain_edges,
        }
    }
}

impl CactusTree {
    pub fn chain_pair_dfs(
        &self,
        first_step: (usize, usize),
        mut chain_edge_f: impl FnMut(usize, usize),
    ) {
        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(first_step.0);

        let mut stack: Vec<(usize, usize)> = vec![first_step];

        while let Some((from, to)) = stack.pop() {
            if !visited.contains(&to) {
                visited.insert(to);

                let from_is_chain = from >= self.net_vertices;
                let to_is_net = to < self.net_vertices;

                if !from_is_chain && !to_is_net {
                    chain_edge_f(from, to);
                }

                if !from_is_chain && to_is_net {
                    // we're traversing a bridge, so we don't want to
                    // continue down this path
                    continue;
                }

                let neighbors = self.graph.adj.outer_view(to).unwrap();

                for &other in neighbors.indices().iter() {
                    if !visited.contains(&other) {
                        stack.push((to, other));
                    }
                }
            }
        }
    }

    /// returns (parent, node) traversal pairs
    pub fn rooted_cactus_forest(&self) -> Vec<Vec<(usize, usize)>> {
        let mut forest = Vec::new();

        let mut visited: HashSet<usize> = HashSet::new();

        let mut stack = Vec::new();

        // iterate through all chain vertices
        let chain_vertex_ixs =
            self.net_vertices..(self.net_vertices + self.chain_vertices);

        for ix in chain_vertex_ixs {
            let mut order = Vec::new();

            stack.push((None, ix));

            while let Some((parent, vi)) = stack.pop() {
                if !visited.contains(&vi) {
                    visited.insert(vi);

                    if let Some(parent) = parent {
                        order.push((parent, vi));
                    }

                    let on_chain = vi >= self.net_vertices;

                    let neighbors =
                        self.graph.neighbors(vi).into_iter().filter(|&vj| {
                            if on_chain {
                                vj < self.net_vertices
                            } else {
                                vj >= self.net_vertices
                            }
                        });

                    for vj in neighbors {
                        stack.push((Some(vi), vj));
                    }
                }
            }

            if !order.is_empty() {
                forest.push(order);
            }
        }

        forest
    }

    // vg_adj is an 2Nx2N adjacency matrix where N is the number of
    // segments in the variation graph; it lacks the connectivity
    // "within" segments (the black edges in the biedged repr.)
    //
    // Returns `None` if the net graph contains a bridge.
    fn chain_edge_net_graph(
        &self,
        vg_adj: &CsMat<u8>,
        chain_pair: (OrientedNode, OrientedNode),
        chain_ix: usize,
    ) -> Option<CsMat<u8>> {
        use sprs::TriMat;

        // chain pairs only have the one edge with one net vertex,
        // so that's the only vertex we need to project from
        let (net, _cycle_ix) =
            if let CacTreeEdge::Chain { net, cycle, .. } =
                self.graph.edge[self.net_edges + chain_ix]
            {
                (net, cycle)
            } else {
                unreachable!();
            };

        let endpoints = self.net_vertex_endpoints(net).collect::<BTreeSet<_>>();
        // let endpoints_vec = endpoints.iter().copied().

        let mut gray_edges: HashSet<(OrientedNode, OrientedNode)> =
            HashSet::new();
        let mut net_adj: TriMat<u8> = TriMat::new(vg_adj.shape());

        // find the edges between segments among the endpoints
        for &si in endpoints.iter() {
            if let Some(column) = vg_adj.outer_view(si.ix()) {
                for (isj, _) in column.iter() {
                    let sj = OrientedNode::from(isj as u32);
                    if endpoints.contains(&sj) {
                        let a = sj.min(si);
                        let b = sj.max(si);
                        gray_edges.insert((a, b));
                        net_adj.add_triplet(isj, si.ix(), 1);
                    }
                }
            }
        }

        // the gray edges are the subset of vg edges that connect endpoints
        // in the subgraph

        // the black edges are created from the cycles that the
        // endpoints are in the subgraph; since this is a chain pair,
        // there's just one contained cycle

        let mut black_edges: Vec<(OrientedNode, OrientedNode)> = Vec::new();

        let mut used_endpoints = HashSet::new();

        let cycles = self.vertex_cycle_map.get(&net).unwrap();

        for &cycle_ix in cycles {
            let cycle = &self.cycles[cycle_ix];

            // start by "flattening" the cycle, so that both segment endpoints
            // are present.
            let steps = cycle
                .steps
                .iter()
                .flat_map(|s| [*s, s.flip()])
                .filter(|s| endpoints.contains(s))
                // .flat_map(|s| [s.flip(), *s])
                .collect::<Vec<_>>();

            // if there's just two endpoints, there's just one step,
            // meaning one edge to add, so we're done
            if let &[a, b] = steps.as_slice() {
                let (ca, cb) = chain_pair;
                let a_chain = a == ca || a == cb;
                let b_chain = b == ca || b == cb;
                if endpoints.contains(&a)
                    && endpoints.contains(&b)
                    && !(a_chain || b_chain)
                {
                    black_edges.push((a, b));
                    used_endpoints.insert(a);
                    used_endpoints.insert(b);
                    continue;
                }
            }

            let mut edge_start: Option<OrientedNode> = None;

            for (i, w) in steps.windows(2).enumerate() {
                let a_in = endpoints.contains(&w[0])
                    && !used_endpoints.contains(&w[0]);
                let b_in = endpoints.contains(&w[1])
                    && !used_endpoints.contains(&w[1]);

                if w[0].node() == w[1].node() {
                    // traversing a segment
                    if edge_start.is_none() && a_in {
                        edge_start = Some(w[0]);
                    } else if let Some(start) = edge_start {
                        // the chain endpoints should have no black edges
                        let start_chain =
                            start == chain_pair.0 || start == chain_pair.1;
                        let end_chain =
                            w[1] == chain_pair.0 || w[1] == chain_pair.1;

                        if b_in && !start_chain && !end_chain {
                            edge_start = None;
                            black_edges.push((start, w[1]));
                        }
                    }
                } else {
                    // traversing an edge
                }
            }
        }

        let black_edges = black_edges
            .into_iter()
            .filter(|&(a, b)| {
                // filter out black edges between endpoints that are
                // already connected with a gray edge
                let min = a.min(b);
                let max = a.max(b);
                !gray_edges.contains(&(min, max))
            })
            .collect::<Vec<_>>();

        if black_edges.is_empty() {
            return None;
        }

        let mut endpoints = endpoints;

        for (a, b) in black_edges {
            if a.is_reverse() != b.is_reverse() {
                endpoints.remove(&a);
                endpoints.remove(&b);
                net_adj.add_triplet(b.ix(), a.ix(), 1);
            }
        }

        // if there's more than two endpoints left at this point,
        // that means there's at least one vertex lacking an incident
        // black edge, meaning this net graph has a bridge
        if endpoints.len() > 2 {
            return None;
        }

        Some(net_adj.to_csc())
    }

    pub fn project_segment_end(&self, end: OrientedNode) -> usize {
        let vx = self.cactus_graph.endpoint_vertex(end);
        vx.ix()
    }

    pub fn net_vertex_endpoints<'a>(
        &'a self,
        net: VertexId,
    ) -> impl Iterator<Item = OrientedNode> + 'a {
        let vx = self.cactus_graph.get_vertex(net);
        vx.hubs.iter().flat_map(move |h| {
            self.cactus_graph.spoke_graph.hub_endpoints[h.ix()]
                .iter()
                .copied()
        })
    }

    /// Returns the chain pairs in the graph, as the pair of segment
    /// endpoints and the corresponding chain edge index.
    pub fn enumerate_chain_pairs(
        &self,
    ) -> Vec<((OrientedNode, OrientedNode), usize)> {
        // let chain_range = (self.net_edges..(self.net_edges + self.chain_edges));

        let mut chain_pairs = Vec::with_capacity(self.chain_edges);

        let chain_range = 0..self.chain_edges;

        for chain_ix in chain_range {
            let edge_ix = self.net_edges + chain_ix;

            let (net, cycle_ix, steps) = if let CacTreeEdge::Chain {
                net,
                cycle,
                prev_step,
                this_step,
            } = self.graph.edge[edge_ix]
            {
                (net, cycle, [prev_step, this_step])
            } else {
                unreachable!();
            };

            // let prev = steps[0].flip();
            // let this = steps[1];

            let cycle = &self.cycles[cycle_ix];

            let net_spokes = self
                .cactus_graph
                .vertex_spokes(net)
                .filter(|(s, _)| {
                    let n = s.node();
                    cycle.steps.iter().any(|is| {
                        let ns = is.node();
                        n == ns
                    })
                })
                .collect::<Vec<_>>();
            if net_spokes.len() == 2 {
                let (a, _) = net_spokes[0];
                let (b, _) = net_spokes[1];
                if a.node() != b.node() {
                    chain_pairs.push(((a, b), chain_ix));
                }
            }
        }

        // output with chain pairs in shortest cycles first
        chain_pairs.sort_by_cached_key(|((_, _), ci)| {
            let edge_i = self.net_edges + *ci;
            let edge = &self.graph.edge[edge_i];
            if let CacTreeEdge::Chain { cycle, .. } = edge {
                self.cycles[*cycle].steps.len()
            } else {
                unreachable!();
            }
        });

        chain_pairs
    }
}

/// Returns the cycles in a cactus graph as a sequence of segment
/// traversals. The graph must be a cactus graph.
pub fn find_cactus_graph_cycles(graph: &HyperSpokeGraph) -> Vec<Cycle> {
    let mut visit = Vec::new();
    let mut visited_segments: HashSet<Node> = HashSet::default();
    let mut vx_visit: HashMap<VertexId, usize> = HashMap::default();
    let mut remaining_segments = RoaringBitmap::default();

    let max_ix = (graph.spoke_graph.max_endpoint.ix() / 2) - 1;
    remaining_segments.insert_range(0..max_ix as u32);

    graph.dfs_preorder(None, |i, step, vertex| {
        vx_visit.insert(vertex, i);
        visit.push((i, step, vertex));

        if let Some((_parent, step)) = step {
            let seg = step.node();
            visited_segments.insert(seg);
            remaining_segments.remove(seg.ix() as u32);
        }
    });

    // the DFS produces a spanning tree; from this, we can start from any
    // of the remaining segments and use the tree to reconstruct the cycle
    // it's part of

    let mut cycles: Vec<Cycle> = Vec::new();

    for seg_ix in remaining_segments {
        let node = Node::from(seg_ix);

        let l = graph.endpoint_vertex(node.as_reverse());
        let r = graph.endpoint_vertex(node.as_forward());

        let l_ix = *vx_visit.get(&l).unwrap();
        let r_ix = *vx_visit.get(&r).unwrap();

        if l_ix == r_ix {
            cycles.push(Cycle {
                endpoint: l,
                steps: vec![node.as_forward()],
                step_endpoints: vec![r],
            });
            continue;
        }

        let start = l_ix.min(r_ix);
        let end = l_ix.max(r_ix);

        let mut cur_ix = end;

        let mut cycle_steps = Vec::new();
        let mut step_endpoints = Vec::new();

        loop {
            if let Some((parent, incoming)) = visit[cur_ix].1 {
                cycle_steps.push(incoming.flip());
                step_endpoints.push(parent);

                // if parent's visit ix == start, we're done
                let parent_ix = *vx_visit.get(&parent).unwrap();
                cur_ix = parent_ix;

                if parent_ix == start {
                    break;
                }
            } else {
                break;
            }
        }

        step_endpoints.push(r);

        if start == l_ix {
            cycle_steps.push(node.as_reverse());
        } else {
            cycle_steps.push(node.as_forward());
        }

        cycle_steps.reverse();

        cycles.push(Cycle {
            endpoint: l,
            steps: cycle_steps,
            step_endpoints,
        });
    }

    cycles.sort_by_key(|c| c.steps.len());

    cycles
}

fn net_graph_is_acyclic(start: OrientedNode, graph: &CsMat<u8>) -> bool {
    let mut visited: HashSet<OrientedNode> = HashSet::new();
    let mut in_path: HashSet<OrientedNode> = HashSet::new();

    enum Inst {
        Push(OrientedNode),
        Pop(OrientedNode),
    }

    let mut stack: Vec<Inst> = vec![Inst::Push(start)];

    while let Some(inst) = stack.pop() {
        match inst {
            Inst::Push(vx) => {
                visited.insert(vx);
                in_path.insert(vx);
                // in_path.insert(

                let neighbors = graph.outer_view(vx.ix()).unwrap();
                let neighbors = neighbors
                    .indices()
                    .iter()
                    .map(|i| OrientedNode::from(*i as u32));

                for adj in neighbors {
                    if in_path.contains(&adj) {
                        return false;
                    } else if !visited.contains(&adj) {
                        stack.push(Inst::Push(adj));
                    }
                }

                stack.push(Inst::Pop(vx));
            }
            Inst::Pop(vx) => {
                in_path.remove(&vx);
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use sprs::vec::SparseIterTools;
    use waragraph_core::graph::PathIndex;

    use super::*;

    fn print_step(step: OrientedNode) {
        let c = ('a' as u8 + step.node().ix() as u8) as char;
        let o = if step.is_reverse() { "-" } else { "+" };
        print!("{c}{o}")
    }

    fn print_step_pair(steps: (OrientedNode, OrientedNode)) {
        print_step(steps.0);
        print!(", ");
        print_step(steps.1);
    }

    #[test]
    fn paper_fig3_ultrabubbles() {
        let node_count = 18;
        let edges = paper_fig3_graph_edges();
        let saboten = Saboten::from_edges(node_count, edges);

        // this one's finding too many
        println!("found {} ultrabubbles", saboten.ultrabubbles.len());

        for (i, (from, to)) in saboten.ultrabubbles.iter().enumerate() {
            //
            print!("{i}\t{from:?}{to:?}\t");
            print_step(*from);
            print!(", ");
            print_step(*to);
            println!();
        }
    }

    #[test]
    fn paper_fig5_ultrabubbles() {
        let node_count = 14;
        let edges = paper_fig5_graph_edges();
        let saboten = Saboten::from_edges(node_count, edges);

        println!("found {} ultrabubbles", saboten.ultrabubbles.len());

        for (i, (from, to)) in saboten.ultrabubbles.iter().enumerate() {
            //
            print!("{i}\t");
            print_step(*from);
            print!(", ");
            print_step(*to);
            // println!();
            println!("\t{from:?}, {to:?}");
        }

        println!(" --- cycles --- ");

        for (i, cycle) in saboten.cactus_tree.cycles.iter().enumerate() {
            print!("cycle {i} - ");
            for (j, &step) in cycle.steps.iter().enumerate() {
                if j > 0 {
                    print!(", ");
                }
                print_step(step);
            }
            println!();
        }
    }

    #[test]
    fn paper_fig3_cactus_tree() {
        let cactus_graph = paper_fig3_cactus_graph();

        let cactus_tree = CactusTree::from_cactus_graph(cactus_graph);

        println!("vertex_count: {}", cactus_tree.graph.vertex_count);
        println!("edge_count: {}", cactus_tree.graph.edge_count);

        assert_eq!(cactus_tree.graph.vertex_count, 19);
        assert_eq!(cactus_tree.graph.edge_count, 18);

        println!("-----------------------");
        cactus_tree.graph.print_adj();
        println!("-----------------------");
        cactus_tree.graph.print_inc();

        println!("---");

        println!("enumerating chain pairs!");
        println!("---");
        println!();
        let mut chain_pairs = cactus_tree.enumerate_chain_pairs();

        println!("{chain_pairs:?}");

        println!("\n\n------------\n\n");

        for (i, cycle) in cactus_tree.cycles.iter().enumerate() {
            // println!("{i}\t{cycle:?}");

            print!("cycle {i} steps\t[");

            for (i, step) in cycle.steps.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print_step(*step);
            }
            println!("]");

            /////////////

            print!("cycle {i} step endpoints\t[");

            for (i, &net_vx) in cycle.step_endpoints.iter().enumerate() {
                if i > 0 {
                    print!(";  ");
                }
                print!("[{net_vx:?}] ");
                let endpoints = cactus_tree.net_vertex_endpoints(net_vx);

                for (j, s) in endpoints.into_iter().enumerate() {
                    if j > 0 {
                        print!(", ");
                    }
                    print_step(s);
                }
            }
            println!("]");
        }

        println!();
        // println!(");

        for net_ix in 0..cactus_tree.net_vertices {
            let net_vx = VertexId(net_ix as u32);
            let endpoints = cactus_tree.net_vertex_endpoints(net_vx);

            print!("net vertex {net_ix}\t[");

            for s in endpoints {
                print_step(s);
            }
            println!("]");
        }

        println!();

        /*
        for seg in 0u32..18 {
            let node = Node::from(seg);
            let r = node.as_reverse();
            let f = node.as_forward();

            let pr = cactus_tree.project_segment_end(r);
            let pf = cactus_tree.project_segment_end(f);

            print!("{seg:2} - ");
            print_step(r);
            println!(" : Vertex {pr}");

            print!("{seg:2} - ");
            print_step(f);
            println!(" : Vertex {pf}");
        }
        */

        println!("chain pair count: {}", chain_pairs.len());
        chain_pairs.reverse();

        for ((a, b), chain_ix) in chain_pairs {
            print!("chain {chain_ix}: (");
            print_step(a);
            print!(", ");
            print_step(b);
            println!(")");
        }
    }

    #[test]
    fn paper_fig5_cactus_tree() {
        let cactus_graph = paper_fig5_cactus_graph();

        println!("cactus graph vertex count: {}", cactus_graph.vertex_count());

        let cactus_tree = CactusTree::from_cactus_graph(cactus_graph);

        println!("vertex_count: {}", cactus_tree.graph.vertex_count);
        println!("edge_count: {}", cactus_tree.graph.edge_count);

        assert_eq!(cactus_tree.graph.vertex_count, 15);
        assert_eq!(cactus_tree.graph.edge_count, 14);

        println!("enumerating chain pairs!");
        let mut chain_pairs = cactus_tree.enumerate_chain_pairs();

        for net_ix in 0..cactus_tree.net_vertices {
            let net_vx = VertexId(net_ix as u32);
            let endpoints = cactus_tree.net_vertex_endpoints(net_vx);

            print!("net vertex {net_ix}\t[");

            for s in endpoints {
                print_step(s);
            }
            println!("]");
        }

        println!();

        println!("chain pair count: {}", chain_pairs.len());
        chain_pairs.reverse();

        for ((a, b), chain_ix) in chain_pairs {
            print!("chain {chain_ix}: (");
            print_step(a);
            print!(", ");
            print_step(b);
            println!(")");
        }
    }

    #[test]
    fn chain_pair_containment() {
        let cactus_graph = paper_fig5_cactus_graph();
        let edges = paper_fig5_graph_edges();
        let cactus_tree = CactusTree::from_cactus_graph(cactus_graph);

        let forest = cactus_tree.rooted_cactus_forest();

        for (i, tree) in forest.iter().enumerate() {
            println!("{i}\t{tree:#?}");
        }

        println!();

        println!(" -- chain pairs --");

        let chain_pairs = cactus_tree.enumerate_chain_pairs();

        for &((a, b), chain_ix) in chain_pairs.iter() {
            println!("\n--------\n");
            print!("Chain pair: ");
            print_step(a);
            print!(", ");
            print_step(b);
            println!(", chain: {chain_ix}");
            // if let Some(net_graph) =
            //     cactus_tree.chain_edge_net_graph(&vg_adj, (a, b), chain_ix)
            // {
            //     sprs::visu::print_nnz_pattern(net_graph.view());
            // } else {
            //     zero_bes += 1;
            // }
        }
    }

    #[test]
    fn test_chain_pair_net_graph() {
        let cactus_graph = paper_fig3_cactus_graph();
        let edges = paper_fig3_graph_edges();
        let cactus_tree = CactusTree::from_cactus_graph(cactus_graph);

        let vg_adj =
            PathIndex::directed_adjacency_matrix(18, edges.iter().copied());

        // let cactus_graph = super::super::hyper::tests::alt_paper_cactus_graph();
        // let edges = super::super::tests::alt_paper_graph_edges();

        // let cactus_tree = CactusTree::from_cactus_graph(cactus_graph);

        // let vg_adj =
        //     PathIndex::directed_adjacency_matrix(14, edges.iter().copied());

        println!();

        let chain_pairs = cactus_tree.enumerate_chain_pairs();

        println!("---\nchain pair net graphs\n----\n");

        let mut zero_bes = 0;

        for &((a, b), chain_ix) in chain_pairs.iter() {
            println!("\n--------\n");
            print!("Chain pair: ");
            print_step(a);
            print!(", ");
            print_step(b);
            println!(", chain: {chain_ix}");
            if let Some(net_graph) =
                cactus_tree.chain_edge_net_graph(&vg_adj, (a, b), chain_ix)
            {
                sprs::visu::print_nnz_pattern(net_graph.view());
            } else {
                zero_bes += 1;
            }

            println!("\n--------\n");
        }

        println!("number of net graphs with zero black edges: {zero_bes}");
    }

    #[test]
    fn test_rooted_forest() {
        let graph_fig3 = paper_fig3_cactus_graph();

        let tree_fig3 = CactusTree::from_cactus_graph(graph_fig3);
        let forest_fig3 = tree_fig3.rooted_cactus_forest();

        let graph_fig5 = paper_fig5_cactus_graph();
        let tree_fig5 = CactusTree::from_cactus_graph(graph_fig5);
        let forest_fig5 = tree_fig5.rooted_cactus_forest();

        // TODO better tests once bridges are in

        assert_eq!(forest_fig3.len(), 3);
        assert_eq!(forest_fig3[0].len(), 7);
        assert_eq!(forest_fig3[1].len(), 4);
        assert_eq!(forest_fig3[2].len(), 2);

        assert_eq!(forest_fig5.len(), 1);
        assert_eq!(forest_fig5[0].len(), 12);

        println!("\n");

        let fig3_expected = vec![(6, 1), (3, 1), (1, 1)];

        // this looks correct (chain edges only, missing edges when
        // backtracking)
        for (i, (tree, expected)) in
            forest_fig3.iter().zip(fig3_expected).enumerate()
        {
            let inc = &tree_fig3.graph.inc;

            let (edges, missing): (Vec<_>, Vec<_>) = tree
                .iter()
                .map(|&(i, j)| {
                    // let (edges, missing): (Vec<_>, Vec<_>) = tree
                    //     .windows(2)
                    // .map(|ixs| {
                    // let i_row = inc.outer_view(ixs[0].1).unwrap();
                    // let j_row = inc.outer_view(ixs[1].1).unwrap();
                    let i_row = inc.outer_view(i).unwrap();
                    let j_row = inc.outer_view(j).unwrap();

                    let edge = i_row
                        .iter()
                        .nnz_zip(j_row.iter())
                        .map(|(i, _a, _b)| i)
                        .next()
                        .map(|edge_ix| tree_fig3.graph.edge[edge_ix]);

                    edge
                })
                .partition(|edge| edge.is_some());

            println!("{i}");
            for edge in &edges {
                println!("  {edge:?}");
            }
            println!();

            let (exp_edges, exp_missing) = expected;

            assert_eq!(edges.len(), exp_edges);
            assert_eq!(missing.len(), exp_missing);
        }

        println!("\n");

        let (edges, missing): (Vec<_>, Vec<_>) = forest_fig5[0]
            .windows(2)
            .map(|ixs| {
                let inc = &tree_fig5.graph.inc;
                let i_row = inc.outer_view(ixs[0].1).unwrap();
                let j_row = inc.outer_view(ixs[1].1).unwrap();

                let edge = i_row
                    .iter()
                    .nnz_zip(j_row.iter())
                    .map(|(i, _a, _b)| i)
                    .next()
                    .map(|edge_ix| tree_fig5.graph.edge[edge_ix]);

                edge
            })
            .partition(|edge| edge.is_some());

        assert_eq!(edges.len(), 7);
        assert_eq!(missing.len(), 5);
    }

    fn paper_fig3_cactus_graph() -> HyperSpokeGraph {
        let node_count = 18;
        let edges = paper_fig3_graph_edges();
        cactus_graph_from_edges(node_count, edges)
    }

    fn paper_fig5_cactus_graph() -> HyperSpokeGraph {
        let node_count = 14;
        let edges = paper_fig5_graph_edges();
        cactus_graph_from_edges(node_count, edges)
    }

    fn paper_fig3_graph_edges() -> Vec<Edge> {
        let oriented_node = |c: char, rev: bool| -> OrientedNode {
            let node = (c as u32) - 'a' as u32;
            OrientedNode::new(node, rev)
        };

        let edge = |a: char, a_r: bool, b: char, b_r: bool| -> Edge {
            let a = oriented_node(a, a_r);
            let b = oriented_node(b, b_r);
            Edge::new(a, b)
        };

        let edges = [
            ('a', 'b'),
            ('a', 'c'),
            ('b', 'd'),
            ('c', 'd'),
            ('d', 'e'),
            ('d', 'f'),
            ('e', 'g'),
            ('f', 'g'),
            ('f', 'h'),
            ('g', 'k'),
            ('g', 'l'),
            ('h', 'i'),
            ('h', 'j'),
            ('i', 'j'),
            ('j', 'l'),
            ('k', 'l'),
            ('l', 'm'),
            ('m', 'n'),
            ('m', 'o'),
            ('n', 'p'),
            ('o', 'p'),
            ('p', 'm'),
            ('p', 'q'),
            ('p', 'r'),
        ]
        .into_iter()
        .map(|(a, b)| edge(a, false, b, false))
        .collect::<Vec<_>>();

        edges
    }

    // corresponds to the graph in fig 5A in the paper
    fn paper_fig5_graph_edges() -> Vec<Edge> {
        let oriented_node = |c: char, rev: bool| -> OrientedNode {
            let node = (c as u32) - 'a' as u32;
            OrientedNode::new(node, rev)
        };

        let edge = |s: &str| -> Edge {
            let chars = s.chars().collect::<Vec<_>>();
            let a = chars[0];
            let a_rev = chars[1] == '-';
            let b = chars[2];
            let b_rev = chars[3] == '-';

            Edge::new(oriented_node(a, a_rev), oriented_node(b, b_rev))
        };

        let edges = [
            "a+n+", //
            "a+b+", //
            "b+c+", "b+d+", "c+e+", "d+e+", //
            "e+f+", "e+g+", "f+h+", "g+h+", //
            "h+m+", "h+i+", //
            "i+j+", "i+k+", "j+l+", "k+l+", //
            "l+m+", //
            "m+n+",
        ]
        .into_iter()
        .map(edge)
        .collect::<Vec<_>>();
        edges
    }
}
