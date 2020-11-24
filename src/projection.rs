use crate::biedgedgraph::BiedgedGraph;

use petgraph::unionfind::UnionFind;

use fxhash::FxHashMap;

/// Encapsulates a mapping of vertices in an original graph to their
/// projections in another. Also provides an inverse mapping, so as to
/// find which vertices were projected into a given vertex.
#[derive(Clone)]
pub struct Projection {
    pub size: usize,
    union_find: UnionFind<usize>,
    inverse: Option<FxHashMap<u64, Vec<u64>>>,
}

pub type InverseProjection = FxHashMap<u64, Vec<u64>>;

impl Projection {
    /// Utility function for use when cloning a graph and its
    /// projection map, with the intention of mutating them. As the
    /// inverse map must be rebuilt when there's been any change to
    /// the projection, there's no sense in cloning that part as well.
    pub fn copy_without_inverse(&self) -> Self {
        Projection {
            size: self.size,
            union_find: self.union_find.clone(),
            inverse: None,
        }
    }

    /// Construct a new projection map for a biedged graph. The graph
    /// must have its vertex IDs tightly packed, starting from zero or
    /// one.
    pub fn new_for_biedged_graph(graph: &BiedgedGraph) -> Self {
        let size = (graph.max_net_vertex + 1) as usize;
        let union_find = UnionFind::new(size);
        let inverse = None;
        Self {
            size,
            union_find,
            inverse,
        }
    }

    #[inline]
    pub fn find(&self, x: u64) -> u64 {
        let x = x as usize;
        self.union_find.find(x) as u64
    }

    #[inline]
    pub fn find_mut(&mut self, x: u64) -> u64 {
        let x = x as usize;
        self.union_find.find_mut(x) as u64
    }

    #[inline]
    pub fn find_edge(&self, x: u64, y: u64) -> (u64, u64) {
        let x = self.union_find.find(x as usize);
        let y = self.union_find.find(y as usize);
        (x as u64, y as u64)
    }

    #[inline]
    pub fn find_edge_mut(&mut self, x: u64, y: u64) -> (u64, u64) {
        let x = self.union_find.find_mut(x as usize);
        let y = self.union_find.find_mut(y as usize);
        (x as u64, y as u64)
    }

    #[inline]
    pub fn union(&mut self, x: u64, y: u64) -> bool {
        self.union_find.union(x as usize, y as usize)
    }

    #[inline]
    pub fn equiv(&self, x: u64, y: u64) -> bool {
        self.union_find.equiv(x as usize, y as usize)
    }

    /// Given a pair of vertices, return a corresponding pair with one
    /// of them replaced with their projection. The first one is
    /// guaranteed to be the representative of the union, so it's safe
    /// to use as an ID in the graph.
    #[inline]
    pub fn kept_pair(&mut self, x: u64, y: u64) -> (u64, u64) {
        let union = self.union_find.find_mut(x as usize) as u64;
        if union == x {
            (union, y)
        } else {
            (union, x)
        }
    }

    /// Constructs the inverse projection map, replacing it if it
    /// already exists.
    fn build_inverse_replace(&mut self) {
        let mut inverse: InverseProjection = FxHashMap::default();
        let reps = self.union_find.clone().into_labeling();

        for (i, k) in reps.iter().enumerate() {
            let i = i as u64;
            let k = *k as u64;
            inverse.entry(k).or_default().push(i);
        }

        self.inverse = Some(inverse);
    }

    /// Constructs the inverse projection map if it does not already
    /// exist. Returns false if the map already existed and did not
    /// have to be built.
    pub fn build_inverse(&mut self) -> bool {
        if self.inverse.is_none() {
            self.build_inverse_replace();
            true
        } else {
            false
        }
    }

    /// Retrieves the inverse map, building it if it does not already
    /// exist.
    pub fn mut_get_inverse(&mut self) -> &InverseProjection {
        if let Some(ref inv) = self.inverse {
            inv
        } else {
            self.build_inverse_replace();
            self.inverse.as_ref().unwrap()
        }
    }

    /// Retrieves the inverse map, or None if it hasn't been built.
    pub fn get_inverse(&self) -> Option<&InverseProjection> {
        self.inverse.as_ref()
    }

    /// Given a projected vertex, return a slice containing all the
    /// vertex in the original graph that projected to it. Returns
    /// None if the inverse map hasn't been built.
    pub fn projected_from(&self, x: u64) -> Option<&[u64]> {
        let inverse = self.inverse.as_ref()?;
        let projected = inverse.get(&x)?;
        Some(projected.as_slice())
    }
}

/// Maps a vertex ID in the original (non-biedged) graph to its black
/// edge vertices in the corresponding biedged graph.
#[inline]
pub fn id_to_black_edge(n: u64) -> (u64, u64) {
    let left = n * 2;
    let right = left + 1;
    (left, right)
}

/// Given a vertex ID in a biedged graph, retrieve its opposite vertex
/// and return their black edge.
#[inline]
pub fn end_to_black_edge(n: u64) -> (u64, u64) {
    if n % 2 == 0 {
        (n, n + 1)
    } else {
        (n - 1, n)
    }
}

/// Given a vertex in a biedged graph, retrieve its opposite vertex.
#[inline]
pub fn opposite_vertex(n: u64) -> u64 {
    if n % 2 == 0 {
        n + 1
    } else {
        n - 1
    }
}

#[inline]
/// Maps a vertex in a biedged graph to its ID in the original,
/// non-biedged graph.
pub fn id_from_black_edge(n: u64) -> u64 {
    n / 2
}
