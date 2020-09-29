use crate::biedgedgraph::BiedgedGraph;

use petgraph::unionfind::UnionFind;

use fnv::FnvHashMap;

#[derive(Clone)]
pub struct Projection {
    pub size: usize,
    union_find: UnionFind<usize>,
    inverse: Option<FnvHashMap<u64, Vec<u64>>>,
}

pub type InverseProjection = FnvHashMap<u64, Vec<u64>>;

impl Projection {
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

    pub fn from_union_find(union_find: UnionFind<usize>, size: usize) -> Self {
        Self {
            size,
            union_find,
            inverse: None,
        }
    }

    pub fn projected(&self, x: u64) -> u64 {
        let x = x as usize;
        let p_x = self.union_find.find(x) as u64;
        p_x
    }

    pub fn find(&self, x: u64) -> u64 {
        self.projected(x)
    }

    pub fn projected_edge(&self, x: u64, y: u64) -> (u64, u64) {
        let x = self.union_find.find(x as usize);
        let y = self.union_find.find(y as usize);
        (x as u64, y as u64)
    }

    pub fn projected_mut(&mut self, x: u64) -> u64 {
        let x = x as usize;
        let p_x = self.union_find.find_mut(x) as u64;
        p_x
    }

    pub fn find_mut(&mut self, x: u64) -> u64 {
        self.projected_mut(x)
    }

    pub fn projected_edge_mut(&mut self, x: u64, y: u64) -> (u64, u64) {
        let x = self.union_find.find_mut(x as usize);
        let y = self.union_find.find_mut(y as usize);
        (x as u64, y as u64)
    }

    pub fn union(&mut self, x: u64, y: u64) -> bool {
        self.union_find.union(x as usize, y as usize)
    }

    /// Given a pair of vertices, return a corresponding pair with one
    /// of them replaced with their projection.
    pub fn kept_pair(&mut self, x: u64, y: u64) -> (u64, u64) {
        let union = self.union_find.find_mut(x as usize) as u64;
        if union == x {
            (union, y)
        } else {
            (union, x)
        }
    }

    fn build_inverse_replace(&mut self) {
        let mut inverse: InverseProjection = FnvHashMap::default();
        let reps = self.union_find.clone().into_labeling();

        for (i, k) in reps.iter().enumerate() {
            let i = i as u64;
            let k = *k as u64;
            inverse.entry(k).or_default().push(i);
        }

        self.inverse = Some(inverse);
    }

    pub fn build_inverse(&mut self) -> bool {
        if self.inverse.is_none() {
            self.build_inverse_replace();

            true
        } else {
            false
        }
    }

    pub fn mut_get_inverse(&mut self) -> &InverseProjection {
        if let Some(ref inv) = self.inverse {
            inv
        } else {
            self.build_inverse_replace();
            self.inverse.as_ref().unwrap()
        }
    }

    pub fn get_inverse(&self) -> Option<&InverseProjection> {
        self.inverse.as_ref()
    }

    pub fn projected_from(&self, x: u64) -> Option<&[u64]> {
        let inverse = self.inverse.as_ref()?;
        let projected = inverse.get(&x)?;
        Some(projected.as_slice())
    }
}
