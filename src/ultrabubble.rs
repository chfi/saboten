use fnv::{FnvHashMap, FnvHashSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Snarl {
    ChainPair { x: u64, y: u64 },
    BridgePair { x: u64, y: u64 },
}

pub enum SnarlType {
    ChainPair,
    BridgePair,
}

pub struct Snarl_ {
    snarl_type: SnarlType,
    pub x: u64,
    pub y: u64,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub struct ChainPair {
    pub x: u64,
    pub y: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BridgePair {
    pub x: u64,
    pub y: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChainEdge {
    pub net: u64,
    pub chain: u64,
}

impl Snarl {
    pub fn chain_pair(x: u64, y: u64) -> Self {
        Snarl::ChainPair { x, y }
    }

    pub fn bridge_pair(x: u64, y: u64) -> Self {
        Snarl::BridgePair { x, y }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ultrabubble {
    pub snarl: Snarl,
    pub contained_ultrabubbles: Vec<Snarl>,
}

impl Ultrabubble {
    pub fn from_snarl(snarl: Snarl) -> Self {
        Ultrabubble {
            snarl,
            contained_ultrabubbles: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ultrabubbles {
    pub bridge_pairs: FnvHashMap<BridgePair, FnvHashSet<ChainPair>>,
    pub chain_pairs: FnvHashMap<ChainPair, FnvHashSet<ChainPair>>,
}
