use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Snarl {
    ChainPair { x: u64, y: u64 },
    BridgePair { x: u64, y: u64 },
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
pub struct Ultrabubble {
    pub start: u64,
    pub end: u64,
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
