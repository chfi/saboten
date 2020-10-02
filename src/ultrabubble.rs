#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Snarl {
    ChainPair { x: u64, y: u64 },
    BridgePair { x: u64, y: u64 },
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
