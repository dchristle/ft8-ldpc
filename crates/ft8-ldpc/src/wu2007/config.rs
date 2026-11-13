//! Decoder configuration and validation.
use super::basis::RedundancyOrder;

/// Chooses a winner from the candidates actually searched, independently of acceptance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SelectionPolicy {
    /// Select the first minimum-distance candidate, then check its CRC.
    MinMetricThenCrc,
    /// Prefer the first minimum-distance CRC-valid candidate, falling back to the
    /// first minimum-distance candidate if none passes CRC.
    #[default]
    BestCrcValid,
}

/// Search controls for the Wu2007 decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Number of CRC bits folded into the code, from 0 through 14.
    pub cascade_bits: usize,
    /// Ordering of the non-basis positions after reduction.
    pub redundancy_order: RedundancyOrder,
    /// Maximum main-search pattern weight; preprocessing adds one or two positions.
    pub search_order: usize,
    /// Search patterns of weight `search_order + 1` using a parity-prefix filter.
    pub enable_first_preprocessing: bool,
    /// Search patterns of weight `search_order + 2` using pair-box lookups.
    pub enable_second_preprocessing: bool,
    /// First-preprocessing prefix length; `None` uses half the redundancy length.
    /// Values above the redundancy length are clamped; ignored when the stage is disabled.
    pub parity_prefix_len: Option<usize>,
    /// First-preprocessing mismatch threshold; `None` uses `2 * (search_order + 3)`.
    /// Ignored when first preprocessing is disabled.
    pub preprocessing_threshold: Option<usize>,
    /// Second-preprocessing prefix length; `None` derives it from the message length and order.
    /// Values above the redundancy length are clamped; ignored when the stage is disabled.
    pub second_preprocessing_prefix_len: Option<usize>,
    /// Maximum admitted candidates; zero produces an empty result with `Status::CandidateCapHit`.
    pub max_recoded_candidates: usize,
    /// Maximum collected candidates; zero keeps the list empty without changing selection.
    pub max_list_candidates: usize,
    /// Include a CRC-preferred ordered candidate list in the result. This checks
    /// CRC and reconstructs every scored candidate under either selection policy,
    /// even when `max_list_candidates` is zero.
    pub collect_candidates: bool,
    /// How the searched candidates are ranked for final selection.
    pub selection_policy: SelectionPolicy,
    /// Require CRC validity for success and the baseline zero-error early stop.
    /// This does not change candidate selection.
    pub require_crc: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cascade_bits: 0,
            redundancy_order: RedundancyOrder::ReliabilityDescending,
            search_order: 3,
            enable_first_preprocessing: true,
            enable_second_preprocessing: false,
            parity_prefix_len: None,
            preprocessing_threshold: None,
            second_preprocessing_prefix_len: None,
            max_recoded_candidates: 300_000,
            max_list_candidates: 32,
            collect_candidates: false,
            selection_policy: SelectionPolicy::BestCrcValid,
            require_crc: true,
        }
    }
}

/// A configuration that cannot be represented by the decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The CRC cascade exceeds the fourteen available bits.
    CascadeBitsOutOfRange {
        /// Supplied number of cascade bits.
        value: usize,
        /// Largest supported number of cascade bits.
        max: usize,
    },
    /// Main search and preprocessing exceed the candidate position capacity.
    CandidatePositionCapacityExceeded {
        /// Requested main-search pattern weight.
        search_order: usize,
        /// Extra positions required by enabled preprocessing.
        preprocessing_extra_layers: usize,
        /// Number of positions available in a candidate record.
        capacity: usize,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CascadeBitsOutOfRange { value, max } => {
                write!(f, "cascade_bits must be in 0..={max}, got {value}")
            }
            Self::CandidatePositionCapacityExceeded { .. } => {
                f.write_str("search_order plus preprocessing layers must fit 6 candidate positions")
            }
        }
    }
}
impl std::error::Error for ConfigError {}

impl Config {
    /// Checks cascade size, then the combined search and preprocessing order.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.cascade_bits > 14 {
            return Err(ConfigError::CascadeBitsOutOfRange {
                value: self.cascade_bits,
                max: 14,
            });
        }
        let layers = if self.enable_second_preprocessing {
            2
        } else {
            usize::from(self.enable_first_preprocessing)
        };
        if self.search_order.checked_add(layers).is_none_or(|n| n > 6) {
            return Err(ConfigError::CandidatePositionCapacityExceeded {
                search_order: self.search_order,
                preprocessing_extra_layers: layers,
                capacity: 6,
            });
        }
        Ok(())
    }
}
