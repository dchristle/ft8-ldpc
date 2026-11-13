/// Maximum number of flipped information positions recorded in a candidate.
pub const CANDIDATE_POSITION_CAPACITY: usize = 6;
/// Outcome of the search and configured CRC requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Status {
    /// A candidate satisfies the configured success requirement.
    Success,
    /// The search completed without a CRC-valid candidate when one was required.
    NoValidCrc,
    /// The received ordering did not yield a full-rank generator basis.
    RankFailure,
    /// An admission was attempted beyond the configured candidate limit.
    CandidateCapHit,
}
/// Selected candidate, search outcome, and diagnostics.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodeResult {
    /// Search outcome under the configured CRC requirement.
    pub status: Status,
    /// Selected codeword in natural order, with bit zero in the least significant position.
    /// Zero-filled when no candidate is selected.
    pub codeword: [u64; 3],
    /// First 77 codeword bits, packed least-significant-bit first.
    /// Zero-filled when no candidate is selected.
    pub payload: [u64; 2],
    /// First 91 codeword bits, packed least-significant-bit first.
    /// Zero-filled when no candidate is selected.
    pub message_with_crc: [u64; 2],
    /// Sum of input LLR magnitudes at differing hard-decision positions; infinity when no candidate is selected.
    /// The sum can also overflow to infinity for a selected candidate.
    pub llr_distance: f32,
    /// Number of positions differing from the input hard decisions; zero for an empty result.
    pub hard_error_count: u16,
    /// Whether the selected codeword passed the CRC.
    /// False when no candidate is selected, even though the zero-filled placeholder passes CRC.
    pub crc_valid: bool,
    /// Counters and best-any-candidate measurements from this decode.
    pub diagnostics: Diagnostics,
    /// Collected candidates in selection order; empty when collection is disabled.
    pub candidates: Vec<Candidate>,
}
/// One admitted codeword and the error pattern that produced it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Candidate {
    /// Candidate codeword in natural order, with bit zero in the least significant position.
    pub codeword: [u64; 3],
    /// First 77 codeword bits, packed least-significant-bit first.
    pub payload: [u64; 2],
    /// First 91 codeword bits, packed least-significant-bit first.
    pub message_with_crc: [u64; 2],
    /// Sum of input LLR magnitudes at differing hard-decision positions.
    pub llr_distance: f32,
    /// Number of positions differing from the input hard decisions.
    pub hard_error_count: u16,
    /// Whether the candidate codeword passed the CRC.
    pub crc_valid: bool,
    /// Number of flipped positions in the generating error pattern.
    pub order: u8,
    /// Systematic-basis positions flipped by the error pattern, followed by zero padding.
    pub positions: [u8; 6],
    /// Number of meaningful entries in `positions`.
    pub num_positions: u8,
}
/// Search counters and the best candidate before CRC preference.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct Diagnostics {
    /// Rank obtained during basis construction.
    pub rank: usize,
    /// Configured number of CRC bits folded into the code.
    pub cascade_bits: usize,
    /// Number of active information positions.
    pub active_k: usize,
    /// Number of active redundancy positions.
    pub active_r: usize,
    /// Search steps visited across all stages.
    pub visited_steps: usize,
    /// Patterns examined across all stages, including rejected and cap-limited attempts.
    pub enumerated_patterns: usize,
    /// Candidates admitted for re-encoding.
    pub recoded_candidates: usize,
    /// Candidates assigned a weighted distance.
    pub scored_candidates: usize,
    /// Patterns examined by the main search, excluding the baseline.
    pub main_enumerated_patterns: usize,
    /// Candidates scored by the main search, excluding the baseline.
    pub main_scored_candidates: usize,
    /// Candidates converted from the systematic basis to result records.
    pub materialized_candidates: usize,
    /// Candidates checked against the CRC.
    pub crc_checked_candidates: usize,
    /// Patterns examined by first preprocessing.
    pub preprocess1_examined: usize,
    /// Patterns rejected by the first-preprocessing prefix filter.
    pub preprocess1_rejects: usize,
    /// Candidates scored by first preprocessing.
    pub preprocess1_scored_candidates: usize,
    /// Second-preprocessing patterns submitted for admission, including a cap-limited attempt.
    pub preprocess2_enumerated_patterns: usize,
    /// Candidates scored by second preprocessing.
    pub preprocess2_scored_candidates: usize,
    /// Number of masks prepared for second-preprocessing key lookups.
    pub preprocess2_probe_masks: usize,
    /// Pair-box lookups performed by second preprocessing, including empty matches.
    pub preprocess2_key_probes: usize,
    /// Candidates whose CRC check failed.
    pub crc_failures: usize,
    /// Whether an admission was attempted beyond the cap; reaching the cap alone does not set it.
    pub candidate_cap_hit: bool,
    /// Smallest scored distance in LLR units regardless of CRC validity; absent before any candidate.
    pub best_any_llr_distance: Option<f32>,
    /// Whether the candidate at `best_any_llr_distance` passed the CRC.
    pub best_any_crc_valid: bool,
}
