//! Decoder configuration and CRC-gated candidate selection.
mod pre2;
mod search;
#[cfg(test)]
mod selection_tests;
#[cfg(test)]
mod smoke;
#[cfg(test)]
pub(crate) mod test_support;
use crate::wu2007::{basis::*, config::*, packed::*, result::*, scoring::*};
use ft8_ldpc_core::crc::verify_crc14_packed_message;
use ft8_ldpc_core::fec::CascadedGeneratorMatrix;
use ft8_ldpc_core::protocol::N;

#[derive(Clone, Copy, Default)]
struct PendingCandidate {
    order: u8,
    positions: [u8; 6],
}
// Fixed-capacity candidate storage reused across decodes.
#[derive(Clone, Copy, Default)]
struct CandidateBatch {
    pending: [PendingCandidate; 4],
    diffs: [PackedCodeword; 4],
    len: usize,
}

/// Stages that tests may replace with plain reference implementations.
#[cfg(test)]
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct ReferenceStages {
    pub fill: bool,
    pub dispatch: bool,
    pub basis: bool,
    pub main: bool,
    pub pre1: bool,
    pub list: bool,
    /// Second preprocessing without boxes.
    pub pre2_box_free: bool,
    /// The general step at every τ for w = 3.
    pub pre2_general: bool,
}

#[derive(Clone)]
/// Reusable ordered-statistics decoder with preprocessing and CRC-aware selection.
pub struct Decoder {
    config: Config,
    generator: CascadedGeneratorMatrix,
    scorer: PackedWeightedHammingScorer,
    basis: SystematicBasis,
    scratch: BasisBuildScratch,
    batch: CandidateBatch,
    prefix_words: [u64; 91],
    prefix_pairs: [PackedRedundancy; 91],
    pairs: pre2::PairWorkspace,
    #[cfg(test)]
    reference: ReferenceStages,
    #[cfg(test)]
    last_trace: Vec<(PackedCodeword, u8, [u8; 6])>,
}
// Only the packed difference and distance are needed by deferred selection.
// The optional CRC caches the baseline zero-error stopping check.
#[derive(Clone, Copy)]
struct MetricMinimum {
    diff: PackedCodeword,
    distance: f32,
    crc_valid: Option<bool>,
}

// MIN specializes the non-list minimum-metric path. Collection shares the
// original CRC-aware search and chooses its final winner outside the loop.
struct SearchState<const MIN: bool = false> {
    received: PackedCodeword,
    minimum: Option<MetricMinimum>,
    best_any: Option<Candidate>,
    best_valid: Option<Candidate>,
    candidates: Vec<Candidate>,
    diagnostics: Diagnostics,
    /// Selected candidates in selection order.
    #[cfg(test)]
    trace: Vec<(PackedCodeword, u8, [u8; 6])>,
}
impl Decoder {
    /// Validates configuration and allocates reusable decoder workspaces.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        config.validate()?;
        let k = 91 - config.cascade_bits;
        let pairs = if config.enable_second_preprocessing {
            pre2::PairWorkspace::new(pre2::resolve_tau(k, &config))
        } else {
            pre2::PairWorkspace::empty()
        };
        Ok(Self {
            generator: crate::wu2007::decoder::generator(config.cascade_bits),
            config,
            scorer: PackedWeightedHammingScorer::new(&[0.; N], [0; 3]),
            basis: SystematicBasis::new(),
            scratch: BasisBuildScratch::new(),
            batch: CandidateBatch::default(),
            prefix_words: [0; 91],
            prefix_pairs: [[0; 2]; 91],
            pairs,
            #[cfg(test)]
            reference: ReferenceStages::default(),
            #[cfg(test)]
            last_trace: Vec::new(),
        })
    }
    /// Decodes one finite LLR array in natural codeword order.
    pub fn decode(&mut self, llr: &[f32; N]) -> DecodeResult {
        match self.config.selection_policy {
            SelectionPolicy::BestCrcValid => self.decode_search::<false>(llr, true),
            SelectionPolicy::MinMetricThenCrc if self.config.collect_candidates => {
                self.decode_search::<false>(llr, false)
            }
            SelectionPolicy::MinMetricThenCrc => self.decode_search::<true>(llr, false),
        }
    }

    fn decode_search<const MIN: bool>(&mut self, llr: &[f32; N], prefer_crc: bool) -> DecodeResult {
        let k = 91 - self.config.cascade_bits;
        let mut diagnostics = Diagnostics {
            cascade_bits: self.config.cascade_bits,
            active_k: k,
            active_r: N - k,
            ..Default::default()
        };
        if let Err(rank) = self.rebuild_basis(llr, k) {
            diagnostics.rank = rank;
            return empty(Status::RankFailure, diagnostics);
        }
        diagnostics.rank = k;
        let mut alpha = [0.; N];
        let mut received = [0; 3];
        for s in 0..N {
            let value = llr[self.basis.sys_to_natural[s] as usize];
            alpha[s] = value.abs();
            if value < 0. {
                set_codeword_bit(&mut received, s);
            }
        }
        self.refill_scorer(&alpha);
        let estimate = extract_prefix(&received, k);
        let mut baseline = [0; 3];
        for q in 0..k {
            if get_message_bit(&estimate, q) {
                xor_in_place(&mut baseline, &self.basis.encoded_basis_deltas[q]);
            }
        }
        let baseline = xor_delta(baseline, received);
        let mut state: SearchState<MIN> = SearchState {
            received,
            minimum: None,
            best_any: None,
            best_valid: None,
            candidates: Vec::new(),
            diagnostics,
            #[cfg(test)]
            trace: Vec::new(),
        };
        self.admit(&mut state, baseline, 0, [0; 6]);
        if !self.config.collect_candidates && self.baseline_stops(&mut state) {
            return self.finish(state, prefer_crc);
        }
        self.batch.len = 0;
        let before_enumerated = state.diagnostics.enumerated_patterns;
        let before_scored = state.diagnostics.scored_candidates;
        for m in 1..=self.config.search_order {
            if state.diagnostics.candidate_cap_hit {
                break;
            }
            self.main_order(&mut state, baseline, k, m);
            self.flush_batch(&mut state);
        }
        state.diagnostics.main_enumerated_patterns =
            state.diagnostics.enumerated_patterns - before_enumerated;
        state.diagnostics.main_scored_candidates =
            state.diagnostics.scored_candidates - before_scored;
        if self.config.enable_first_preprocessing && !state.diagnostics.candidate_cap_hit {
            let before_scored = state.diagnostics.scored_candidates;
            self.first_preprocessing(&mut state, baseline, k);
            self.flush_batch(&mut state);
            state.diagnostics.preprocess1_scored_candidates =
                state.diagnostics.scored_candidates - before_scored;
        }
        if self.config.enable_second_preprocessing && !state.diagnostics.candidate_cap_hit {
            let before_enumerated = state.diagnostics.enumerated_patterns;
            let before_scored = state.diagnostics.scored_candidates;
            self.second_preprocessing(&mut state, baseline, k);
            self.flush_batch(&mut state);
            state.diagnostics.preprocess2_enumerated_patterns =
                state.diagnostics.enumerated_patterns - before_enumerated;
            state.diagnostics.preprocess2_scored_candidates =
                state.diagnostics.scored_candidates - before_scored;
        }
        self.finish(state, prefer_crc)
    }
    fn rebuild_basis(&mut self, llr: &[f32; N], k: usize) -> Result<(), usize> {
        #[cfg(test)]
        if self.reference.basis {
            return self.basis.rebuild_reference(
                &mut self.scratch,
                self.generator.generator(),
                llr,
                k,
            );
        }
        self.basis.rebuild_with_redundancy_order(
            &mut self.scratch,
            &self.generator,
            llr,
            k,
            self.config.redundancy_order,
        )
    }
    fn refill_scorer(&mut self, alpha: &[f32; N]) {
        #[cfg(test)]
        if self.reference.fill {
            self.scorer.update_reference(alpha, [0; 3]);
            return;
        }
        self.scorer.update(alpha, [0; 3]);
    }
    fn baseline_stops<const MIN: bool>(&self, state: &mut SearchState<MIN>) -> bool {
        if MIN {
            let Some(best) = &mut state.minimum else {
                return false;
            };
            // Preserve the original zero-hard-error stop, including zero LLRs:
            // a zero weighted distance alone is not sufficient.
            if best.diff != [0; 3] {
                return false;
            }
            if !self.config.require_crc {
                return true;
            }
            let message =
                extract_natural_message_with_crc(&state.received, &self.basis.natural_to_sys);
            let crc_valid = verify_crc14_packed_message(message);
            best.crc_valid = Some(crc_valid);
            state.diagnostics.crc_checked_candidates += 1;
            state.diagnostics.crc_failures += usize::from(!crc_valid);
            crc_valid
        } else {
            state.best_any.as_ref().is_some_and(|b| {
                b.hard_error_count == 0 && (!self.config.require_crc || b.crc_valid)
            })
        }
    }

    fn finish<const MIN: bool>(
        &mut self,
        mut state: SearchState<MIN>,
        prefer_crc: bool,
    ) -> DecodeResult {
        debug_assert_eq!(self.batch.len, 0);
        #[cfg(test)]
        {
            self.last_trace = std::mem::take(&mut state.trace);
        }
        if MIN {
            if let Some(best) = state.minimum {
                let codeword = inverse_permute(
                    &xor_delta(best.diff, state.received),
                    &self.basis.sys_to_natural,
                );
                let message_with_crc = message_with_crc_from_codeword(&codeword);
                let crc_valid = best.crc_valid.unwrap_or_else(|| {
                    state.diagnostics.crc_checked_candidates += 1;
                    let valid = verify_crc14_packed_message(message_with_crc);
                    state.diagnostics.crc_failures += usize::from(!valid);
                    valid
                });
                state.diagnostics.materialized_candidates += 1;
                state.diagnostics.best_any_llr_distance = Some(best.distance);
                state.diagnostics.best_any_crc_valid = crc_valid;
                return DecodeResult {
                    status: selected_status(
                        state.diagnostics.candidate_cap_hit,
                        crc_valid,
                        self.config.require_crc,
                    ),
                    codeword,
                    payload: extract_prefix(&codeword, 77),
                    message_with_crc,
                    llr_distance: best.distance,
                    hard_error_count: popcount_codeword(best.diff),
                    crc_valid,
                    diagnostics: state.diagnostics,
                    candidates: state.candidates,
                };
            }
        }
        if !prefer_crc {
            state.best_valid = None;
        }
        finalize(state, self.config.require_crc)
    }
    fn main_order<const MIN: bool>(
        &mut self,
        state: &mut SearchState<MIN>,
        baseline: PackedCodeword,
        k: usize,
        m: usize,
    ) {
        #[cfg(test)]
        if self.reference.main {
            return self.main_generic(state, baseline, k, m);
        }
        // Carry leading XORs through the fixed-depth loops.
        match m {
            1 => {
                for i0 in 0..k {
                    let row = self.basis.encoded_basis_deltas[i0];
                    if state.diagnostics.candidate_cap_hit {
                        return;
                    }
                    self.admit(
                        state,
                        xor_delta(baseline, row),
                        1,
                        [i0 as u8, 0, 0, 0, 0, 0],
                    );
                }
            }
            2 => {
                for i0 in 0..k - 1 {
                    let d0 = xor_delta(baseline, self.basis.encoded_basis_deltas[i0]);
                    for i1 in i0 + 1..k {
                        let row = self.basis.encoded_basis_deltas[i1];
                        if state.diagnostics.candidate_cap_hit {
                            return;
                        }
                        self.admit(
                            state,
                            xor_delta(d0, row),
                            2,
                            [i0 as u8, i1 as u8, 0, 0, 0, 0],
                        );
                    }
                }
            }
            3 => {
                for i0 in 0..k - 2 {
                    let d0 = xor_delta(baseline, self.basis.encoded_basis_deltas[i0]);
                    for i1 in i0 + 1..k - 1 {
                        let d01 = xor_delta(d0, self.basis.encoded_basis_deltas[i1]);
                        for i2 in i1 + 1..k {
                            let row = self.basis.encoded_basis_deltas[i2];
                            if state.diagnostics.candidate_cap_hit {
                                return;
                            }
                            self.admit(
                                state,
                                xor_delta(d01, row),
                                3,
                                [i0 as u8, i1 as u8, i2 as u8, 0, 0, 0],
                            );
                        }
                    }
                }
            }
            _ => self.main_generic(state, baseline, k, m),
        }
    }
    #[inline(always)]
    fn admit<const MIN: bool>(
        &mut self,
        state: &mut SearchState<MIN>,
        diff: PackedCodeword,
        order: u8,
        positions: [u8; 6],
    ) {
        state.diagnostics.visited_steps += 1;
        state.diagnostics.enumerated_patterns += 1;
        self.reserve_and_score(state, diff, order, positions);
    }
    // Preprocessing counts examined patterns separately, including rejections.
    #[inline(always)]
    fn reserve_and_score<const MIN: bool>(
        &mut self,
        state: &mut SearchState<MIN>,
        diff: PackedCodeword,
        order: u8,
        positions: [u8; 6],
    ) {
        if !self.reserve(state) {
            return;
        }
        // Score the baseline immediately so it can stop the search before batching.
        #[cfg(test)]
        let batched = order != 0 && !self.reference.dispatch;
        #[cfg(not(test))]
        let batched = order != 0;
        if batched {
            let slot = self.batch.len;
            if !MIN || cfg!(test) {
                self.batch.pending[slot] = PendingCandidate { order, positions };
            }
            self.batch.diffs[slot] = diff;
            self.batch.len += 1;
            if self.batch.len == 4 {
                self.flush_batch(state);
            }
            return;
        }
        let distance = self.scorer.distance_from_diff(&diff);
        self.select(state, diff, order, positions, distance);
    }
    #[inline(always)]
    fn reserve<const MIN: bool>(&self, state: &mut SearchState<MIN>) -> bool {
        if state.diagnostics.recoded_candidates >= self.config.max_recoded_candidates {
            state.diagnostics.candidate_cap_hit = true;
            return false;
        }
        state.diagnostics.recoded_candidates += 1;
        state.diagnostics.scored_candidates += 1;
        true
    }
    fn flush_batch<const MIN: bool>(&mut self, state: &mut SearchState<MIN>) {
        let len = self.batch.len;
        if len == 4 {
            let distances = self
                .scorer
                .calculate_batch_weighted_hamming_f32_from_diffs_portable_simd(&self.batch.diffs);
            for (i, distance) in distances.into_iter().enumerate() {
                let pending = self.batch.pending[i];
                self.select(
                    state,
                    self.batch.diffs[i],
                    pending.order,
                    pending.positions,
                    distance,
                );
            }
        } else {
            for i in 0..len {
                let pending = self.batch.pending[i];
                let distance = self.scorer.distance_from_diff(&self.batch.diffs[i]);
                self.select(
                    state,
                    self.batch.diffs[i],
                    pending.order,
                    pending.positions,
                    distance,
                );
            }
        }
        self.batch.len = 0;
    }
    #[inline(always)]
    fn select<const MIN: bool>(
        &self,
        state: &mut SearchState<MIN>,
        diff: PackedCodeword,
        order: u8,
        positions: [u8; 6],
        distance: f32,
    ) {
        #[cfg(test)]
        state.trace.push((diff, order, positions));
        if MIN {
            if state.minimum.as_ref().is_none_or(|b| distance < b.distance) {
                state.minimum = Some(MetricMinimum {
                    diff,
                    distance,
                    crc_valid: None,
                });
            }
            return;
        }
        let diag = &mut state.diagnostics;
        if diag.best_any_llr_distance.is_none_or(|d| distance < d) {
            diag.best_any_llr_distance = Some(distance);
        }
        let ba = state
            .best_any
            .as_ref()
            .is_none_or(|b| distance < b.llr_distance);
        let bv = state
            .best_valid
            .as_ref()
            .is_none_or(|b| distance < b.llr_distance);
        if !ba && !bv && !self.config.collect_candidates {
            return;
        }
        self.materialize(state, diff, order, positions, distance, ba, bv);
    }
    // Keep the infrequent CRC/unpermutation/record path out of the scoring
    // loop. The distance-only pruning above is inlined into batch selection.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn materialize<const MIN: bool>(
        &self,
        state: &mut SearchState<MIN>,
        diff: PackedCodeword,
        order: u8,
        positions: [u8; 6],
        distance: f32,
        ba: bool,
        bv: bool,
    ) {
        let diag = &mut state.diagnostics;
        let x = xor_delta(diff, state.received);
        diag.crc_checked_candidates += 1;
        let codeword;
        let message_with_crc;
        let crc_valid;
        if ba || self.config.collect_candidates {
            codeword = inverse_permute(&x, &self.basis.sys_to_natural);
            message_with_crc = message_with_crc_from_codeword(&codeword);
            crc_valid = verify_crc14_packed_message(message_with_crc);
            if !crc_valid {
                diag.crc_failures += 1;
            }
        } else {
            message_with_crc = extract_natural_message_with_crc(&x, &self.basis.natural_to_sys);
            crc_valid = verify_crc14_packed_message(message_with_crc);
            if !crc_valid {
                diag.crc_failures += 1;
                return;
            }
            codeword = inverse_permute(&x, &self.basis.sys_to_natural);
        }
        diag.materialized_candidates += 1;
        let rec = Candidate {
            codeword,
            payload: extract_prefix(&codeword, 77),
            message_with_crc,
            llr_distance: distance,
            hard_error_count: popcount_codeword(diff),
            crc_valid,
            order,
            positions,
            num_positions: order,
        };
        if ba {
            diag.best_any_crc_valid = crc_valid;
            state.best_any = Some(rec.clone());
        }
        if crc_valid && bv {
            state.best_valid = Some(rec.clone());
        }
        if self.config.collect_candidates {
            #[cfg(test)]
            if self.reference.list {
                list_insert_reference(&mut state.candidates, rec, self.config.max_list_candidates);
                return;
            }
            list_insert(&mut state.candidates, rec, self.config.max_list_candidates);
        }
    }
}
/// Inserts into the bounded sorted list, placing exact ties at any index among equals.
fn list_insert(list: &mut Vec<Candidate>, rec: Candidate, max: usize) {
    if max == 0 {
        return;
    }
    let p = list
        .binary_search_by(|old| compare_candidates(old, &rec))
        .unwrap_or_else(|p| p);
    if p < max {
        list.insert(p, rec);
        if list.len() > max {
            list.pop();
        }
    }
}
/// Inserts linearly after every entry that compares equal.
#[cfg(test)]
pub(crate) fn list_insert_reference(list: &mut Vec<Candidate>, rec: Candidate, max: usize) {
    if max == 0 {
        return;
    }
    let p = list
        .iter()
        .position(|old| compare_candidates(old, &rec).is_gt())
        .unwrap_or(list.len());
    if p >= max {
        return;
    }
    list.insert(p, rec);
    if list.len() > max {
        list.pop();
    }
}
fn compare_candidates(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    b.crc_valid
        .cmp(&a.crc_valid)
        .then_with(|| a.llr_distance.partial_cmp(&b.llr_distance).unwrap())
        .then(a.hard_error_count.cmp(&b.hard_error_count))
}
fn empty(status: Status, diagnostics: Diagnostics) -> DecodeResult {
    DecodeResult {
        status,
        codeword: [0; 3],
        payload: [0; 2],
        message_with_crc: [0; 2],
        llr_distance: f32::INFINITY,
        hard_error_count: 0,
        crc_valid: false,
        diagnostics,
        candidates: Vec::new(),
    }
}
fn finalize<const MIN: bool>(state: SearchState<MIN>, require_crc: bool) -> DecodeResult {
    let winner = state.best_valid.or(state.best_any);
    let Some(winner) = winner else {
        return empty(
            if state.diagnostics.candidate_cap_hit {
                Status::CandidateCapHit
            } else {
                Status::NoValidCrc
            },
            state.diagnostics,
        );
    };
    let status = selected_status(
        state.diagnostics.candidate_cap_hit,
        winner.crc_valid,
        require_crc,
    );
    DecodeResult {
        status,
        codeword: winner.codeword,
        payload: winner.payload,
        message_with_crc: winner.message_with_crc,
        llr_distance: winner.llr_distance,
        hard_error_count: winner.hard_error_count,
        crc_valid: winner.crc_valid,
        diagnostics: state.diagnostics,
        candidates: state.candidates,
    }
}

// Acceptance is shared by both policies, after the winner has been selected.
fn selected_status(candidate_cap_hit: bool, crc_valid: bool, require_crc: bool) -> Status {
    if candidate_cap_hit {
        Status::CandidateCapHit
    } else if crc_valid || !require_crc {
        Status::Success
    } else {
        Status::NoValidCrc
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new(Config::default()).expect("valid default configuration")
    }
}
