//! Second preprocessing: test error patterns of weight
//! w + 2 are found by probing an index of all position pairs (the pair boxes)
//! with the key of each weight-w step.
#[cfg(test)]
use super::search::enumerate;
use super::search::enumerate_from;
use super::*;
#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;

/// Keys of at most this many bits use the bucket array; longer keys use the
/// sorted array.
const BUCKET_TAU_MAX: usize = 16;
const MAX_PAIRS: usize = 91 * 90 / 2;

/// Resolves the redundancy-prefix length, clamped to the available redundancy.
pub(super) fn resolve_tau(k: usize, config: &Config) -> usize {
    let r = N - k;
    let w = config.search_order;
    match config.second_preprocessing_prefix_len {
        Some(tau) => tau.min(r),
        None if w == 0 => r,
        None => r.min((3.0f32 * (k as f32 / w as f32).log2()).ceil() as usize),
    }
}

/// Extracts the low τ bits of a redundancy value as a key.
#[inline(always)]
fn key_of(value: PackedRedundancy, tau: usize) -> u128 {
    if tau <= 64 {
        (value[0] & low_mask(tau)) as u128
    } else {
        value[0] as u128 | ((value[1] & low_mask(tau - 64)) as u128) << 64
    }
}

/// One record of the sorted array: ordering by (key, i, j) equals a stable
/// sort by key of the pairs generated in lexicographic order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SortedPair {
    key: u128,
    pair: [u8; 2],
}

/// Pair boxes and probe masks, owned by the decoder and rebuilt on every
/// decode that runs the stage.
#[derive(Clone)]
pub(super) struct PairWorkspace {
    tau: usize,
    offsets: Vec<u16>,
    cursors: Vec<u16>,
    entries: Vec<[u8; 2]>,
    sorted: Vec<SortedPair>,
    masks: Vec<u128>,
}

impl PairWorkspace {
    pub(super) fn empty() -> Self {
        Self {
            tau: 0,
            offsets: Vec::new(),
            cursors: Vec::new(),
            entries: Vec::new(),
            sorted: Vec::new(),
            masks: Vec::new(),
        }
    }

    pub(super) fn new(tau: usize) -> Self {
        let mut workspace = Self::empty();
        workspace.tau = tau;
        if tau <= BUCKET_TAU_MAX {
            workspace.offsets = vec![0; (1 << tau) + 1];
            workspace.cursors = vec![0; 1 << tau];
            workspace.entries = vec![[0; 2]; MAX_PAIRS];
        } else {
            workspace.sorted = Vec::with_capacity(MAX_PAIRS);
        }
        // Probe the exact key first, then each single-bit difference from high to low.
        workspace.masks.push(0);
        workspace
            .masks
            .extend((1..=tau).map(|u| 1u128 << (tau - u)));
        workspace
    }

    /// Build the boxes of all pairs i < j < k from the redundancy rows.
    fn build(&mut self, rows: &[PackedRedundancy], k: usize) {
        let tau = self.tau;
        if tau <= BUCKET_TAU_MAX {
            let groups = 1usize << tau;
            let mask = low_mask(tau);
            self.cursors.fill(0);
            for i in 0..k {
                for j in i + 1..k {
                    self.cursors[((rows[i][0] ^ rows[j][0]) & mask) as usize] += 1;
                }
            }
            self.offsets[0] = 0;
            for g in 0..groups {
                self.offsets[g + 1] = self.offsets[g] + self.cursors[g];
            }
            self.cursors.copy_from_slice(&self.offsets[..groups]);
            for i in 0..k {
                for j in i + 1..k {
                    let g = ((rows[i][0] ^ rows[j][0]) & mask) as usize;
                    self.entries[self.cursors[g] as usize] = [i as u8, j as u8];
                    self.cursors[g] += 1;
                }
            }
        } else {
            self.sorted.clear();
            for i in 0..k {
                for j in i + 1..k {
                    self.sorted.push(SortedPair {
                        key: key_of(xor_red(rows[i], rows[j]), tau),
                        pair: [i as u8, j as u8],
                    });
                }
            }
            self.sorted.sort_unstable();
        }
    }

    /// Returns the entry range containing every pair with the given key.
    #[inline(always)]
    fn range(&self, key: u128) -> (usize, usize) {
        if self.tau <= BUCKET_TAU_MAX {
            let g = key as usize;
            (self.offsets[g] as usize, self.offsets[g + 1] as usize)
        } else {
            let first = self.sorted.partition_point(|entry| entry.key < key);
            let mut end = first;
            while end < self.sorted.len() && self.sorted[end].key == key {
                end += 1;
            }
            (first, end)
        }
    }

    #[inline(always)]
    fn entry(&self, index: usize) -> [u8; 2] {
        if self.tau <= BUCKET_TAU_MAX {
            self.entries[index]
        } else {
            self.sorted[index].pair
        }
    }
}

/// A weight-w step with its carried diff and parity, for the general walkers.
struct Step {
    diff: PackedCodeword,
    parity: PackedRedundancy,
    /// Systematic positions of the step, in slots 2.. of the array.
    positions: [u8; 6],
    floor: usize,
}

impl Decoder {
    pub(super) fn second_preprocessing(
        &mut self,
        state: &mut SearchState,
        baseline: PackedCodeword,
        k: usize,
    ) {
        let w = self.config.search_order;
        let tau = self.pairs.tau;
        state.diagnostics.preprocess2_probe_masks = self.pairs.masks.len();
        let parity = extract_slice(&baseline, k, N - k);
        #[cfg(test)]
        if self.reference.pre2_box_free {
            self.pre2_box_free_reference(state, baseline, k);
            return;
        }
        #[cfg(test)]
        let lazy = w == 3 && tau <= 64 && !self.reference.pre2_general;
        #[cfg(not(test))]
        let lazy = w == 3 && tau <= 64;
        if lazy {
            let mask = low_mask(tau);
            for i in 0..k {
                self.prefix_words[i] = self.basis.parity_basis_deltas[i][0] & mask;
            }
        }
        self.pairs.build(&self.basis.parity_basis_deltas[..k], k);
        // The walkers read the boxes through a local, so their loops need not
        // reload the workspace after every admission.
        let pairs = std::mem::replace(&mut self.pairs, PairWorkspace::empty());
        match w {
            0 => {
                let step = Step {
                    diff: baseline,
                    parity,
                    positions: [0; 6],
                    floor: k,
                };
                self.pre2_general_step(state, &pairs, baseline, 0, &step);
            }
            1 => self.pre2_walk_w1(state, &pairs, baseline, parity, k),
            2 => self.pre2_walk_w2(state, &pairs, baseline, parity, k),
            3 if lazy && tau <= BUCKET_TAU_MAX => self.pre2_walk_w3_lazy::<true>(
                state,
                &pairs,
                baseline,
                parity[0] & low_mask(tau),
                k,
            ),
            3 if lazy => self.pre2_walk_w3_lazy::<false>(
                state,
                &pairs,
                baseline,
                parity[0] & low_mask(tau),
                k,
            ),
            3 => self.pre2_walk_w3_general(state, &pairs, baseline, parity, k),
            _ => self.pre2_walk_generic(state, &pairs, baseline, parity, k, w),
        }
        self.pairs = pairs;
    }

    fn pre2_walk_w1(
        &mut self,
        state: &mut SearchState,
        pairs: &PairWorkspace,
        baseline: PackedCodeword,
        parity: PackedRedundancy,
        k: usize,
    ) {
        for s0 in 2..k {
            if state.diagnostics.candidate_cap_hit {
                return;
            }
            let step = Step {
                diff: xor_delta(baseline, self.basis.encoded_basis_deltas[s0]),
                parity: xor_red(parity, self.basis.parity_basis_deltas[s0]),
                positions: [0, 0, s0 as u8, 0, 0, 0],
                floor: s0,
            };
            self.pre2_general_step(state, pairs, baseline, 1, &step);
        }
    }

    fn pre2_walk_w2(
        &mut self,
        state: &mut SearchState,
        pairs: &PairWorkspace,
        baseline: PackedCodeword,
        parity: PackedRedundancy,
        k: usize,
    ) {
        for s0 in 2..k - 1 {
            let d0 = xor_delta(baseline, self.basis.encoded_basis_deltas[s0]);
            let p0 = xor_red(parity, self.basis.parity_basis_deltas[s0]);
            for s1 in s0 + 1..k {
                if state.diagnostics.candidate_cap_hit {
                    return;
                }
                let step = Step {
                    diff: xor_delta(d0, self.basis.encoded_basis_deltas[s1]),
                    parity: xor_red(p0, self.basis.parity_basis_deltas[s1]),
                    positions: [0, 0, s0 as u8, s1 as u8, 0, 0],
                    floor: s0,
                };
                self.pre2_general_step(state, pairs, baseline, 2, &step);
            }
        }
    }

    fn pre2_walk_w3_general(
        &mut self,
        state: &mut SearchState,
        pairs: &PairWorkspace,
        baseline: PackedCodeword,
        parity: PackedRedundancy,
        k: usize,
    ) {
        for s0 in 2..k - 2 {
            let d0 = xor_delta(baseline, self.basis.encoded_basis_deltas[s0]);
            let p0 = xor_red(parity, self.basis.parity_basis_deltas[s0]);
            for s1 in s0 + 1..k - 1 {
                let d01 = xor_delta(d0, self.basis.encoded_basis_deltas[s1]);
                let p01 = xor_red(p0, self.basis.parity_basis_deltas[s1]);
                for s2 in s1 + 1..k {
                    if state.diagnostics.candidate_cap_hit {
                        return;
                    }
                    let step = Step {
                        diff: xor_delta(d01, self.basis.encoded_basis_deltas[s2]),
                        parity: xor_red(p01, self.basis.parity_basis_deltas[s2]),
                        positions: [0, 0, s0 as u8, s1 as u8, s2 as u8, 0],
                        floor: s0,
                    };
                    self.pre2_general_step(state, pairs, baseline, 3, &step);
                }
            }
        }
    }

    /// Steps of weight 4 (or any weight, in principle) through the generic
    /// enumerator, starting at position 2 because steps with s0 < 2 have no
    /// pair below them.
    fn pre2_walk_generic(
        &mut self,
        state: &mut SearchState,
        pairs: &PairWorkspace,
        baseline: PackedCodeword,
        parity: PackedRedundancy,
        k: usize,
        w: usize,
    ) {
        enumerate_from(2, k, w, |steps| {
            if state.diagnostics.candidate_cap_hit {
                return false;
            }
            let mut step = Step {
                diff: baseline,
                parity,
                positions: [0; 6],
                floor: steps[0] as usize,
            };
            for (slot, &s) in steps[..w].iter().enumerate() {
                xor_in_place(&mut step.diff, &self.basis.encoded_basis_deltas[s as usize]);
                step.parity = xor_red(step.parity, self.basis.parity_basis_deltas[s as usize]);
                step.positions[2 + slot] = s;
            }
            self.pre2_general_step(state, pairs, baseline, w, &step);
            !state.diagnostics.candidate_cap_hit
        });
    }

    /// Probes every mask in list order using the key from the carried parity.
    fn pre2_general_step(
        &mut self,
        state: &mut SearchState,
        pairs: &PairWorkspace,
        baseline: PackedCodeword,
        w: usize,
        step: &Step,
    ) {
        if step.floor < 2 || state.diagnostics.candidate_cap_hit {
            return;
        }
        let key = key_of(step.parity, pairs.tau);
        for &mask in &pairs.masks {
            if state.diagnostics.candidate_cap_hit {
                return;
            }
            state.diagnostics.preprocess2_key_probes += 1;
            let (first, end) = pairs.range(key ^ mask);
            let mut diff = Some(step.diff);
            self.pre2_admit_range(
                state,
                pairs,
                baseline,
                (first, end),
                step.floor,
                &step.positions,
                w,
                &mut diff,
            );
        }
    }

    /// Walks weight-three steps with one-word keys, building each diff only at its first match.
    #[inline(never)]
    fn pre2_walk_w3_lazy<const BUCKET: bool>(
        &mut self,
        state: &mut SearchState,
        pairs: &PairWorkspace,
        baseline: PackedCodeword,
        baseline_key: u64,
        k: usize,
    ) {
        let tau = pairs.tau;
        let mut masks = [0u64; 65];
        for (u, mask) in masks.iter_mut().enumerate().take(tau + 1).skip(1) {
            *mask = 1 << (tau - u);
        }
        let masks = &masks[..=tau];
        let groups = if BUCKET { 1usize << tau } else { 1 };
        let offsets = if BUCKET {
            &pairs.offsets[..=groups]
        } else {
            &[][..]
        };
        let mut probes = 0;
        'stage: for s0 in 2..k - 2 {
            let p0 = baseline_key ^ self.prefix_words[s0];
            for s1 in s0 + 1..k - 1 {
                let p01 = p0 ^ self.prefix_words[s1];
                for s2 in s1 + 1..k {
                    let key = p01 ^ self.prefix_words[s2];
                    let positions = [0, 0, s0 as u8, s1 as u8, s2 as u8, 0];
                    let mut diff = None;
                    for &mask in masks {
                        let probe = key ^ mask;
                        let range = if BUCKET {
                            let g = probe as usize & (groups - 1);
                            (offsets[g] as usize, offsets[g + 1] as usize)
                        } else {
                            pairs.range(probe as u128)
                        };
                        probes += 1;
                        // An empty box cannot set the cap flag, so check it only after admission.
                        if range.0 != range.1 {
                            self.pre2_admit_range(
                                state, pairs, baseline, range, s0, &positions, 3, &mut diff,
                            );
                            if state.diagnostics.candidate_cap_hit {
                                break 'stage;
                            }
                        }
                    }
                }
            }
        }
        state.diagnostics.preprocess2_key_probes += probes;
    }

    /// Admits the passing pairs of one box; `step_diff` is the
    /// carried diff of the step, or `None` until the first passing pair builds
    /// it from the baseline and the step positions.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn pre2_admit_range(
        &mut self,
        state: &mut SearchState,
        pairs: &PairWorkspace,
        baseline: PackedCodeword,
        (first, end): (usize, usize),
        floor: usize,
        positions: &[u8; 6],
        w: usize,
        step_diff: &mut Option<PackedCodeword>,
    ) {
        for index in first..end {
            if state.diagnostics.candidate_cap_hit {
                return;
            }
            let [i, j] = pairs.entry(index);
            if j as usize >= floor {
                continue;
            }
            let diff = *step_diff
                .get_or_insert_with(|| self.diff_from_positions(baseline, &positions[2..2 + w]));
            let mut pair_positions = *positions;
            pair_positions[0] = i;
            pair_positions[1] = j;
            let diff = xor_delta(
                xor_delta(diff, self.basis.encoded_basis_deltas[i as usize]),
                self.basis.encoded_basis_deltas[j as usize],
            );
            self.admit(state, diff, (w + 2) as u8, pair_positions);
        }
    }
}
