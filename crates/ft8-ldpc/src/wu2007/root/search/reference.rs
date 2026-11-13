//! Plain reference versions of the first-preprocessing kernels, compiled for
//! tests only. They count per pattern and test the flag
//! before every pattern, with no blocks, lanes or deferred counts.
use super::*;

/// Sets lane l when m + popcount(p ^ x[l]) <= theta.
pub(crate) fn survivor_mask_reference(p: u64, x: &[u64], m: usize, theta: usize) -> u8 {
    x.iter().enumerate().fold(0, |mask, (lane, &x)| {
        mask | (u8::from(m + (p ^ x).count_ones() as usize <= theta) << lane)
    })
}

impl Decoder {
    /// Counts one examined pattern and returns whether it is rejected.
    fn reference_examine(state: &mut SearchState, m: usize, q: u32, theta: usize) -> bool {
        let diagnostics = &mut state.diagnostics;
        diagnostics.visited_steps += 1;
        diagnostics.enumerated_patterns += 1;
        diagnostics.preprocess1_examined += 1;
        let rejected = m + q as usize > theta;
        diagnostics.preprocess1_rejects += usize::from(rejected);
        rejected
    }

    pub(in crate::wu2007::root) fn pre1_reference(
        &mut self,
        state: &mut SearchState,
        baseline: PackedCodeword,
        k: usize,
        m: usize,
        b: u64,
        theta: usize,
    ) {
        match m {
            2 => self.pre1_reference_order2(state, baseline, k, b, theta),
            3 => self.pre1_reference_order3(state, baseline, k, b, theta),
            4 => self.pre1_reference_order4(state, baseline, k, b, theta),
            _ => unreachable!("references exist for orders 2 to 4"),
        }
    }

    pub(in crate::wu2007::root) fn pre1_reference_order2(
        &mut self,
        state: &mut SearchState,
        baseline: PackedCodeword,
        k: usize,
        b: u64,
        theta: usize,
    ) {
        let (g, w) = (self.basis.encoded_basis_deltas, self.prefix_words);
        for i0 in 0..k {
            let p0 = b ^ w[i0];
            let d0 = xor_delta(baseline, g[i0]);
            for i1 in i0 + 1..k {
                if state.diagnostics.candidate_cap_hit {
                    return;
                }
                if Self::reference_examine(state, 2, (p0 ^ w[i1]).count_ones(), theta) {
                    continue;
                }
                self.reserve_and_score(
                    state,
                    xor_delta(d0, g[i1]),
                    2,
                    [i0 as u8, i1 as u8, 0, 0, 0, 0],
                );
            }
        }
    }

    /// Builds each surviving diff from the baseline without carrying partial diffs.
    pub(in crate::wu2007::root) fn pre1_reference_order3(
        &mut self,
        state: &mut SearchState,
        baseline: PackedCodeword,
        k: usize,
        b: u64,
        theta: usize,
    ) {
        let (g, w) = (self.basis.encoded_basis_deltas, self.prefix_words);
        for i0 in 0..k {
            let p0 = b ^ w[i0];
            for i1 in i0 + 1..k {
                let p01 = p0 ^ w[i1];
                for i2 in i1 + 1..k {
                    if state.diagnostics.candidate_cap_hit {
                        return;
                    }
                    if Self::reference_examine(state, 3, (p01 ^ w[i2]).count_ones(), theta) {
                        continue;
                    }
                    let diff = xor_delta(xor_delta(xor_delta(baseline, g[i0]), g[i1]), g[i2]);
                    self.reserve_and_score(state, diff, 3, [i0 as u8, i1 as u8, i2 as u8, 0, 0, 0]);
                }
            }
        }
    }

    pub(in crate::wu2007::root) fn pre1_reference_order4(
        &mut self,
        state: &mut SearchState,
        baseline: PackedCodeword,
        k: usize,
        b: u64,
        theta: usize,
    ) {
        let (g, w) = (self.basis.encoded_basis_deltas, self.prefix_words);
        for i0 in 0..k {
            let (p0, d0) = (b ^ w[i0], xor_delta(baseline, g[i0]));
            for i1 in i0 + 1..k {
                let (p01, d01) = (p0 ^ w[i1], xor_delta(d0, g[i1]));
                for i2 in i1 + 1..k {
                    let (p012, d012) = (p01 ^ w[i2], xor_delta(d01, g[i2]));
                    for i3 in i2 + 1..k {
                        if state.diagnostics.candidate_cap_hit {
                            return;
                        }
                        if Self::reference_examine(state, 4, (p012 ^ w[i3]).count_ones(), theta) {
                            continue;
                        }
                        self.reserve_and_score(
                            state,
                            xor_delta(d012, g[i3]),
                            4,
                            [i0 as u8, i1 as u8, i2 as u8, i3 as u8, 0, 0],
                        );
                    }
                }
            }
        }
    }
}
