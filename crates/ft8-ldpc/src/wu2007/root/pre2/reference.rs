//! Box-free second-preprocessing reference that scans every pair for every probe
//! without carrying partial XORs or constructing diffs lazily.
use super::*;

impl Decoder {
    pub(super) fn pre2_box_free_reference<const MIN: bool>(
        &mut self,
        state: &mut SearchState<MIN>,
        baseline: PackedCodeword,
        k: usize,
    ) {
        let w = self.config.search_order;
        let tau = self.pairs.tau;
        let g = self.basis.encoded_basis_deltas;
        let rows = self.basis.parity_basis_deltas;
        let baseline_parity = extract_slice(&baseline, k, N - k);
        let masks: Vec<u128> = std::iter::once(0)
            .chain((1..=tau).map(|u| 1u128 << (tau - u)))
            .collect();
        enumerate(k, w, |step| {
            if state.diagnostics.candidate_cap_hit {
                return false;
            }
            let floor = if w >= 1 { step[0] as usize } else { k };
            if floor < 2 {
                return true;
            }
            let mut parity = baseline_parity;
            for &s in &step[..w] {
                parity = xor_red(parity, rows[s as usize]);
            }
            let step_key = key_of(parity, tau);
            for &mask in &masks {
                if state.diagnostics.candidate_cap_hit {
                    return false;
                }
                state.diagnostics.preprocess2_key_probes += 1;
                for i in 0..k {
                    for j in i + 1..k {
                        if key_of(xor_red(rows[i], rows[j]), tau) != step_key ^ mask || j >= floor {
                            continue;
                        }
                        if state.diagnostics.candidate_cap_hit {
                            return false;
                        }
                        let mut diff = baseline;
                        for &s in &step[..w] {
                            diff = xor_delta(diff, g[s as usize]);
                        }
                        diff = xor_delta(xor_delta(diff, g[i]), g[j]);
                        let mut positions = [0; 6];
                        positions[0] = i as u8;
                        positions[1] = j as u8;
                        positions[2..2 + w].copy_from_slice(&step[..w]);
                        self.admit(state, diff, (w + 2) as u8, positions);
                    }
                }
            }
            true
        });
    }
}
