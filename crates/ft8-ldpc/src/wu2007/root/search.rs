//! Main-search and first-preprocessing kernels.
#[cfg(test)]
pub(super) mod reference;
mod vector;
use super::*;

// Use one zero-padded buffer, last position fastest. A false return
// unwinds the walk only after the visitor has set the cap flag.
pub(super) fn enumerate(k: usize, m: usize, visit: impl FnMut([u8; 6]) -> bool) {
    enumerate_from(0, k, m, visit);
}

/// The same walk over the positions `first..k`.
pub(super) fn enumerate_from(
    first: usize,
    k: usize,
    m: usize,
    mut visit: impl FnMut([u8; 6]) -> bool,
) {
    fn walk(
        k: usize,
        m: usize,
        depth: usize,
        start: usize,
        positions: &mut [u8; 6],
        visit: &mut impl FnMut([u8; 6]) -> bool,
    ) -> bool {
        if depth == m {
            return visit(*positions);
        }
        for p in start..=k - (m - depth) {
            positions[depth] = p as u8;
            if !walk(k, m, depth + 1, p + 1, positions, visit) {
                return false;
            }
        }
        true
    }
    if first + m <= k {
        walk(k, m, 0, first, &mut [0; 6], &mut visit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent small pattern space: enumerate subsets as integers, convert
    // to ascending positions, then sort. This also checks the recursive walk.
    fn patterns(k: usize, m: usize) -> Vec<[u8; 6]> {
        let mut out = Vec::new();
        for subset in 0u32..1 << k {
            if subset.count_ones() as usize == m {
                let mut positions = [0; 6];
                let mut slot = 0;
                for p in 0..k {
                    if subset & (1 << p) != 0 {
                        positions[slot] = p as u8;
                        slot += 1;
                    }
                }
                out.push(positions);
            }
        }
        out.sort();
        out
    }

    fn state() -> SearchState {
        SearchState {
            received: [0; 3],
            minimum: None,
            best_any: None,
            best_valid: None,
            candidates: Vec::new(),
            diagnostics: Diagnostics::default(),
            trace: Vec::new(),
        }
    }

    fn synthetic_decoder() -> (Decoder, PackedCodeword) {
        let mut config = Config {
            cascade_bits: 0,
            ..Default::default()
        };
        config.collect_candidates = true;
        config.max_list_candidates = 100;
        let mut decoder = Decoder::new(config).unwrap();
        for n in 0..N {
            decoder.basis.sys_to_natural[n] = n as u16;
            decoder.basis.natural_to_sys[n] = n as u16;
        }
        // Seven information positions vary within a 91-bit information set
        // and 83-bit redundancy.
        for i in 0..7 {
            let row = &mut decoder.basis.encoded_basis_deltas[i];
            set_codeword_bit(row, i);
            for n in 91..N {
                if (n * 19 + i * 31) % 17 < 8 {
                    set_codeword_bit(row, n);
                }
            }
            decoder.basis.parity_basis_deltas[i] = extract_slice(row, 91, 83);
        }
        decoder
            .scorer
            .update(&std::array::from_fn(|n| (n % 11 + 1) as f32 / 8.), [0; 3]);
        let mut baseline = [0; 3];
        for n in 91..N {
            if n % 3 == 0 {
                set_codeword_bit(&mut baseline, n);
            }
        }
        (decoder, baseline)
    }

    #[test]
    fn generic_walk_is_lexicographic_zero_padded_and_interruptible() {
        for m in 0..=6 {
            let expected = patterns(7, m);
            let mut actual = Vec::new();
            enumerate(7, m, |p| {
                actual.push(p);
                true
            });
            assert_eq!(actual, expected);
            let mut prefix = Vec::new();
            enumerate(7, m, |p| {
                prefix.push(p);
                prefix.len() < 3
            });
            assert_eq!(prefix, expected[..expected.len().min(3)]);
        }
        enumerate(2, 3, |_| panic!("impossible pattern"));
    }

    #[test]
    fn every_pre1_order_matches_direct_bit_test_at_prefix_and_cap_boundaries() {
        let (mut decoder, baseline) = synthetic_decoder();
        for m in 1..=6 {
            let all = patterns(7, m);
            for t in [0, 1, 63, 64, 65, 83] {
                let mask = prefix_mask_red(t);
                let b = mask_red(extract_slice(&baseline, 91, 83), mask);
                for i in 0..7 {
                    let pair = mask_red(decoder.basis.parity_basis_deltas[i], mask);
                    decoder.prefix_words[i] = pair[0];
                    decoder.prefix_pairs[i] = pair;
                }
                for theta in [m - 1, m, m + t / 2, m + t, usize::MAX] {
                    // Reference q is counted directly from the complete diff,
                    // independently of the masked prefix tables being tested.
                    let direct = |positions: [u8; 6]| {
                        let mut diff = baseline;
                        for &p in &positions[..m] {
                            diff = xor_delta(diff, decoder.basis.encoded_basis_deltas[p as usize]);
                        }
                        let q = (91..91 + t).filter(|&n| get_codeword_bit(&diff, n)).count();
                        (diff, m + q <= theta)
                    };
                    let survivors = all.iter().filter(|&&p| direct(p).1).count();
                    for cap in [0, 1, 3, survivors, survivors + 1] {
                        decoder.config.max_recoded_candidates = cap;
                        let mut expected = state();
                        for &positions in &all {
                            let mut diff = baseline;
                            for &p in &positions[..m] {
                                diff =
                                    xor_delta(diff, decoder.basis.encoded_basis_deltas[p as usize]);
                            }
                            let q = (91..91 + t).filter(|&n| get_codeword_bit(&diff, n)).count();
                            expected.diagnostics.visited_steps += 1;
                            expected.diagnostics.enumerated_patterns += 1;
                            expected.diagnostics.preprocess1_examined += 1;
                            if m + q > theta {
                                expected.diagnostics.preprocess1_rejects += 1;
                            } else {
                                if decoder.reserve(&mut expected) {
                                    let distance = decoder.scorer.distance_from_diff(&diff);
                                    decoder.select(
                                        &mut expected,
                                        diff,
                                        m as u8,
                                        positions,
                                        distance,
                                    );
                                }
                                if expected.diagnostics.candidate_cap_hit {
                                    break;
                                }
                            }
                        }
                        let mut actual = state();
                        if t > 64 {
                            decoder.pre1_generic::<true, false>(
                                &mut actual,
                                baseline,
                                7,
                                m,
                                b,
                                theta,
                            );
                        } else {
                            match m {
                                1 => decoder.pre1_order1(&mut actual, baseline, 7, b[0], theta),
                                2 => decoder.pre1_order2_vector(
                                    &mut actual,
                                    baseline,
                                    7,
                                    b[0],
                                    theta,
                                ),
                                3 => decoder.pre1_order3_vector(
                                    &mut actual,
                                    baseline,
                                    7,
                                    b[0],
                                    theta,
                                ),
                                4 => decoder.pre1_order4_vector(
                                    &mut actual,
                                    baseline,
                                    7,
                                    b[0],
                                    theta,
                                ),
                                _ => decoder.pre1_generic::<false, false>(
                                    &mut actual,
                                    baseline,
                                    7,
                                    m,
                                    b,
                                    theta,
                                ),
                            }
                        }
                        decoder.flush_batch(&mut actual);
                        // Includes ordered candidates, winner bits, distances,
                        // positions and all diagnostics, including lazy cap flag.
                        assert_eq!(
                            format!("{:?}", finalize(actual, true)),
                            format!("{:?}", finalize(expected, true)),
                            "m={m}, t={t}, theta={theta}, cap={cap}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn batch_reservations_and_every_partial_flush_match_immediate_selection() {
        let (mut decoder, baseline) = synthetic_decoder();
        for count in 0..=12 {
            for cap in 0..=13 {
                decoder.config.max_recoded_candidates = cap;
                let mut expected = state();
                let mut actual = state();
                for i in 0..count {
                    let positions = [(i % 7) as u8, 0, 0, 0, 0, 0];
                    let diff = decoder.diff_from_positions(baseline, &positions[..1]);
                    expected.diagnostics.visited_steps += 1;
                    expected.diagnostics.enumerated_patterns += 1;
                    if decoder.reserve(&mut expected) {
                        let distance = decoder.scorer.distance_from_diff(&diff);
                        decoder.select(&mut expected, diff, 1, positions, distance);
                    }
                    decoder.admit(&mut actual, diff, 1, positions);
                    assert_eq!(
                        actual.diagnostics.recoded_candidates,
                        expected.diagnostics.recoded_candidates
                    );
                    assert_eq!(
                        actual.diagnostics.scored_candidates,
                        expected.diagnostics.scored_candidates
                    );
                    assert_eq!(decoder.batch.len, actual.diagnostics.recoded_candidates % 4);
                    if expected.diagnostics.candidate_cap_hit {
                        break;
                    }
                }
                decoder.flush_batch(&mut actual);
                assert_eq!(decoder.batch.len, 0);
                // A second empty flush cannot change selection or diagnostics.
                decoder.flush_batch(&mut actual);
                assert_eq!(
                    format!("{:?}", finalize(actual, true)),
                    format!("{:?}", finalize(expected, true)),
                    "count={count}, cap={cap}"
                );
            }
        }
    }

    #[test]
    fn generic_main_orders_preserve_refused_pattern_and_exact_cap() {
        let (mut decoder, baseline) = synthetic_decoder();
        for m in 4..=6 {
            let all = patterns(7, m);
            for cap in [0, 1, 3, all.len(), all.len() + 1] {
                decoder.config.max_recoded_candidates = cap;
                let mut expected = state();
                for &positions in &all {
                    let mut diff = baseline;
                    for &p in &positions[..m] {
                        diff = xor_delta(diff, decoder.basis.encoded_basis_deltas[p as usize]);
                    }
                    expected.diagnostics.visited_steps += 1;
                    expected.diagnostics.enumerated_patterns += 1;
                    if decoder.reserve(&mut expected) {
                        let distance = decoder.scorer.distance_from_diff(&diff);
                        decoder.select(&mut expected, diff, m as u8, positions, distance);
                    }
                    if expected.diagnostics.candidate_cap_hit {
                        break;
                    }
                }
                let mut actual = state();
                decoder.main_order(&mut actual, baseline, 7, m);
                decoder.flush_batch(&mut actual);
                assert_eq!(
                    format!("{:?}", finalize(actual, true)),
                    format!("{:?}", finalize(expected, true)),
                    "m={m}, cap={cap}"
                );
            }
        }
    }

    #[test]
    fn order_zero_preprocessing_resolves_prefix_bounds_and_matches_main_order_one() {
        let llr = std::array::from_fn(|n| {
            let magnitude = (n % 19 + 1) as f32 / 8.;
            if n % 3 == 0 {
                -magnitude
            } else {
                magnitude
            }
        });
        for cascade in [0, 7, 14] {
            let mut config = Config {
                cascade_bits: cascade,
                ..Default::default()
            };
            config.collect_candidates = true;
            config.search_order = 1;
            config.enable_first_preprocessing = false;
            let main = Decoder::new(config.clone()).unwrap().decode(&llr);
            config.search_order = 0;
            config.enable_first_preprocessing = true;
            config.preprocessing_threshold = Some(usize::MAX);
            for t in [0, 63, 64, 65, 174 - (91 - cascade), usize::MAX] {
                config.parity_prefix_len = Some(t);
                let mut expected = main.clone();
                let d = &mut expected.diagnostics;
                d.preprocess1_examined = d.main_enumerated_patterns;
                d.preprocess1_scored_candidates = d.main_scored_candidates;
                d.main_enumerated_patterns = 0;
                d.main_scored_candidates = 0;
                let actual = Decoder::new(config.clone()).unwrap().decode(&llr);
                assert_eq!(
                    format!("{actual:?}"),
                    format!("{expected:?}"),
                    "cascade={cascade}, t={t}"
                );
            }
            // t=0 must still reject every pattern when theta < m, even
            // though the prefix has no set bits.
            config.parity_prefix_len = Some(0);
            config.preprocessing_threshold = Some(0);
            let rejected = Decoder::new(config).unwrap().decode(&llr);
            assert_eq!(rejected.diagnostics.preprocess1_examined, 91 - cascade);
            assert_eq!(rejected.diagnostics.preprocess1_rejects, 91 - cascade);
            assert_eq!(rejected.diagnostics.preprocess1_scored_candidates, 0);
            assert_eq!(rejected.diagnostics.scored_candidates, 1);
            assert!(!rejected.diagnostics.candidate_cap_hit);
        }
    }
}

impl Decoder {
    #[inline(always)]
    pub(super) fn diff_from_positions(
        &self,
        mut baseline: PackedCodeword,
        positions: &[u8],
    ) -> PackedCodeword {
        for &p in positions {
            xor_in_place(&mut baseline, &self.basis.encoded_basis_deltas[p as usize]);
        }
        baseline
    }

    pub(super) fn main_generic<const MIN: bool>(
        &mut self,
        state: &mut SearchState<MIN>,
        baseline: PackedCodeword,
        k: usize,
        m: usize,
    ) {
        enumerate(k, m, |positions| {
            if state.diagnostics.candidate_cap_hit {
                return false;
            }
            let diff = self.diff_from_positions(baseline, &positions[..m]);
            self.admit(state, diff, m as u8, positions);
            !state.diagnostics.candidate_cap_hit
        });
    }

    pub(super) fn first_preprocessing<const MIN: bool>(
        &mut self,
        state: &mut SearchState<MIN>,
        baseline: PackedCodeword,
        k: usize,
    ) {
        let r = N - k;
        let t = self.config.parity_prefix_len.unwrap_or(r / 2).min(r);
        let theta = self
            .config
            .preprocessing_threshold
            .unwrap_or(2 * (self.config.search_order + 3));
        let m = self.config.search_order + 1;
        let mask = prefix_mask_red(t);
        let b = mask_red(extract_slice(&baseline, k, r), mask);
        if t <= 64 {
            for i in 0..k {
                self.prefix_words[i] = self.basis.parity_basis_deltas[i][0] & mask[0];
            }
            #[cfg(test)]
            if self.reference.pre1 && (2..=4).contains(&m) {
                self.pre1_reference(state, baseline, k, m, b[0], theta);
                return;
            }
            match m {
                1 => self.pre1_order1(state, baseline, k, b[0], theta),
                2 => self.pre1_order2_vector(state, baseline, k, b[0], theta),
                3 => self.pre1_order3_vector(state, baseline, k, b[0], theta),
                4 => self.pre1_order4_vector(state, baseline, k, b[0], theta),
                _ => self.pre1_generic::<false, MIN>(state, baseline, k, m, b, theta),
            }
        } else {
            for i in 0..k {
                self.prefix_pairs[i] = mask_red(self.basis.parity_basis_deltas[i], mask);
            }
            self.pre1_generic::<true, MIN>(state, baseline, k, m, b, theta);
        }
    }

    // Count here so admission does not count the same pattern twice.
    #[inline(always)]
    fn pre1_reject<const MIN: bool>(
        state: &mut SearchState<MIN>,
        m: usize,
        q: usize,
        theta: usize,
    ) -> bool {
        Self::pre1_counts(state, 1, usize::from(m + q > theta));
        m + q > theta
    }

    #[inline(always)]
    fn pre1_counts<const MIN: bool>(
        state: &mut SearchState<MIN>,
        examined: usize,
        rejected: usize,
    ) {
        state.diagnostics.visited_steps += examined;
        state.diagnostics.enumerated_patterns += examined;
        state.diagnostics.preprocess1_examined += examined;
        state.diagnostics.preprocess1_rejects += rejected;
    }

    fn pre1_order1<const MIN: bool>(
        &mut self,
        state: &mut SearchState<MIN>,
        baseline: PackedCodeword,
        k: usize,
        b: u64,
        theta: usize,
    ) {
        for i0 in 0..k {
            if state.diagnostics.candidate_cap_hit {
                return;
            }
            if Self::pre1_reject(
                state,
                1,
                (b ^ self.prefix_words[i0]).count_ones() as usize,
                theta,
            ) {
                continue;
            }
            self.reserve_and_score(
                state,
                xor_delta(baseline, self.basis.encoded_basis_deltas[i0]),
                1,
                [i0 as u8, 0, 0, 0, 0, 0],
            );
        }
    }

    fn pre1_generic<const TWO_WORDS: bool, const MIN: bool>(
        &mut self,
        state: &mut SearchState<MIN>,
        baseline: PackedCodeword,
        k: usize,
        m: usize,
        b: PackedRedundancy,
        theta: usize,
    ) {
        enumerate(k, m, |positions| {
            if state.diagnostics.candidate_cap_hit {
                return false;
            }
            let mut prefix = b;
            for &p in &positions[..m] {
                if TWO_WORDS {
                    prefix = xor_red(prefix, self.prefix_pairs[p as usize]);
                } else {
                    prefix[0] ^= self.prefix_words[p as usize];
                }
            }
            if !Self::pre1_reject(state, m, popcount_red(prefix), theta) {
                let diff = self.diff_from_positions(baseline, &positions[..m]);
                self.reserve_and_score(state, diff, m as u8, positions);
            }
            !state.diagnostics.candidate_cap_hit
        });
    }
}
