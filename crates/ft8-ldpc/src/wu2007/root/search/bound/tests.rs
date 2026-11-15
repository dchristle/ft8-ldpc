use super::*;
use crate::wu2007::root::test_support::synthetic;
use crate::wu2007::root::*;
use crate::wu2007::scoring::test_inputs::{alpha_vectors, SplitMix64};

#[test]
fn bounds_do_not_exceed_the_actual_binary32_scorer() {
    let mut inputs = alpha_vectors();
    inputs.extend([
        [f32::from_bits(1); N],
        [f32::MIN_POSITIVE; N],
        [f32::MAX; N],
        std::array::from_fn(|i| [0.0, f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX][i % 4]),
        std::array::from_fn(|i| [1.0, 1.0f32.next_up(), 1.0f32.next_down()][i % 3]),
    ]);
    let mut rng = SplitMix64(0xb0_0d_0001);
    for _ in 0..24 {
        inputs.push(std::array::from_fn(|_| {
            f32::from_bits((rng.next() as u32) % 0x7f80_0000)
        }));
    }
    for alpha in inputs {
        let scorer = PackedWeightedHammingScorer::new(&alpha, [0; 3]);
        for k in [4, 9, 17, 65, 77, 84, 91] {
            for m in [3, 4] {
                let bound = InformationBound::new(&alpha[..k], m, 1.0).unwrap();
                // Exhaust small bases; also cross nibble, byte and word boundaries.
                let mut check = |positions: [u8; 6]| {
                    let mut diff = [0; 3];
                    for &i in &positions[..m] {
                        set_codeword_bit(&mut diff, i as usize);
                    }
                    let lower = bound.lower(positions[0] as usize, positions[1] as usize);
                    assert!(
                        !bound.rejects(positions[0] as usize, positions[1] as usize)
                            || scorer.distance_from_diff(&diff) >= 1.0
                    );
                    assert!(
                        lower <= scorer.distance_from_diff(&diff) as f64,
                        "k={k} m={m} positions={positions:?} lower={lower}"
                    );
                    for j in k..N {
                        if rng.next() & 1 != 0 {
                            set_codeword_bit(&mut diff, j);
                        }
                    }
                    assert!(lower <= scorer.distance_from_diff(&diff) as f64);
                    true
                };
                if k <= 17 {
                    super::super::enumerate(k, m, &mut check);
                } else {
                    let mut positions = [0; 6];
                    for start in 0..k - m + 1 {
                        for (j, p) in positions[..m].iter_mut().enumerate() {
                            *p = (start + j) as u8;
                        }
                        check(positions);
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_bounds_and_pattern_counts() {
    for distance in [f32::INFINITY, f32::NAN, -1.0] {
        assert!(InformationBound::new(&[1.0; 8], 3, distance).is_none());
    }
    for a in [f32::INFINITY, f32::NAN, -1.0] {
        assert!(InformationBound::new(&[a; 8], 4, 1.0).is_none());
    }
    assert!(InformationBound::new(&[1.0; 3], 4, 1.0).is_none());
    assert_eq!(pattern_count(91, 3), Some(121_485));
    assert_eq!(pattern_count(91, 4), Some(2_672_670));
    assert_eq!(pattern_count(usize::MAX, 4), None);
    assert_eq!(pattern_count(3, 4), Some(0));
    let bound = InformationBound::new(&[0.0; 8], 4, 0.0).unwrap();
    assert!(bound.rejects(0, 1), "ties cannot replace the incumbent");
}

fn state() -> SearchState<true> {
    SearchState {
        received: [0; 3],
        crc: crc::CandidateCrc::default(),
        minimum: None,
        best_any: None,
        best_valid: None,
        candidates: Vec::new(),
        diagnostics: Diagnostics::default(),
        trace: Vec::new(),
    }
}

#[test]
fn pruning_matches_scalar_traversal_at_budget_and_prefix_boundaries() {
    let mut saw_skip = false;
    let mut saw_fallback = false;
    let mut saw_cap = false;
    let mut saw_mixed = false;
    for m in [3, 4] {
        for k in [5, 9, 12, 19] {
            let total = pattern_count(k, m).unwrap();
            for prefix in [0, 7, 41, 64, 65] {
                for cap in [
                    1,
                    2,
                    3,
                    4,
                    7,
                    8,
                    9,
                    15,
                    16,
                    17,
                    total,
                    total + 1,
                    usize::MAX,
                ] {
                    for second_stage in [false, true] {
                        let config = Config {
                            search_order: m - 1,
                            selection_policy: SelectionPolicy::MinMetricThenCrc,
                            parity_prefix_len: Some(prefix),
                            preprocessing_threshold: Some(m + prefix / 2),
                            max_recoded_candidates: Some(cap),
                            enable_second_preprocessing: second_stage,
                            ..Config::default()
                        };
                        let mut s = synthetic(0xb0_0d_0002, k, 512, config);
                        let alpha =
                            std::array::from_fn(
                                |i| if i < k { [0.125, 1.0, 8.0][i % 3] } else { 1.0 },
                            );
                        s.baseline = [0; 3];
                        set_codeword_bit(&mut s.baseline, k);
                        set_codeword_bit(&mut s.baseline, k + 1);
                        s.decoder.scorer.update(&alpha, [0; 3]);
                        let mut reference = s.decoder.clone();
                        reference.reference.pre1_bound = true;
                        let (mut a, mut b) = (state(), state());
                        s.decoder.admit(&mut a, s.baseline, 0, [0; 6]);
                        reference.admit(&mut b, s.baseline, 0, [0; 6]);
                        s.decoder.first_preprocessing(&mut a, s.baseline, k, &alpha);
                        reference.first_preprocessing(&mut b, s.baseline, k, &alpha);
                        s.decoder.flush_batch(&mut a);
                        reference.flush_batch(&mut b);
                        assert_eq!(a.minimum.unwrap().diff, b.minimum.unwrap().diff);
                        assert_eq!(
                            a.minimum.unwrap().distance.to_bits(),
                            b.minimum.unwrap().distance.to_bits()
                        );
                        assert!(a.trace == b.trace, "scored order and patterns");
                        let (a, b) = (&a.diagnostics, &b.diagnostics);
                        assert_eq!(test_support::counters(a), test_support::counters(b));
                        assert_eq!(
                            a.preprocess1_metric_pruned_patterns,
                            b.preprocess1_metric_pruned_patterns
                        );
                        assert!(a.recoded_candidates <= cap);
                        if prefix > 64 {
                            assert_eq!(a.preprocess1_metric_pruned_patterns, 0);
                            saw_fallback = true;
                        } else {
                            saw_skip |= a.preprocess1_metric_pruned_patterns > 0;
                        }
                        if !a.candidate_cap_hit {
                            assert_eq!(
                                a.preprocess1_examined + a.preprocess1_metric_pruned_patterns,
                                total
                            );
                        }
                        saw_cap |= a.candidate_cap_hit;
                        saw_mixed |=
                            a.scored_candidates > 1 && a.preprocess1_metric_pruned_patterns > 0;
                    }
                }
            }
        }
    }
    assert!(saw_skip && saw_fallback && saw_cap && saw_mixed);
}

#[test]
fn rearranged_threshold_is_safe_at_adjacent_float_boundaries() {
    let mut inputs = alpha_vectors();
    inputs.push(std::array::from_fn(|i| {
        [f32::from_bits(1), f32::MAX, 0.0][i % 3]
    }));
    for alpha in inputs {
        let scorer = PackedWeightedHammingScorer::new(&alpha, [0; 3]);
        for m in [3, 4] {
            let k = 12;
            let mut minima = [[f32::INFINITY; 12]; 12];
            super::super::enumerate(k, m, |positions| {
                let mut diff = [0; 3];
                for &i in &positions[..m] {
                    set_codeword_bit(&mut diff, i as usize);
                }
                let entry = &mut minima[positions[0] as usize][positions[1] as usize];
                *entry = entry.min(scorer.distance_from_diff(&diff));
                true
            });
            for i0 in 0..k - m + 1 {
                for i1 in i0 + 1..k - m + 2 {
                    let minimum = minima[i0][i1];
                    for distance in [minimum.next_down().max(0.0), minimum, minimum.next_up()] {
                        if let Some(bound) = InformationBound::new(&alpha[..k], m, distance) {
                            assert!(!bound.rejects(i0, i1) || minimum >= distance);
                            if !bound.can_prune() {
                                for a in 0..k - m + 1 {
                                    for b in a + 1..k - m + 2 {
                                        assert!(!bound.rejects(a, b));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn zero_ties_and_overflow_keep_the_first_candidate() {
    for m in [3, 4] {
        for weight in [0.0, f32::from_bits(1), f32::MIN_POSITIVE, 1.0, f32::MAX] {
            let config = Config {
                search_order: m - 1,
                selection_policy: SelectionPolicy::MinMetricThenCrc,
                parity_prefix_len: Some(0),
                max_recoded_candidates: None,
                ..Config::default()
            };
            let mut s = synthetic(0xb0_0d_0003, 12, 512, config);
            let alpha = [weight; N];
            s.decoder.scorer.update(&alpha, [0; 3]);
            s.baseline = [0; 3];
            set_codeword_bit(&mut s.baseline, 12);
            set_codeword_bit(&mut s.baseline, 13);
            let mut state = state();
            s.decoder.admit(&mut state, s.baseline, 0, [0; 6]);
            let first = state.minimum.unwrap();
            s.decoder
                .first_preprocessing(&mut state, s.baseline, 12, &alpha);
            s.decoder.flush_batch(&mut state);
            assert_eq!(state.minimum.unwrap().diff, first.diff);
            assert_eq!(
                state.minimum.unwrap().distance.to_bits(),
                first.distance.to_bits()
            );
            assert!(!state.diagnostics.candidate_cap_hit);
            if weight == 0.0 {
                assert_eq!(state.diagnostics.preprocess1_examined, 0);
                assert_eq!(
                    state.diagnostics.preprocess1_metric_pruned_patterns,
                    pattern_count(12, m).unwrap()
                );
            } else if weight == f32::MAX {
                assert_eq!(state.diagnostics.preprocess1_metric_pruned_patterns, 0);
            }
        }
    }
}

#[test]
fn pruned_branches_free_budget_for_a_later_winner() {
    for m in [3, 4] {
        let k = m + 2;
        let config = Config {
            search_order: m - 1,
            selection_policy: SelectionPolicy::MinMetricThenCrc,
            parity_prefix_len: Some(0),
            max_recoded_candidates: Some(2),
            ..Config::default()
        };
        let mut s = synthetic(0xb0_0d_0004, k, 0, config);
        let mut alpha = [2.0; N];
        alpha[..2].fill(8.0);
        alpha[2..k].fill(0.125);
        s.decoder.scorer.update(&alpha, [0; 3]);
        s.baseline = [0; 3];
        set_codeword_bit(&mut s.baseline, k);
        // The final all-cheap pattern cancels the baseline's parity mismatch.
        set_codeword_bit(&mut s.decoder.basis.encoded_basis_deltas[2], k);
        let mut old = s.decoder.clone();
        old.reference.pre1 = true;
        let (mut a, mut b) = (state(), state());
        s.decoder.admit(&mut a, s.baseline, 0, [0; 6]);
        old.admit(&mut b, s.baseline, 0, [0; 6]);
        s.decoder.first_preprocessing(&mut a, s.baseline, k, &alpha);
        old.first_preprocessing(&mut b, s.baseline, k, &alpha);
        s.decoder.flush_batch(&mut a);
        old.flush_batch(&mut b);
        assert_eq!(a.minimum.unwrap().distance, m as f32 * 0.125);
        assert_eq!(b.minimum.unwrap().distance, 2.0);
        assert_eq!(a.diagnostics.recoded_candidates, 2);
        assert_eq!(b.diagnostics.recoded_candidates, 2);
        assert!(
            !a.diagnostics.candidate_cap_hit,
            "reaching the cap is not a refused admission"
        );
        assert!(b.diagnostics.candidate_cap_hit);
        assert_eq!(
            a.diagnostics.preprocess1_metric_pruned_patterns,
            pattern_count(k, m).unwrap() - 1
        );
    }
}
