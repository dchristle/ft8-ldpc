//! Selection oracles over the complete searched sequence, using seeded payloads
//! with CRC-14, this crate's encoder and BPSK AWGN (payload Eb/N0).
use super::*;
use crate::wu2007::scoring::test_inputs::SplitMix64;
use test_support::{llr_of, random_codeword};

fn awgn(rng: &mut SplitMix64, db: f64) -> [f32; N] {
    let word = random_codeword(rng, 0);
    let variance = 1. / (2. * (77. / 174.) * 10f64.powf(db / 10.));
    std::array::from_fn(|i| {
        let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        let v = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        let noise = (-2. * (1. - u).ln()).sqrt() * (std::f64::consts::TAU * v).cos();
        let signal = if get_codeword_bit(&word, i) { -1. } else { 1. };
        (2. * (signal + variance.sqrt() * noise) / variance) as f32
    })
}

fn search_counts(d: &Diagnostics) -> [usize; 17] {
    [
        d.rank,
        d.visited_steps,
        d.enumerated_patterns,
        d.recoded_candidates,
        d.scored_candidates,
        d.main_enumerated_patterns,
        d.main_scored_candidates,
        d.preprocess1_examined,
        d.preprocess1_rejects,
        d.preprocess1_scored_candidates,
        d.preprocess2_enumerated_patterns,
        d.preprocess2_scored_candidates,
        d.preprocess2_probe_masks,
        d.preprocess2_key_probes,
        usize::from(d.candidate_cap_hit),
        d.active_k,
        d.active_r,
    ]
}

// The oracle computes every scalar score in the recorded traversal order. It
// uses strict < rather than min_by (which can choose a different tied entry).
fn check_pair(
    best: &mut Decoder,
    min: &mut Decoder,
    llr: &[f32; N],
) -> (DecodeResult, DecodeResult) {
    let a = best.decode(llr);
    let b = min.decode(llr);
    assert_eq!(
        best.last_trace, min.last_trace,
        "admission order and patterns"
    );
    assert_eq!(best.basis.sys_to_natural, min.basis.sys_to_natural);
    assert_eq!(search_counts(&a.diagnostics), search_counts(&b.diagnostics));
    let mut first = None;
    for &(diff, _, _) in &best.last_trace {
        let score = best.scorer.distance_from_diff(&diff);
        assert_eq!(
            score.to_bits(),
            min.scorer.distance_from_diff(&diff).to_bits()
        );
        if first.is_none_or(|(_, distance)| score < distance) {
            first = Some((diff, score));
        }
    }
    let Some((diff, distance)) = first else {
        assert_eq!(b.status, Status::CandidateCapHit);
        assert_eq!(b.diagnostics.crc_checked_candidates, 0);
        assert_eq!(b.diagnostics.materialized_candidates, 0);
        return (a, b);
    };
    let natural_diff = inverse_permute(&diff, &best.basis.sys_to_natural);
    let mut hard = [0; 3];
    for (i, &x) in llr.iter().enumerate() {
        if x < 0. {
            set_codeword_bit(&mut hard, i);
        }
    }
    let expected = xor_delta(natural_diff, hard);
    assert_eq!(b.codeword, expected);
    assert_eq!(b.llr_distance.to_bits(), distance.to_bits());
    assert_eq!(b.payload, extract_prefix(&expected, 77));
    assert_eq!(b.message_with_crc, extract_prefix(&expected, 91));
    assert_eq!(b.hard_error_count, popcount_codeword(diff));
    assert_eq!(b.crc_valid, verify_crc14_packed_message(b.message_with_crc));
    assert_eq!(
        b.diagnostics.best_any_llr_distance.map(f32::to_bits),
        Some(distance.to_bits())
    );
    assert_eq!(b.diagnostics.best_any_crc_valid, b.crc_valid);
    let status = if b.diagnostics.candidate_cap_hit {
        Status::CandidateCapHit
    } else if b.crc_valid || !min.config.require_crc {
        Status::Success
    } else {
        Status::NoValidCrc
    };
    assert_eq!(b.status, status);
    if !min.config.collect_candidates {
        assert_eq!(b.diagnostics.crc_checked_candidates, 1);
        assert_eq!(b.diagnostics.materialized_candidates, 1);
        assert_eq!(b.diagnostics.crc_failures, usize::from(!b.crc_valid));
        assert!(b.candidates.is_empty());
    } else {
        // Collection retains its CRC-preferred list order, independently of
        // the selected winner, and checks every admitted candidate.
        assert_eq!(format!("{:?}", a.candidates), format!("{:?}", b.candidates));
        assert_eq!(b.diagnostics.crc_checked_candidates, best.last_trace.len());
        assert_eq!(b.diagnostics.materialized_candidates, best.last_trace.len());
    }
    (a, b)
}

fn decoders(config: Config) -> (Decoder, Decoder) {
    let mut min = config.clone();
    min.selection_policy = SelectionPolicy::MinMetricThenCrc;
    (Decoder::new(config).unwrap(), Decoder::new(min).unwrap())
}

#[test]
fn waterfall_selects_the_exact_first_minimum_and_never_substitutes_a_crc_winner() {
    let mut rng = SplitMix64(0x5345_4c45_4354_494f);
    let config = Config {
        search_order: 2,
        ..Config::default()
    };
    let (mut best, mut min) = decoders(config);
    let mut farther_valid = 0;
    let mut successes = [0; 5];
    for (snr, db) in [-2., 0., 2., 4., 6.].into_iter().enumerate() {
        for _ in 0..32 {
            let llr = awgn(&mut rng, db);
            let (a, b) = check_pair(&mut best, &mut min, &llr);
            successes[snr] += usize::from(b.status == Status::Success);
            if !b.crc_valid && a.crc_valid && b.llr_distance < a.llr_distance {
                farther_valid += 1;
                assert_eq!(b.status, Status::NoValidCrc);
                // Acceptance is independent: relaxing it preserves the winner.
                min.config.require_crc = false;
                let open = min.decode(&llr);
                assert_eq!(open.codeword, b.codeword);
                assert_eq!(open.status, Status::Success);
                min.config.require_crc = true;
            }
        }
    }
    assert!(
        farther_valid > 0,
        "no CRC-policy difference in seeded waterfall"
    );
    assert!(successes[0] < 16 && successes[4] > 28, "{successes:?}");
}

#[test]
fn policies_preserve_all_search_stages_caps_and_collection_contract() {
    let mut rng = SplitMix64(0x7365_6c65_6374_696f);
    let llr = awgn(&mut rng, 0.);
    for (order, first, second, prefix, tau) in [
        (0, false, false, 0, 0),
        (0, true, true, 65, 70),
        (1, true, true, 0, 0),
        (2, true, true, 64, 16),
        (3, true, true, 41, 16),
        (3, true, true, 41, 18),
        (3, true, true, 65, 65),
        (4, false, true, 0, 70),
        (5, true, false, 65, 0),
        (6, false, false, 0, 0),
    ] {
        for cascade in [0, 7, 14] {
            for cap in [0, 1, 7, 512] {
                for collect in [false, true] {
                    let config = Config {
                        cascade_bits: cascade,
                        search_order: order,
                        enable_first_preprocessing: first,
                        enable_second_preprocessing: second,
                        parity_prefix_len: Some(prefix),
                        second_preprocessing_prefix_len: Some(tau),
                        max_recoded_candidates: cap,
                        collect_candidates: collect,
                        max_list_candidates: if cap == 7 { 0 } else { 8 },
                        require_crc: cap % 2 == 0,
                        ..Config::default()
                    };
                    let (mut a, mut b) = decoders(config);
                    check_pair(&mut a, &mut b, &llr);
                }
            }
        }
    }
    // Reach the second-preprocessing walkers beyond a complete main search.
    for (order, tau) in [(0, 70), (1, 0), (2, 16), (3, 16), (3, 18), (3, 65)] {
        let (mut a, mut b) = decoders(Config {
            search_order: order,
            enable_second_preprocessing: true,
            second_preprocessing_prefix_len: Some(tau),
            max_recoded_candidates: 150_000,
            ..Config::default()
        });
        let (_, r) = check_pair(&mut a, &mut b, &llr);
        assert!(r.diagnostics.preprocess2_key_probes > 0);
    }
}

#[test]
fn strict_ties_infinite_metrics_and_baseline_crc_cache() {
    let mut rng = SplitMix64(0x7469_6573);
    let llr = awgn(&mut rng, -2.);
    // Equal magnitudes produce many exact ties; overflowing sums must still
    // select the first candidate even when every score is infinite.
    for input in [
        llr.map(|x| if x < 0. { -1. } else { 1. }),
        llr.map(|x| if x < 0. { -f32::MAX } else { f32::MAX }),
        [0.; N],
    ] {
        let (mut a, mut b) = decoders(Config {
            search_order: 2,
            enable_first_preprocessing: false,
            ..Config::default()
        });
        check_pair(&mut a, &mut b, &input);
    }
    // Explicit repeated minima, including signed zero, test first-wins without
    // depending on the tie order of the candidate list.
    let decoder = Decoder::default();
    for distances in [[5., 3., 3., 4.], [f32::INFINITY; 4], [0., -0., 0., 1.]] {
        let mut state: SearchState<true> = SearchState {
            received: [0; 3],
            minimum: None,
            best_any: None,
            best_valid: None,
            candidates: Vec::new(),
            diagnostics: Diagnostics::default(),
            trace: Vec::new(),
        };
        let mut expected = None;
        for (i, distance) in distances.into_iter().enumerate() {
            let diff = random_codeword(&mut rng, 0);
            decoder.select(&mut state, diff, 0, [0; 6], distance);
            if expected.is_none_or(|(_, d)| distance < d) {
                expected = Some((diff, distance));
            }
            if i == 0 {
                assert_eq!(state.minimum.unwrap().diff, diff);
            }
        }
        let (diff, distance) = expected.unwrap();
        assert_eq!(state.minimum.unwrap().diff, diff);
        assert_eq!(
            state.minimum.unwrap().distance.to_bits(),
            distance.to_bits()
        );
    }
    let word = random_codeword(&mut rng, 0);
    let mut wrong_crc = extract_prefix(&word, 91);
    wrong_crc[1] ^= 1 << (80 - 64);
    let broken = CascadedGeneratorMatrix::new(0)
        .unwrap()
        .encode_packed(wrong_crc);
    for require_crc in [false, true] {
        let config = Config {
            search_order: 1,
            enable_first_preprocessing: false,
            require_crc,
            ..Config::default()
        };
        let (mut a, mut b) = decoders(config);
        for codeword in [word, broken] {
            let (_, result) = check_pair(&mut a, &mut b, &llr_of(&codeword, 4.));
            assert_eq!(result.codeword, codeword);
            assert_eq!(
                result.diagnostics.enumerated_patterns,
                if require_crc && codeword == broken {
                    92
                } else {
                    1
                }
            );
        }
    }
}
