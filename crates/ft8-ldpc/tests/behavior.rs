//! Hard slicing, early stop, CRC-gated preference and search counts.
use ft8_ldpc_core::fec::CascadedGeneratorMatrix;
fn get_codeword_bit(words: &[u64; 3], bit: usize) -> bool {
    words[bit / 64] & (1 << (bit % 64)) != 0
}
use ft8_ldpc::wu2007::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn noise(rng: &mut Rng) -> [f32; 174] {
    std::array::from_fn(|_| {
        let magnitude = (1 + (rng.next() >> 56)) as f32 / 32.;
        if rng.next() & 1 == 1 {
            -magnitude
        } else {
            magnitude
        }
    })
}

/// A codeword for a random payload with a valid CRC, for `c` cascaded bits.
fn codeword(rng: &mut Rng, c: usize) -> [u64; 3] {
    let payload: Vec<bool> = (0..77).map(|_| rng.next() & 1 == 1).collect();
    let crc = ft8_ldpc_core::crc::compute_crc14_from_bits(&payload);
    let mut message = [0u64; 2];
    for (i, &bit) in payload.iter().enumerate() {
        message[i / 64] |= (bit as u64) << (i % 64);
    }
    for j in 0..14 {
        message[(77 + j) / 64] |= (((crc >> (13 - j)) & 1) as u64) << ((77 + j) % 64);
    }
    let k = 91 - c;
    let input = [
        message[0],
        message[1] & if k > 64 { (1u64 << (k - 64)) - 1 } else { 0 },
    ];
    CascadedGeneratorMatrix::new(c)
        .unwrap()
        .encode_packed(input)
}

fn signed(codeword: &[u64; 3], mut amplitude: impl FnMut(usize) -> f32) -> [f32; 174] {
    std::array::from_fn(|n| {
        if get_codeword_bit(codeword, n) {
            -amplitude(n)
        } else {
            amplitude(n)
        }
    })
}

/// Only strictly negative values give a hard 1.
#[test]
fn hard_slicing_sends_signed_zeros_to_zero_and_subnormals_by_sign() {
    for zero in [0.0f32, -0.0] {
        let result = Decoder::new(Config::default())
            .unwrap()
            .decode(&[zero; 174]);
        assert_eq!(result.status, Status::Success);
        assert_eq!(result.codeword, [0; 3]);
        assert_eq!(result.hard_error_count, 0);
    }
    let mut rng = Rng(0x6);
    let word = codeword(&mut rng, 0);
    let tiny = f32::from_bits(1);
    let llr = signed(&word, |_| tiny);
    assert!(llr
        .iter()
        .zip(0..)
        .all(|(&x, n)| (x < 0.) == get_codeword_bit(&word, n)));
    let result = Decoder::new(Config::default()).unwrap().decode(&llr);
    assert_eq!(result.status, Status::Success);
    assert_eq!(result.codeword, word);
    assert_eq!(result.diagnostics.recoded_candidates, 1);
}

#[test]
fn the_zero_error_early_stop_visits_nothing_else() {
    let mut rng = Rng(0x88);
    let word = codeword(&mut rng, 0);
    let llr = signed(&word, |n| 3.3 + (n % 7) as f32 / 16.);
    let result = Decoder::new(Config::default()).unwrap().decode(&llr);
    assert_eq!(result.status, Status::Success);
    assert_eq!(result.codeword, word);
    let d = result.diagnostics;
    assert_eq!(
        (
            d.enumerated_patterns,
            d.recoded_candidates,
            d.crc_checked_candidates,
            d.materialized_candidates,
            d.crc_failures
        ),
        (1, 1, 1, 1, 0)
    );
    assert_eq!(
        (
            d.main_enumerated_patterns,
            d.main_scored_candidates,
            d.preprocess1_examined,
            d.preprocess1_scored_candidates
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(d.preprocess2_probe_masks, 0);

    // The same input with collection on runs the whole search.
    let mut config = Config {
        cascade_bits: 0,
        ..Default::default()
    };
    config.collect_candidates = true;
    config.search_order = 1;
    let listed = Decoder::new(config).unwrap().decode(&llr);
    assert_eq!(listed.codeword, word);
    assert!(listed.diagnostics.enumerated_patterns > 1);

    // A hard-decision codeword with a bad CRC stops only without the gate.
    // A genuine LDPC codeword whose message carries a wrong CRC.
    let mut message = [word[0], word[1] & ((1u64 << 27) - 1)];
    message[1] ^= 1 << (80 - 64);
    let broken = CascadedGeneratorMatrix::new(0)
        .unwrap()
        .encode_packed(message);
    assert_ne!(broken, word);
    let bad = signed(&broken, |_| 4.0);
    let mut config = Config {
        cascade_bits: 0,
        ..Default::default()
    };
    config.search_order = 1;
    let gated = Decoder::new(config.clone()).unwrap().decode(&bad);
    assert!(gated.diagnostics.enumerated_patterns > 1);
    config.require_crc = false;
    let open = Decoder::new(config).unwrap().decode(&bad);
    assert_eq!(open.diagnostics.enumerated_patterns, 1);
    assert_eq!(open.diagnostics.crc_failures, 1);
    assert!(!open.crc_valid);
    assert_eq!(open.status, Status::Success);
}

/// The CRC-passing candidate wins over a closer one that fails.
#[test]
fn a_crc_valid_candidate_beats_a_closer_failing_one() {
    let mut rng = Rng(0x12);
    let word = codeword(&mut rng, 0);
    let mut llr = signed(&word, |n| {
        if n < 91 {
            3.1 + rng.unit() * 2.3
        } else {
            0.01 + rng.unit() * 0.04
        }
    });
    for n in [3, 40] {
        llr[n] = -llr[n];
    }
    let mut config = Config {
        cascade_bits: 0,
        ..Default::default()
    };
    config.search_order = 2;
    config.enable_first_preprocessing = false;
    let result = Decoder::new(config).unwrap().decode(&llr);
    assert_eq!(result.status, Status::Success);
    assert!(result.crc_valid);
    assert_eq!(result.codeword, word);
    assert!(result.diagnostics.crc_failures > 0);
    assert!(result.diagnostics.best_any_llr_distance.unwrap() < result.llr_distance);
    assert!(!result.diagnostics.best_any_crc_valid);
}

/// Checks search pattern counts at k = 91 and k = 77 with a large cap.
#[test]
fn pattern_counts_match_the_expected_totals() {
    let mut rng = Rng(0x15);
    let llr = noise(&mut rng);
    let main = |w: usize| (1..=w).map(|m| choose(91, m)).sum::<usize>();
    for w in 0..=3 {
        let mut config = Config {
            cascade_bits: 0,
            ..Default::default()
        };
        config.search_order = w;
        config.max_recoded_candidates = 1 << 30;
        // This seeded input avoids early stopping, allowing the full pattern counts to be checked.
        let d = Decoder::new(config).unwrap().decode(&llr).diagnostics;
        assert_eq!(d.main_enumerated_patterns, main(w), "w {w}");
        assert_eq!(d.preprocess1_examined, choose(91, w + 1), "w {w}");
        assert_eq!(
            d.enumerated_patterns,
            1 + d.main_enumerated_patterns + d.preprocess1_examined
        );
    }
    assert_eq!(main(3), 125_671);
    assert_eq!(choose(91, 4), 2_672_670);
    let mut config = Config {
        cascade_bits: 14,
        ..Default::default()
    };
    config.max_recoded_candidates = 1 << 30;
    let d = Decoder::new(config).unwrap().decode(&llr).diagnostics;
    assert_eq!(d.main_enumerated_patterns, 76_153);
    assert_eq!(d.preprocess1_examined, 1_353_275);

    // At w = 4 the default cap is reached in main order 4 and nothing more is
    // visited: the refused pattern is number 174,329 of order 4.
    let mut config = Config {
        cascade_bits: 0,
        ..Default::default()
    };
    config.search_order = 4;
    let result = Decoder::new(config).unwrap().decode(&llr);
    assert_eq!(result.status, Status::CandidateCapHit);
    let d = result.diagnostics;
    assert_eq!(d.enumerated_patterns, 300_001);
    assert_eq!(d.recoded_candidates, 300_000);
    assert_eq!(d.preprocess1_examined, 0);
    assert_eq!(300_001 - (1 + main(3)), 174_329);
}

fn choose(n: usize, r: usize) -> usize {
    (0..r).fold(1, |acc, i| acc * (n - i) / (i + 1))
}
