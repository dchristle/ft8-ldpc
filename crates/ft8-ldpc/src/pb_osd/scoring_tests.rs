//! Tests for the packed weighted-Hamming lookup tables used by the PB-OSD search.

#[cfg(test)]
mod tests {
    use crate::pb_osd::distance::{DistanceCalculator, OptimizedWeightedHammingTables};
    use crate::pb_osd::tep::TestErrorPattern;
    use ft8_ldpc_core::protocol::N;

    fn bit_at(words: &[u64; 3], pos: usize) -> bool {
        ((words[pos / 64] >> (pos % 64)) & 1) != 0
    }

    fn scalar_boolean_weighted_hamming(cw: &[u64; 3], c0: &[u64; 3], alpha: &[f32; N]) -> f32 {
        let mut total = 0.0f32;
        for pos in 0..N {
            if bit_at(cw, pos) != bit_at(c0, pos) {
                total += alpha[pos];
            }
        }
        total
    }

    fn explicit_debw(tep: &TestErrorPattern, alpha: &[f32; N]) -> f32 {
        tep.positions()
            .iter()
            .take(tep.num_positions() as usize)
            .map(|&pos| alpha[pos as usize])
            .sum()
    }

    #[test]
    fn bitpacked_weighted_hamming_matches_boolean_scalar_reference() {
        let mut alpha = [0.0f32; N];
        for (i, value) in alpha.iter_mut().enumerate() {
            *value = 0.125 + (i as f32 * 0.03125);
        }
        let c0 = [
            0xA55A_F00D_1234_5678,
            0x0F0F_F0F0_9876_5432,
            0x0000_1234_5678_9ABC,
        ];
        let scorer = OptimizedWeightedHammingTables::new(&alpha, c0);

        let codewords = [
            [0, 0, 0],
            [u64::MAX, 0, 0],
            [
                0x0123_4567_89AB_CDEF,
                0xFEDC_BA98_7654_3210,
                0x0000_3FFF_FFFF_FFFF,
            ],
            c0,
        ];

        for cw in codewords {
            let expected = scalar_boolean_weighted_hamming(&cw, &c0, &alpha);
            let actual = scorer.calculate_weighted_hamming_f32(&cw);
            assert!(
                (actual - expected).abs() <= 1e-4,
                "cw={cw:?} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn debw_matches_explicit_per_position_weight_sum() {
        let mut alpha = [0.0f32; N];
        for (i, value) in alpha.iter_mut().enumerate() {
            *value = 0.2 + ((i * 37) % 29) as f32 * 0.0625;
        }
        let c0 = [
            0x0101_2020_4040_8080,
            0x1122_3344_5566_7788,
            0x0000_1357_9BDF_2468,
        ];
        let calc = DistanceCalculator::new(alpha, c0);
        let teps = [
            TestErrorPattern::new(),
            TestErrorPattern::with_positions(&[0]),
            TestErrorPattern::with_positions(&[0, 1, 63, 64, 127, 128, 173]),
            TestErrorPattern::with_positions(&[7, 19, 31, 59, 83, 109, 149, 167]),
        ];

        for tep in teps {
            let expected = explicit_debw(&tep, &calc.alpha_mrb);
            let actual = calc.calculate_debw(&tep);
            assert!(
                (actual - expected).abs() <= f32::EPSILON,
                "tep={tep:?} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn diff_scoring_matches_direct_scoring() {
        let mut alpha = [0.0f32; N];
        for (i, value) in alpha.iter_mut().enumerate() {
            *value = ((i % 11) as f32 + 1.0) * 0.2;
        }
        let c0 = [
            0x1111_2222_3333_4444,
            0x5555_6666_7777_8888,
            0x0000_9999_AAAA_BBBB,
        ];
        let scorer = OptimizedWeightedHammingTables::new(&alpha, c0);
        let cw = [
            0xFFFF_0000_FFFF_0000,
            0x1234_5678_9ABC_DEF0,
            0x0000_3C3C_C3C3_0F0F,
        ];

        let diff = scorer.make_diff(&cw);
        assert_eq!(
            scorer.distance_from_diff(&diff),
            scorer.calculate_weighted_hamming_f32(&cw)
        );
    }

    #[test]
    fn update_replaces_weights_and_baseline() {
        let alpha_a = [1.0f32; N];
        let c0_a = [0u64; 3];
        let mut scorer = OptimizedWeightedHammingTables::new(&alpha_a, c0_a);
        let cw = [0b1011, 0, 0];
        assert_eq!(scorer.calculate_weighted_hamming_f32(&cw), 3.0);

        let mut alpha_b = [0.5f32; N];
        alpha_b[1] = 3.0;
        let c0_b = [0b0010, 0, 0];
        scorer.update(&alpha_b, c0_b);

        let expected = scalar_boolean_weighted_hamming(&cw, &c0_b, &alpha_b);
        assert_eq!(scorer.calculate_weighted_hamming_f32(&cw), expected);
    }
}
