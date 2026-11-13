//! Tests for PB-OSD probability, distance, and TEP components.

#[cfg(test)]
mod tests {
    use super::super::{
        distance::{utils::*, DistanceCalculator},
        probability::{q_function, q_function_4x, ProbabilityCalculator},
        tep::TestErrorPattern,
        tep_iterator::{TepConfig, TepIterator},
    };
    use ft8_ldpc_core::protocol::N;

    /// Test that the probability calculator produces consistent results
    #[test]
    fn test_probability_calculator_consistency() {
        // Create a simple test setup
        let alpha_mrb = [1.0f32; N]; // All reliability values = 1.0
        let c0_mrb = [false; N]; // All baseline bits = false

        let calc = ProbabilityCalculator {
            log_p: [-1.0; N],         // log(P(i)) = -1.0 for all bits
            log_1_minus_p: [-0.1; N], // log(1-P(i)) = -0.1 for all bits
            sum_log_1_minus_p: -0.1 * N as f32,
            e_p_alpha: 0.5,
            parity_p_avg: 0.1,
            alpha_mrb,
            c0_mrb,
            baseline_codeword_bool: [false; N], // Simple baseline for testing
            k: 91,
            nmk: 83,
            log_two_pow_nmk: 83.0 * 2.0_f32.ln(),
            d_min_w: 10.0,
        };

        // Test with a simple TEP
        let tep = TestErrorPattern::with_positions(&[0, 5, 10]);
        let log_pe = calc.calculate_log_pe(&tep);

        // Should be sum_log_1_minus_p + 3 * (log_p - log_1_minus_p)
        let expected = calc.sum_log_1_minus_p + 3.0 * (-1.0 - (-0.1));
        assert!(
            (log_pe - expected).abs() < 1e-6,
            "log_pe calculation mismatch: {} vs {}",
            log_pe,
            expected
        );
    }

    /// Test that distance calculations are consistent across different implementations
    #[test]
    fn test_distance_calculator_consistency() {
        let alpha_mrb = [1.0f32; N]; // All weights = 1.0 for simplicity
        let c0_mrb_bitpacked = [0u64; 3]; // All baseline bits = 0

        let calc = DistanceCalculator::new(alpha_mrb, c0_mrb_bitpacked);

        // Create a test codeword with a few bits set
        let mut test_cw_bool = [false; N];
        test_cw_bool[0] = true;
        test_cw_bool[63] = true;
        test_cw_bool[64] = true;
        test_cw_bool[173] = true;

        let test_cw_packed = pack_bool_array(&test_cw_bool);
        let baseline_bool = [false; N];

        // Test different implementations give same result
        let distance_bool = calc.calculate_weighted_hamming(&test_cw_bool, &baseline_bool);
        let distance_packed = calc.calculate_weighted_hamming_fast(&test_cw_packed);
        let distance_parallel = calc.calculate_weighted_hamming_parallel(&test_cw_packed);

        assert!((distance_bool - distance_packed).abs() < 1e-6);
        assert!((distance_bool - distance_parallel).abs() < 1e-6);

        // With unit weights, distance should equal number of differing bits
        assert!((distance_bool - 4.0).abs() < 1e-6);
    }

    /// Test that the TEP iterator generates valid patterns
    #[test]
    fn test_tep_iterator_basic_functionality() {
        let config = TepConfig {
            max_order: 2,
            k: 10,
        };
        let mut iterator = TepIterator::new(config);

        // Should start with empty pattern
        let first_tep = iterator.next().unwrap();
        assert_eq!(first_tep.num_positions(), 0);

        // Collect all TEPs up to order 1
        let mut all_teps = Vec::new();
        for tep in iterator {
            all_teps.push(tep);
            if all_teps.len() > 20 {
                // Safety limit for test
                break;
            }
        }

        // Count single-position patterns
        let single_position_count = all_teps
            .iter()
            .filter(|tep| tep.num_positions() == 1)
            .count();

        assert!(
            single_position_count > 0,
            "Should generate at least some single-position patterns, got {}",
            single_position_count
        );

        // Verify all generated TEPs are valid
        for tep in &all_teps {
            assert!(tep.num_positions() <= config.max_order as u8);
            for i in 0..tep.num_positions() {
                assert!(tep.positions()[i as usize] < config.k);
            }
        }
    }

    /// Test Q-function implementations
    #[test]
    fn test_q_function_implementations() {
        let test_values = [0.0, 1.0, -1.0, 2.5, -2.5];

        for &x in &test_values {
            let scalar_result = q_function(x);
            let vector_input = [x, x, x, x];
            let vector_result = q_function_4x(&vector_input);

            // All elements of vector result should match scalar
            for &v in &vector_result {
                assert!(
                    (scalar_result - v).abs() < 1e-6,
                    "Q-function mismatch for x={}: scalar={}, vector={}",
                    x,
                    scalar_result,
                    v
                );
            }

            // Sanity check: Q(0) should be approximately 0.5
            if x == 0.0 {
                assert!(
                    (scalar_result - 0.5).abs() < 0.1,
                    "Q(0) should be near 0.5, got {}",
                    scalar_result
                );
            }
        }
    }

    /// Test bit packing/unpacking utilities
    #[test]
    fn test_bit_packing_utilities() {
        // Test various bit patterns
        let test_patterns = [
            vec![0, 63, 64, 127, 128, 173], // Edge cases
            vec![1, 2, 3],                  // Simple pattern
            vec![],                         // Empty pattern
        ];

        for positions in test_patterns {
            let mut original = [false; N];
            for &pos in &positions {
                if pos < N {
                    original[pos] = true;
                }
            }

            let packed = pack_bool_array(&original);
            let unpacked = unpack_to_bool_array(&packed);

            assert_eq!(
                original, unpacked,
                "Pack/unpack mismatch for positions: {:?}",
                positions
            );
        }
    }

    /// Test TEP creation and manipulation
    #[test]
    fn test_tep_creation() {
        // Test creating TEPs with different position sets
        let positions_sets = [
            vec![],
            vec![0],
            vec![0, 5, 10],
            vec![1, 3, 7, 15, 31, 63, 87, 90], // Max 8 positions
        ];

        for positions in positions_sets {
            let tep = TestErrorPattern::with_positions(&positions);
            assert_eq!(tep.num_positions() as usize, positions.len());

            // Verify positions are stored correctly
            for (i, &expected_pos) in positions.iter().enumerate() {
                assert_eq!(tep.positions()[i], expected_pos);
            }

            // Test conversion to bit array
            let bit_array = tep.to_bit_array();
            for (i, &bit) in bit_array.iter().enumerate() {
                let should_be_set = positions.contains(&(i as u8));
                assert_eq!(
                    bit == 1,
                    should_be_set,
                    "Bit array mismatch at position {}",
                    i
                );
            }
        }
    }

    /// Integration test: verify modular components work together
    #[test]
    fn test_modular_integration() {
        // Create test components
        let alpha_mrb = [1.0f32; N];
        let c0_mrb = [false; N];
        let c0_mrb_bitpacked = [0u64; 3];

        let prob_calc = ProbabilityCalculator {
            log_p: [-1.0; N],
            log_1_minus_p: [-0.1; N],
            sum_log_1_minus_p: -0.1 * N as f32,
            e_p_alpha: 0.5,
            parity_p_avg: 0.1,
            alpha_mrb,
            c0_mrb,
            baseline_codeword_bool: [false; N], // Simple baseline for testing
            k: 10,
            nmk: 164,
            log_two_pow_nmk: 164.0 * 2.0_f32.ln(),
            d_min_w: 10.0,
        };

        let dist_calc = DistanceCalculator::new(alpha_mrb, c0_mrb_bitpacked);

        let config = TepConfig {
            max_order: 2,
            k: 10,
        };
        let mut iterator = TepIterator::new(config);
        iterator.set_probability_calculator(prob_calc.clone());

        // Process a few TEPs and verify calculations
        let mut processed_count = 0;
        for tep in iterator {
            if processed_count >= 5 {
                break;
            } // Limit for test performance

            // Calculate probability
            let log_prob = prob_calc.calculate_log_pe(&tep);
            assert!(log_prob.is_finite(), "Invalid log probability");

            // Calculate DEBW distance
            let debw_distance = dist_calc.calculate_debw(&tep);
            assert!(debw_distance >= 0.0, "DEBW distance should be non-negative");
            assert!(debw_distance.is_finite(), "DEBW distance should be finite");

            processed_count += 1;
        }

        assert!(
            processed_count > 0,
            "Should have processed at least one TEP"
        );
    }
}
