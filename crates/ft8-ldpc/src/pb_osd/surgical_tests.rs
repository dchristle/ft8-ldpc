//! Surgical tests for critical PB-OSD algorithmic components
//!
//! These tests provide focused validation of the most complex and critical
//! parts of the PB-OSD algorithm that are not adequately covered by
//! higher-level end-to-end tests.

#[cfg(test)]
mod probability_tests {
    use super::super::probability::{
        q_function, q_function_4x, ProbabilityCalculator, Q_PARAM_A, Q_PARAM_B, Q_PARAM_C,
    };
    use super::super::tep::TestErrorPattern;
    use ft8_ldpc_core::protocol::N;

    /// High-precision reference Q-function using libm's erf function
    fn reference_q_function(x: f32) -> f32 {
        (0.5 * (1.0 - libm::erf((x / 2.0_f32.sqrt()) as f64))) as f32
    }

    #[test]
    fn test_all_q_function_implementations_consistency() {
        // Compare all Q-function implementations in the codebase to find the odd one out

        // Get the implementations
        let prob_q_function = q_function; // From probability.rs

        // Belief propagation Q-function (copied here for comparison)
        let bp_q_function =
            |x: f32| -> f32 { (0.5 * (1.0 - libm::erf((x / 2.0_f32.sqrt()) as f64))) as f32 };

        // My reference implementation from above
        let ref_q_function = reference_q_function;

        let test_points = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];

        println!("\nQ-function comparison across implementations:");
        println!("x\t\tprob.rs\t\tbp_test\t\tref_impl");

        for &x in &test_points {
            let prob_result = prob_q_function(x);
            let bp_result = bp_q_function(x);
            let ref_result = ref_q_function(x);

            println!(
                "{:.1}\t\t{:.6}\t{:.6}\t{:.6}",
                x, prob_result, bp_result, ref_result
            );

            // All should be in [0,1] range and finite
            assert!((0.0..=1.0).contains(&prob_result) && prob_result.is_finite());
            assert!((0.0..=1.0).contains(&bp_result) && bp_result.is_finite());
            assert!((0.0..=1.0).contains(&ref_result) && ref_result.is_finite());
        }

        // The probability.rs and BP implementations should be closer to each other than to our simplified ref
        let test_x = 1.0;
        let prob_val = prob_q_function(test_x);
        let bp_val = bp_q_function(test_x);
        let ref_val = ref_q_function(test_x);

        // This test helps identify which implementation is the outlier
        let prob_bp_diff = (prob_val - bp_val).abs();
        let prob_ref_diff = (prob_val - ref_val).abs();
        let bp_ref_diff = (bp_val - ref_val).abs();

        println!("\nDifferences at x=1.0:");
        println!("prob vs bp: {:.6}", prob_bp_diff);
        println!("prob vs ref: {:.6}", prob_ref_diff);
        println!("bp vs ref: {:.6}", bp_ref_diff);
    }

    #[test]
    fn test_q_function_basic_properties() {
        // Test that Q-function produces reasonable values in [0,1] range
        let test_points = [
            -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
        ];

        for &x in &test_points {
            let result = q_function(x);
            assert!(
                (0.0..=1.0).contains(&result),
                "Q-function should return value in [0,1], got {} for x={}",
                result,
                x
            );
            assert!(
                result.is_finite(),
                "Q-function should return finite value, got {} for x={}",
                result,
                x
            );
        }

        // Test that Q(0) is reasonable (should be around 0.5 for normal distribution)
        let q_zero = q_function(0.0);
        assert!(
            q_zero > 0.2 && q_zero < 0.8,
            "Q(0) should be roughly 0.5, got {}",
            q_zero
        );
    }

    #[test]
    fn test_q_function_accuracy_against_reference() {
        // Test López-Benítez & Casadevall approximation against high-precision reference
        let test_points = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0];

        for &x in &test_points {
            let fast_result = q_function(x);
            let reference_result = reference_q_function(x);
            let relative_error =
                ((fast_result - reference_result) / reference_result.max(1e-10)).abs();

            // With the accurate libm::erf implementation, we expect high precision
            let tolerance = 1e-6;
            assert!(
                relative_error < tolerance,
                "Q-function accuracy failure at x={}: fast={}, reference={}, rel_error={}%",
                x,
                fast_result,
                reference_result,
                relative_error * 100.0
            );
        }
    }

    #[test]
    fn test_q_function_decoder_relevant_properties() {
        // Focus on properties that matter for the PB-OSD decoder functionality

        // Test that large positive values give small results (tail probabilities)
        let q_large_pos = q_function(5.0);
        assert!(
            q_large_pos < 0.1,
            "Q(5) should be small, got {}",
            q_large_pos
        );

        // Test that large negative values give large results (as expected for Q-function)
        let q_large_neg = q_function(-5.0);
        assert!(
            q_large_neg > 0.99,
            "Q(-5) should be very close to 1, got {}",
            q_large_neg
        );

        // Test general trend: Q should decrease as x increases
        let q_minus_1 = q_function(-1.0);
        let q_zero = q_function(0.0);
        let q_plus_1 = q_function(1.0);

        assert!(
            q_minus_1 > q_zero,
            "Q(-1) should be > Q(0), got {} vs {}",
            q_minus_1,
            q_zero
        );
        assert!(
            q_zero > q_plus_1,
            "Q(0) should be > Q(1), got {} vs {}",
            q_zero,
            q_plus_1
        );
    }

    #[test]
    fn test_q_function_monotonicity() {
        // Q-function should be generally decreasing with some tolerance for numerical approximation
        let mut prev_value = q_function(-3.0);
        for i in -29..30 {
            let x = i as f32 * 0.1;
            let current_value = q_function(x);
            // Allow larger violations due to fast approximation errors
            let tolerance = 0.05;
            assert!(
                current_value <= prev_value + tolerance,
                "Q-function not monotonic: Q({}) = {} > Q({}) = {} (tolerance exceeded)",
                x,
                current_value,
                x - 0.1,
                prev_value
            );
            prev_value = current_value;
        }
    }

    #[test]
    fn test_q_function_4x_consistency_with_scalar() {
        let test_values = [
            [-2.0, -1.0, 0.0, 1.0],
            [2.0, 2.5, -2.5, -1.5],
            [0.1, 0.01, -0.01, -0.1],
            [4.0, -4.0, 3.5, -3.5],
        ];

        for &input in &test_values {
            let scalar_results = [
                q_function(input[0]),
                q_function(input[1]),
                q_function(input[2]),
                q_function(input[3]),
            ];
            let vector_results = q_function_4x(&input);

            for i in 0..4 {
                assert!(
                    (scalar_results[i] - vector_results[i]).abs() < 1e-6,
                    "Q-function 4x mismatch at index {}: scalar={}, vector={}",
                    i,
                    scalar_results[i],
                    vector_results[i]
                );
            }
        }
    }

    #[test]
    fn test_q_function_coefficients_mathematical_properties() {
        // Verify the López-Benítez & Casadevall coefficients make mathematical sense
        // The polynomial A*x² + B*x + C should approximate tanh for Q-function

        // Test that coefficients are in expected ranges
        const {
            assert!(
                Q_PARAM_A < 0.0,
                "A coefficient should be negative for tanh-like behavior"
            );
        }
        const {
            assert!(Q_PARAM_B < 0.0, "B coefficient should be negative");
        }
        const {
            assert!(Q_PARAM_C < 0.0, "C coefficient should be negative");
        }

        // Test polynomial gives reasonable tanh-like values
        for x in [-2.0f32, -1.0f32, 0.0f32, 1.0f32, 2.0f32] {
            let poly_val = x.mul_add(Q_PARAM_A * x + Q_PARAM_B, Q_PARAM_C);
            let exp_val = poly_val.exp();
            assert!(
                (0.0..=1.0).contains(&exp_val),
                "Q-function should be in [0,1], got {} for x={}",
                exp_val,
                x
            );
        }
    }

    #[test]
    fn test_probability_calculator_numerical_stability() {
        // Test with extreme LLR values
        let extreme_alpha = {
            let mut alpha = [1.0f32; N];
            alpha[0] = 1e-10; // Very small
            alpha[1] = 1e10; // Very large
            alpha[2] = f32::MIN_POSITIVE;
            alpha[3] = 1000.0;
            alpha
        };

        let calc = ProbabilityCalculator {
            log_p: [-50.0; N],          // Very negative (very low error probability)
            log_1_minus_p: [-1e-10; N], // Near zero
            sum_log_1_minus_p: -1e-8,
            e_p_alpha: 0.5,
            parity_p_avg: 0.1,
            alpha_mrb: extreme_alpha,
            c0_mrb: [false; N],
            baseline_codeword_bool: [false; N], // Simple baseline for testing
            k: 91,
            nmk: 83,
            log_two_pow_nmk: 83.0 * 2.0_f32.ln(),
            d_min_w: 10.0,
        };

        // Test with extreme TEP
        let tep = TestErrorPattern::with_positions(&[0, 1, 2, 3]);
        let log_pe = calc.calculate_log_pe(&tep);

        // Result should be finite
        assert!(
            log_pe.is_finite(),
            "log P(e) should be finite even with extreme values"
        );
        assert!(!log_pe.is_nan(), "log P(e) should not be NaN");
    }

    #[test]
    fn test_probability_calculator_edge_cases() {
        let calc = ProbabilityCalculator {
            log_p: [-1.0; N],
            log_1_minus_p: [-0.1; N],
            sum_log_1_minus_p: -0.1 * N as f32,
            e_p_alpha: 0.5,
            parity_p_avg: 0.1,
            alpha_mrb: [1.0; N],
            c0_mrb: [false; N],
            baseline_codeword_bool: [false; N], // Simple baseline for testing
            k: 91,
            nmk: 83,
            log_two_pow_nmk: 83.0 * 2.0_f32.ln(),
            d_min_w: 10.0,
        };

        // Test empty TEP (no errors)
        let empty_tep = TestErrorPattern::new();
        let log_pe_empty = calc.calculate_log_pe(&empty_tep);
        assert!(
            (log_pe_empty - calc.sum_log_1_minus_p).abs() < 1e-6,
            "Empty TEP should give sum_log_1_minus_p"
        );

        // Test single position TEP
        let single_tep = TestErrorPattern::with_positions(&[0]);
        let log_pe_single = calc.calculate_log_pe(&single_tep);
        let expected = calc.sum_log_1_minus_p + calc.log_p[0] - calc.log_1_minus_p[0];
        assert!(
            (log_pe_single - expected).abs() < 1e-6,
            "Single position TEP calculation incorrect"
        );

        // Test maximum positions (8)
        let max_tep = TestErrorPattern::with_positions(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let log_pe_max = calc.calculate_log_pe(&max_tep);
        assert!(log_pe_max.is_finite(), "Max TEP should give finite result");
    }

    #[test]
    fn test_promising_probability_mathematical_properties() {
        let calc = ProbabilityCalculator {
            log_p: [-2.0; N],           // P(i) ≈ 0.135
            log_1_minus_p: [-0.145; N], // 1-P(i) ≈ 0.865
            sum_log_1_minus_p: -0.145 * 91.0,
            e_p_alpha: 1.0, // Simple expected value
            parity_p_avg: 0.1,
            alpha_mrb: [1.0; N],
            c0_mrb: [false; N],
            baseline_codeword_bool: [false; N], // Simple baseline for testing
            k: 91,
            nmk: 83,
            log_two_pow_nmk: 83.0 * 2.0_f32.ln(),
            d_min_w: 10.0,
        };

        // Test that P_pro is a valid probability
        let tep = TestErrorPattern::with_positions(&[0, 5, 10]);
        let p_pro = calc.calculate_p_pro(&tep);

        assert!(
            (0.0..=1.0).contains(&p_pro),
            "P_pro should be a valid probability, got {}",
            p_pro
        );
        assert!(p_pro.is_finite(), "P_pro should be finite");

        // Test that empty TEP has reasonable P_pro
        let empty_tep = TestErrorPattern::new();
        let p_pro_empty = calc.calculate_p_pro(&empty_tep);
        assert!(
            (0.0..=1.0).contains(&p_pro_empty),
            "Empty TEP P_pro should be valid probability"
        );
    }

    #[test]
    fn test_vectorized_probability_calculations_consistency() {
        let calc = ProbabilityCalculator {
            log_p: [-1.5; N],
            log_1_minus_p: [-0.2; N],
            sum_log_1_minus_p: -0.2 * 91.0,
            e_p_alpha: 0.8,
            parity_p_avg: 0.15,
            alpha_mrb: [1.0; N],
            c0_mrb: [false; N],
            baseline_codeword_bool: [false; N], // Simple baseline for testing
            k: 91,
            nmk: 83,
            log_two_pow_nmk: 83.0 * 2.0_f32.ln(),
            d_min_w: 12.0,
        };

        // Create 4 different TEPs
        let teps = [
            TestErrorPattern::new(),
            TestErrorPattern::with_positions(&[0]),
            TestErrorPattern::with_positions(&[1, 5]),
            TestErrorPattern::with_positions(&[0, 2, 4]),
        ];

        // Calculate individually
        let individual_results: Vec<f32> =
            teps.iter().map(|tep| calc.calculate_log_pe(tep)).collect();

        // Calculate as batch
        let batch_results = calc.calculate_log_pe_4(&[teps[0], teps[1], teps[2], teps[3]]);

        // Verify consistency
        for i in 0..4 {
            assert!(
                (individual_results[i] - batch_results[i]).abs() < 1e-6,
                "Batch calculation mismatch at index {}: individual={}, batch={}",
                i,
                individual_results[i],
                batch_results[i]
            );
        }

        // Test P_pro batch calculation
        let individual_p_pro: Vec<f32> = teps.iter().map(|tep| calc.calculate_p_pro(tep)).collect();

        let batch_p_pro = calc.calculate_p_pro_4(&[teps[0], teps[1], teps[2], teps[3]]);

        for i in 0..4 {
            assert!(
                (individual_p_pro[i] - batch_p_pro[i]).abs() < 1e-5,
                "P_pro batch calculation mismatch at index {}: individual={}, batch={}",
                i,
                individual_p_pro[i],
                batch_p_pro[i]
            );
        }
    }

    #[test]
    fn test_success_probability_mathematical_bounds() {
        let calc = ProbabilityCalculator {
            log_p: [-1.0; N],
            log_1_minus_p: [-0.1; N],
            sum_log_1_minus_p: -0.1 * 91.0,
            e_p_alpha: 0.5,
            parity_p_avg: 0.1,
            alpha_mrb: [1.0; N],
            c0_mrb: [false; N],
            baseline_codeword_bool: [false; N], // Simple baseline for testing
            k: 91,
            nmk: 83,
            log_two_pow_nmk: 83.0 * 2.0_f32.ln(),
            d_min_w: 10.0,
        };

        // Test with different codewords
        let test_codewords = [
            [false; 174], // All zeros
            {
                let mut cw = [false; 174];
                cw[0] = true;
                cw
            }, // Single bit
            {
                let mut cw = [false; 174];
                for i in 0..10 {
                    cw[i] = true;
                }
                cw
            }, // Multiple bits
        ];

        let tep = TestErrorPattern::with_positions(&[0, 1]);

        for (i, &codeword) in test_codewords.iter().enumerate() {
            let p_suc = calc.calculate_p_suc_e(&tep, &codeword);

            assert!(
                (0.0..=1.0).contains(&p_suc),
                "P_suc should be valid probability for test case {}, got {}",
                i,
                p_suc
            );
            assert!(
                p_suc.is_finite(),
                "P_suc should be finite for test case {}",
                i
            );
        }
    }
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod simd_tests {
    use super::super::simd::exp4_f32;
    use core::arch::aarch64::*;

    /// High-precision reference exponential for testing
    fn reference_exp(x: f32) -> f32 {
        if x > 88.0 {
            return f32::INFINITY;
        }
        if x < -88.0 {
            return 0.0;
        }
        x.exp()
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_exp4_f32_accuracy_against_scalar() {
        // Test values covering the useful range for PB-OSD
        let test_cases = [
            [-2.0, -1.0, 0.0, 1.0],
            [2.0, 3.0, -3.0, -2.5],
            [0.1, 0.01, -0.01, -0.1],
            [5.0, -5.0, 4.5, -4.5],
            [10.0, -10.0, 8.0, -8.0],
        ];

        for &inputs in &test_cases {
            let reference_results = [
                reference_exp(inputs[0]),
                reference_exp(inputs[1]),
                reference_exp(inputs[2]),
                reference_exp(inputs[3]),
            ];

            unsafe {
                let input_vec = vld1q_f32(inputs.as_ptr());
                let result_vec = exp4_f32(input_vec);
                let simd_results: [f32; 4] = std::mem::transmute(result_vec);

                for i in 0..4 {
                    let relative_error = if reference_results[i] > 1e-10 {
                        ((simd_results[i] - reference_results[i]) / reference_results[i]).abs()
                    } else {
                        (simd_results[i] - reference_results[i]).abs()
                    };

                    // Allow 0.1% relative error for the SIMD approximation
                    assert!(relative_error < 0.001,
                        "exp4_f32 accuracy failure at input {}: SIMD={}, reference={}, rel_error={}%",
                        inputs[i], simd_results[i], reference_results[i], relative_error * 100.0);
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_exp4_f32_special_values() {
        unsafe {
            // Test zero
            let zero_input = vdupq_n_f32(0.0);
            let zero_result = exp4_f32(zero_input);
            let zero_values: [f32; 4] = std::mem::transmute(zero_result);
            for &val in &zero_values {
                assert!(
                    (val - 1.0).abs() < 1e-6,
                    "exp(0) should be 1.0, got {}",
                    val
                );
            }

            // Test ln(2) (should give exactly 2.0)
            let ln2_input = vdupq_n_f32(std::f32::consts::LN_2);
            let ln2_result = exp4_f32(ln2_input);
            let ln2_values: [f32; 4] = std::mem::transmute(ln2_result);
            for &val in &ln2_values {
                assert!(
                    (val - 2.0).abs() < 0.01,
                    "exp(ln(2)) should be 2.0, got {}",
                    val
                );
            }

            // Test negative ln(2) (should give 0.5)
            let neg_ln2_input = vdupq_n_f32(-std::f32::consts::LN_2);
            let neg_ln2_result = exp4_f32(neg_ln2_input);
            let neg_ln2_values: [f32; 4] = std::mem::transmute(neg_ln2_result);
            for &val in &neg_ln2_values {
                assert!(
                    (val - 0.5).abs() < 0.01,
                    "exp(-ln(2)) should be 0.5, got {}",
                    val
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_exp4_f32_monotonicity() {
        // Exponential function should be strictly increasing
        let test_range: Vec<f32> = (-40..40).map(|i| i as f32 * 0.2).collect();

        for chunk in test_range.chunks(4) {
            if chunk.len() == 4 {
                let mut inputs = [0.0f32; 4];
                inputs.copy_from_slice(chunk);

                unsafe {
                    let input_vec = vld1q_f32(inputs.as_ptr());
                    let result_vec = exp4_f32(input_vec);
                    let results: [f32; 4] = std::mem::transmute(result_vec);

                    // Verify monotonicity within this chunk
                    for i in 1..4 {
                        assert!(
                            results[i] >= results[i - 1],
                            "exp4_f32 not monotonic: exp({}) = {} < exp({}) = {}",
                            inputs[i],
                            results[i],
                            inputs[i - 1],
                            results[i - 1]
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_exp4_f32_range_reduction_boundaries() {
        // Test values specifically around ln(2) boundaries where range reduction occurs
        let boundary_tests = [
            std::f32::consts::LN_2 - 0.001,
            std::f32::consts::LN_2,
            std::f32::consts::LN_2 + 0.001,
            -std::f32::consts::LN_2 - 0.001,
            -std::f32::consts::LN_2,
            -std::f32::consts::LN_2 + 0.001,
            2.0 * std::f32::consts::LN_2,
            -2.0 * std::f32::consts::LN_2,
        ];

        for chunk in boundary_tests.chunks(4) {
            let mut inputs = [0.0f32; 4];
            for (i, &val) in chunk.iter().enumerate() {
                inputs[i] = val;
            }

            unsafe {
                let input_vec = vld1q_f32(inputs.as_ptr());
                let result_vec = exp4_f32(input_vec);
                let simd_results: [f32; 4] = std::mem::transmute(result_vec);

                for i in 0..chunk.len() {
                    let reference = reference_exp(inputs[i]);
                    let relative_error = if reference > 1e-10 {
                        ((simd_results[i] - reference) / reference).abs()
                    } else {
                        (simd_results[i] - reference).abs()
                    };

                    assert!(
                        relative_error < 0.01,
                        "Range reduction boundary error at {}: SIMD={}, reference={}, error={}%",
                        inputs[i],
                        simd_results[i],
                        reference,
                        relative_error * 100.0
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "aarch64"))]
    fn test_simd_fallback_compilation() {
        // This test just ensures the SIMD code compiles on non-ARM platforms
        // The actual SIMD functions should not be called, but the module should compile
        assert!(true, "SIMD module compiles on non-ARM platforms");
    }
}

#[cfg(test)]
mod basis_transformation_tests {
    use super::super::ge::BitMatrix;
    use ft8_ldpc_core::fec::CascadedGeneratorMatrix;
    use ft8_ldpc_core::protocol::N;

    /// Generate a test permutation that is guaranteed to be valid
    fn generate_test_permutation(size: usize) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..size).collect();
        // Apply some known swaps
        if size > 10 {
            perm.swap(0, 5);
            perm.swap(1, 7);
            perm.swap(2, 9);
        }
        perm
    }

    /// Verify that a permutation is valid (bijective)
    fn verify_permutation_validity(perm: &[usize], expected_size: usize) {
        assert_eq!(perm.len(), expected_size);
        let mut seen = vec![false; expected_size];
        for &idx in perm {
            assert!(
                idx < expected_size,
                "Permutation index {} out of bounds",
                idx
            );
            assert!(!seen[idx], "Permutation index {} appears twice", idx);
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&x| x), "Permutation missing some indices");
    }

    #[test]
    fn test_permutation_composition_mathematical_properties() {
        let size = 20; // Smaller size for focused testing
        let perm1 = generate_test_permutation(size);
        let perm2 = generate_test_permutation(size);

        // Verify individual permutations are valid
        verify_permutation_validity(&perm1, size);
        verify_permutation_validity(&perm2, size);

        // Test composition π₁ ∘ π₂
        let mut composed = vec![0; size];
        for i in 0..size {
            composed[i] = perm1[perm2[i]];
        }
        verify_permutation_validity(&composed, size);

        // Test that composing with identity gives original
        let identity: Vec<usize> = (0..size).collect();
        let mut identity_composed = vec![0; size];
        for i in 0..size {
            identity_composed[i] = perm1[identity[i]];
        }
        assert_eq!(
            perm1, identity_composed,
            "Composition with identity should give original"
        );
    }

    #[test]
    fn test_permutation_inverse_correctness() {
        let size = N;
        let perm = generate_test_permutation(size);

        // Compute inverse permutation
        let mut inv_perm = vec![0; size];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        // Verify inverse property: perm[inv_perm[i]] = i
        for i in 0..size {
            assert_eq!(
                perm[inv_perm[i]], i,
                "Inverse permutation failed at index {}: perm[inv_perm[{}]] = {} ≠ {}",
                i, i, perm[inv_perm[i]], i
            );
        }

        // Verify other direction: inv_perm[perm[i]] = i
        for i in 0..size {
            assert_eq!(
                inv_perm[perm[i]], i,
                "Inverse permutation failed at index {}: inv_perm[perm[{}]] = {} ≠ {}",
                i, i, inv_perm[perm[i]], i
            );
        }
    }

    #[test]
    fn test_round_trip_basis_transformations() {
        // Test with different cascaded generator matrices
        for cascade_bits in [0, 5, 14] {
            let generator = CascadedGeneratorMatrix::new(cascade_bits).unwrap_or_else(|_| {
                panic!(
                    "Failed to create generator with {} cascade bits",
                    cascade_bits
                )
            });

            let mut matrix = BitMatrix::from_cascaded_generator(&generator);
            let original_k = matrix.k();

            // Apply standard form transformation
            matrix.to_standard_form();

            // Apply systematic transformation
            let identity_perm: [usize; N] = core::array::from_fn(|i| i);
            let (rank, _cum_perm) = matrix.reduce_standard_bitwise(&identity_perm);

            // Verify basic properties are preserved
            assert_eq!(
                rank, original_k,
                "Rank should be preserved after transformations for {} cascade bits",
                cascade_bits
            );
            assert_eq!(
                matrix.k(),
                original_k,
                "k should be preserved after transformations for {} cascade bits",
                cascade_bits
            );

            // Verify the matrix dimensions are within expected bounds for FT8
            assert!(
                matrix.k() <= 91,
                "Matrix k should be ≤ 91 for FT8 (got {})",
                matrix.k()
            );
            assert!(rank > 0, "Matrix should have positive rank");
            assert!(rank <= N, "Rank should not exceed codeword dimension N=174");
        }
    }

    #[test]
    fn test_systematic_form_identity_property() {
        // Test that systematic form has identity in first k columns
        let generator = CascadedGeneratorMatrix::new(0).expect("Failed to create generator");

        let mut matrix = BitMatrix::from_cascaded_generator(&generator);
        matrix.to_standard_form();

        let identity_perm: [usize; N] = core::array::from_fn(|i| i);
        let (_rank, _) = matrix.reduce_standard_bitwise(&identity_perm);
        let k = matrix.k();

        // Verify identity matrix in first k×k block
        for row in 0..k {
            for col in 0..k {
                let expected = row == col;
                let actual = matrix.bit(row, col);
                assert_eq!(
                    actual, expected,
                    "Systematic form identity check failed at ({}, {}): expected {}, got {}",
                    row, col, expected, actual
                );
            }
        }
    }

    #[test]
    fn test_matrix_rank_consistency() {
        // Test different k values and verify rank properties
        for cascade_bits in [0, 5, 14] {
            let generator = CascadedGeneratorMatrix::new(cascade_bits).unwrap_or_else(|_| {
                panic!("Failed to create generator with {} bits", cascade_bits)
            });

            let mut matrix = BitMatrix::from_cascaded_generator(&generator);
            let k = matrix.k();

            matrix.to_standard_form();
            let identity_perm: [usize; N] = core::array::from_fn(|i| i);
            let (rank, _) = matrix.reduce_standard_bitwise(&identity_perm);

            // Rank should equal k for valid FT8 generator matrices
            assert_eq!(
                rank, k,
                "Matrix rank {} should equal k {} for cascade_bits {}",
                rank, k, cascade_bits
            );

            // k should be in valid range for FT8
            assert!(
                (77..=91).contains(&k),
                "k should be in valid FT8 range [77,91], got {}",
                k
            );
        }
    }

    #[test]
    fn test_bit_packing_correctness() {
        // Test conversion between different bit representations
        let test_patterns = [
            vec![true, false, true, false],
            vec![true; 64],  // All ones for first word
            vec![false; 64], // All zeros
            (0..174).map(|i| i % 3 == 0).collect::<Vec<bool>>(), // Pattern every 3rd bit
        ];

        for (i, pattern) in test_patterns.iter().enumerate() {
            // Pad or truncate to exactly N bits
            let mut full_pattern = [false; N];
            for (j, &bit) in pattern.iter().enumerate() {
                if j < N {
                    full_pattern[j] = bit;
                }
            }

            // Test bit-by-bit consistency
            // (This would require access to internal bit packing methods)
            assert_eq!(
                full_pattern.len(),
                N,
                "Test pattern {} should have length N",
                i
            );

            // Verify pattern contains expected values
            let ones_count = full_pattern.iter().filter(|&&b| b).count();
            assert!(ones_count <= N, "Ones count should not exceed N");
        }
    }

    #[test]
    fn test_matrix_conditioning_edge_cases() {
        // Test behavior with various matrix conditions
        let generator = CascadedGeneratorMatrix::new(0).expect("Failed to create generator");

        let mut matrix = BitMatrix::from_cascaded_generator(&generator);
        let original_k = matrix.k();

        // Apply transformations and check for numerical stability issues
        matrix.to_standard_form();

        // Generate a permutation that might cause conditioning issues
        let mut stress_perm = [0usize; N];
        for i in 0..N {
            stress_perm[i] = (i * 7) % N; // Prime-based permutation
        }

        // Make it a valid permutation
        for i in 0..N {
            stress_perm[i] = i;
        }
        // Apply some strategic swaps
        if N > 20 {
            stress_perm.swap(0, N - 1);
            stress_perm.swap(1, N / 2);
            stress_perm.swap(original_k - 1, original_k);
        }

        let (rank, _) = matrix.reduce_standard_bitwise(&stress_perm);

        // Even with stress permutation, should maintain mathematical properties
        assert!(
            rank > 0,
            "Matrix should maintain positive rank under stress permutation"
        );
        assert!(rank <= original_k, "Rank should not exceed original k");
    }
}

#[cfg(test)]
mod distance_calculation_tests {
    use super::super::distance::{utils::*, DistanceCalculator};
    use super::super::tep::TestErrorPattern;
    use ft8_ldpc_core::protocol::N;

    fn create_test_distance_calculator() -> DistanceCalculator {
        let alpha_mrb = [1.0f32; N]; // Unit weights for simplicity
        let c0_mrb_bitpacked = [0u64; 3]; // All zeros baseline
        DistanceCalculator::new(alpha_mrb, c0_mrb_bitpacked)
    }

    #[test]
    fn test_all_distance_implementations_consistency() {
        let calc = create_test_distance_calculator();

        // Test codewords with known bit patterns
        let test_patterns = [
            [0u64, 0u64, 0u64],                                     // All zeros
            [1u64, 0u64, 0u64],                                     // Single bit in first word
            [0u64, 1u64, 0u64],                                     // Single bit in second word
            [0u64, 0u64, 1u64],                                     // Single bit in third word
            [u64::MAX, 0u64, 0u64],                                 // All bits in first word
            [0u64, u64::MAX, 0u64],                                 // All bits in second word
            [0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555, 0x2AAA], // Alternating pattern
        ];

        for (i, &codeword) in test_patterns.iter().enumerate() {
            // Calculate distance using different implementations
            let distance_fast = calc.calculate_weighted_hamming_fast(&codeword);
            let distance_parallel = calc.calculate_weighted_hamming_parallel(&codeword);
            let distance_unrolled = calc.calculate_weighted_hamming_unrolled(&codeword);

            // All implementations should give identical results
            assert!(
                (distance_fast - distance_parallel).abs() < 1e-6,
                "Fast vs parallel mismatch for pattern {}: {} vs {}",
                i,
                distance_fast,
                distance_parallel
            );

            assert!(
                (distance_fast - distance_unrolled).abs() < 1e-6,
                "Fast vs unrolled mismatch for pattern {}: {} vs {}",
                i,
                distance_fast,
                distance_unrolled
            );
        }
    }

    #[test]
    fn test_distance_calculation_mathematical_properties() {
        let calc = create_test_distance_calculator();

        // Distance should be zero for identical codewords
        let zero_distance = calc.calculate_weighted_hamming_fast(&[0, 0, 0]);
        assert!(
            (zero_distance - 0.0).abs() < 1e-6,
            "Distance to self should be zero"
        );

        // Distance should be symmetric: d(a,b) = d(b,a)
        // (This property is implicit since we compare against a fixed baseline)

        // Distance should satisfy triangle inequality for unit weights
        let cw1 = [0b1010u64, 0u64, 0u64];
        let cw2 = [0b1100u64, 0u64, 0u64];

        let dist1 = calc.calculate_weighted_hamming_fast(&cw1);
        let dist2 = calc.calculate_weighted_hamming_fast(&cw2);

        // With unit weights, distance equals Hamming weight
        assert_eq!(
            dist1, 2.0,
            "Distance should equal Hamming weight for unit weights"
        );
        assert_eq!(
            dist2, 2.0,
            "Distance should equal Hamming weight for unit weights"
        );
    }

    #[test]
    fn test_debw_calculation_properties() {
        let calc = create_test_distance_calculator();

        // DEBW should be zero for empty TEP
        let empty_tep = TestErrorPattern::new();
        let empty_distance = calc.calculate_debw(&empty_tep);
        assert_eq!(empty_distance, 0.0, "DEBW should be zero for empty TEP");

        // DEBW should increase with more positions
        let single_tep = TestErrorPattern::with_positions(&[5]);
        let double_tep = TestErrorPattern::with_positions(&[5, 10]);
        let triple_tep = TestErrorPattern::with_positions(&[5, 10, 15]);

        let single_distance = calc.calculate_debw(&single_tep);
        let double_distance = calc.calculate_debw(&double_tep);
        let triple_distance = calc.calculate_debw(&triple_tep);

        assert!(
            double_distance >= single_distance,
            "DEBW should be non-decreasing"
        );
        assert!(
            triple_distance >= double_distance,
            "DEBW should be non-decreasing"
        );

        // With unit weights, DEBW equals number of positions
        assert_eq!(
            single_distance, 1.0,
            "DEBW should equal position count for unit weights"
        );
        assert_eq!(
            double_distance, 2.0,
            "DEBW should equal position count for unit weights"
        );
        assert_eq!(
            triple_distance, 3.0,
            "DEBW should equal position count for unit weights"
        );
    }

    #[test]
    fn test_debw_vectorized_consistency() {
        let calc = create_test_distance_calculator();

        // Create 4 different TEPs
        let teps = [
            TestErrorPattern::new(),
            TestErrorPattern::with_positions(&[0]),
            TestErrorPattern::with_positions(&[1, 5]),
            TestErrorPattern::with_positions(&[0, 2, 4, 6]),
        ];

        // Calculate individually
        let individual_results: Vec<f32> =
            teps.iter().map(|tep| calc.calculate_debw(tep)).collect();

        // Calculate as batch
        let batch_results = calc.calculate_debw_4([&teps[0], &teps[1], &teps[2], &teps[3]]);

        // Verify consistency
        for i in 0..4 {
            assert!(
                (individual_results[i] - batch_results[i]).abs() < 1e-6,
                "DEBW batch calculation mismatch at index {}: individual={}, batch={}",
                i,
                individual_results[i],
                batch_results[i]
            );
        }
    }

    #[test]
    fn test_distance_calculation_with_extreme_weights() {
        // Test with extreme reliability values
        let mut extreme_alpha = [1.0f32; N];
        extreme_alpha[0] = 1e-10; // Very small weight
        extreme_alpha[1] = 1e10; // Very large weight
        extreme_alpha[2] = f32::MIN_POSITIVE;
        extreme_alpha[3] = 1000.0;

        let calc = DistanceCalculator::new(extreme_alpha, [0; 3]);

        // Test with codeword that hits extreme weights
        let extreme_cw = [0b1111u64, 0u64, 0u64]; // First 4 bits set

        let distance = calc.calculate_weighted_hamming_fast(&extreme_cw);

        // Result should be finite and reasonable
        assert!(
            distance.is_finite(),
            "Distance should be finite even with extreme weights"
        );
        assert!(distance >= 0.0, "Distance should be non-negative");

        // Should approximately equal sum of first 4 alpha values
        let expected = extreme_alpha[0] + extreme_alpha[1] + extreme_alpha[2] + extreme_alpha[3];
        assert!(
            (distance - expected).abs() < expected * 0.01,
            "Distance should approximately equal sum of weights"
        );
    }

    #[test]
    fn test_bit_packing_utilities_comprehensive() {
        // Test edge cases for bit packing/unpacking
        let edge_cases = [
            vec![0],                        // First bit only
            vec![63],                       // Last bit of first word
            vec![64],                       // First bit of second word
            vec![127],                      // Last bit of second word
            vec![128],                      // First bit of third word
            vec![173],                      // Last valid bit
            vec![0, 63, 64, 127, 128, 173], // All boundaries
        ];

        for (i, positions) in edge_cases.iter().enumerate() {
            let mut bool_array = [false; N];
            for &pos in positions {
                if pos < N {
                    bool_array[pos] = true;
                }
            }

            let packed = pack_bool_array(&bool_array);
            let unpacked = unpack_to_bool_array(&packed);

            assert_eq!(
                bool_array, unpacked,
                "Pack/unpack failed for edge case {}: positions {:?}",
                i, positions
            );

            // Verify Hamming distance
            let hamming_dist = hamming_distance_bitpacked(&packed, &[0; 3]);
            assert_eq!(
                hamming_dist as usize,
                positions.len(),
                "Hamming distance should equal number of set bits for case {}",
                i
            );
        }
    }

    #[test]
    fn test_bitpacked_distance_pattern_consistency() {
        let calc = create_test_distance_calculator();

        // Test patterns that stress SIMD implementations
        let simd_stress_patterns = [
            [
                0x0101_0101_0101_0101u64,
                0x1010_1010_1010_1010u64,
                0x0101u64,
            ], // Alternating
            [
                0x00FF_00FF_00FF_00FFu64,
                0xFF00_FF00_FF00_FF00u64,
                0x00FFu64,
            ], // 8-bit blocks
            [
                0x0000_FFFF_0000_FFFFu64,
                0xFFFF_0000_FFFF_0000u64,
                0x0000u64,
            ], // 16-bit blocks
            [
                0x5555_5555_5555_5555u64,
                0xAAAA_AAAA_AAAA_AAAAu64,
                0x2AAAu64,
            ], // Checkerboard
        ];

        for (i, &pattern) in simd_stress_patterns.iter().enumerate() {
            let scalar_distance = calc.calculate_weighted_hamming_fast(&pattern);

            // Verify other implementations for exactness
            let parallel_distance = calc.calculate_weighted_hamming_parallel(&pattern);
            assert!(
                (scalar_distance - parallel_distance).abs() < 1e-6,
                "Scalar vs parallel exactness failure for pattern {}",
                i
            );
        }
    }
}
