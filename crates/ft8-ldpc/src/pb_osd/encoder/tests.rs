

use super::*;
use ft8_ldpc_core::fec::CascadedGeneratorMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

#[cfg(test)]
use super::*;
const DUMMY_TEP: TestErrorPattern = TestErrorPattern::new();
#[test]
fn test_row_permutation_application() {
    let k = 77;
    let mut input = vec![vec![false; k]; N];

    // Create a pattern in the first 3 columns
    let pattern = [
        [true, false, false], // [1 0 0]
        [false, true, false], // [0 1 0]
        [false, false, true], // [0 0 1]
    ];

    for i in 0..3 {
        for j in 0..3 {
            input[i][j] = pattern[j][i];
        }
    }

    // Create a full-length permutation array (n = 174)
    let mut pi1 = [0; N];
    // Swap columns 0 and 1
    pi1[0] = 1;
    pi1[1] = 0;
    // Leave the rest unchanged
    for i in 2..N {
        pi1[i] = i;
    }

    let mut bit_matrix = BitMatrix::from_row_major(&input, k);
    bit_matrix.apply_row_permutation(&pi1);

    // Verify the permutation is applied correctly
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(
                bit_matrix.bit(i, j),
                pattern[i][pi1[j]],
                "Permutation application failed at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_reliability_ordering_permutation() {
    let mut alpha = [0.0; N];

    // Set test values in the first 4 positions
    alpha[0] = 0.5;
    alpha[1] = 2.0;
    alpha[2] = 1.0;
    alpha[3] = 1.5;

    // Compute reliability ordering permutation
    let pi1 = FrameDecoder::compute_reliability_permutation(&alpha);

    // Verify the permutation orders by descending reliability
    assert_eq!(pi1[0], 1, "Most reliable column should be first");
    assert_eq!(pi1[1], 3, "Second most reliable column should be second");
    assert_eq!(pi1[2], 2, "Third most reliable column should be third");
    assert_eq!(pi1[3], 0, "Least reliable column should be fourth");
}

#[test]
fn test_systematic_form_transformation() {
    let k = 77;
    let mut input = vec![vec![false; k]; N];

    // Create a pattern in the first 3 columns
    let pattern = [
        [false, true, false], // [0 1 0]
        [true, false, false], // [1 0 0]
        [false, false, true], // [0 0 1]
    ];

    for i in 0..3 {
        for j in 0..3 {
            input[i][j] = pattern[j][i];
        }
    }

    // Set identity portion in the first k columns
    for i in 3..k {
        input[i][i] = true;
    }

    // Convert to systematic form
    let mut bit_matrix = BitMatrix::from_row_major(&input, k);
    let rank = bit_matrix.reduce_generator();
    assert_eq!(rank, k, "Matrix should achieve full rank");

    // Verify systematic form in the first k columns
    for i in 0..k {
        for j in 0..k {
            assert_eq!(
                bit_matrix.bit(i, j),
                i == j,
                "Matrix not in systematic form at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_permutation_inversion() {
    // Create a random permutation pi1
    let mut pi1 = [0; N];
    for i in 0..N {
        pi1[i] = i;
    }
    // Shuffle the permutation
    pi1.swap(0, 1);
    pi1.swap(2, 3);

    // Compute the inverse permutation
    let mut inverse_pi1 = [0; N];
    for i in 0..N {
        inverse_pi1[pi1[i]] = i;
    }

    // Verify that applying pi1 and inverse_pi1 restores the original order
    let mut restored = [0; N];
    for i in 0..N {
        restored[i] = inverse_pi1[pi1[i]];
    }
    for i in 0..N {
        assert_eq!(
            restored[i], i,
            "Permutation inversion failed at position {}",
            i
        );
    }
}

#[test]
fn test_reliability_ordering() {
    let k = 77; // k = 77 corresponds to num_cascaded_bits = 14 (since k = 91 - num_cascaded_bits)
    let mut alpha = [0.0; N];

    // Set test values in first 4 positions
    alpha[0] = 0.5;
    alpha[1] = 2.0;
    alpha[2] = 1.0;
    alpha[3] = 1.5;
    for i in 4..N {
        alpha[i] = 0.1;
    }

    let mut matrix = vec![vec![false; k]; N]; // N rows x k columns

    // Set identity portion (first k rows)
    for i in 0..k {
        matrix[i][i] = true;
    }

    // Convert to BitMatrix
    let mut bit_matrix = BitMatrix::from_row_major(&matrix, k);

    let decoder = FrameDecoder::new(alpha, 1.0, k as u8, &mut bit_matrix, [0.0; N], 2);

    // Verify pi1 orders by descending reliability
    assert_eq!(decoder.pi1[0], 1);
    assert_eq!(decoder.pi1[1], 3);
    assert_eq!(decoder.pi1[2], 2);
    assert_eq!(decoder.pi1[3], 0);
}

#[test]
fn test_basis_conversion_step_by_step() {
    let k = 77;
    let mut alpha = [0.0; N];

    // Set test values in the first 4 positions for reliability ordering
    alpha[0] = 0.5;
    alpha[1] = 2.0;
    alpha[2] = 1.0;
    alpha[3] = 1.5;

    // Create k×n input matrix in row-major format
    let mut input = vec![vec![false; k]; N];

    // Set identity portion in the first k columns
    for i in 0..k {
        input[i][i] = true;
    }

    // Add a test pattern in the top-left corner
    let corner_pattern = [[1, 1, 0], [1, 0, 1], [0, 1, 0]];

    for i in 0..3 {
        for j in 0..3 {
            if corner_pattern[j][i] == 1 {
                input[i][j] = !input[i][j]; // XOR with existing value
            }
        }
    }

    // Create BitMatrix
    let mut bit_matrix = BitMatrix::from_row_major(&input, k);

    // Get reliability ordering permutation
    let pi1 = FrameDecoder::compute_reliability_permutation(&alpha);

    let mut inverse_pi1 = [0; N];
    for i in 0..N {
        inverse_pi1[pi1[i]] = i;
    }

    // Apply reliability ordering
    bit_matrix.apply_row_permutation(&pi1);

    // Convert to systematic form (this will give us pi2 permutation)
    let rank = bit_matrix.reduce_generator();
    assert_eq!(rank, k, "Matrix should achieve full rank");

    // Create a test codeword in the systematic basis
    let mut systematic_cw = [false; N];
    systematic_cw[0] = true;
    systematic_cw[2] = true;

    // Convert to the original basis: Undo pi2 (GE permutation) first
    let mut temp = [false; N];
    for i in 0..N {
        temp[i] = systematic_cw[bit_matrix.permutation_index(i)];
    }

    // Then undo pi1 (reliability permutation)
    let mut original_cw = [false; N];
    for i in 0..N {
        original_cw[i] = temp[pi1[i]]; // Read from permuted position
    }

    // Convert back to the systematic basis
    let mut back_to_systematic = [false; N];

    // First apply reliability ordering (pi1)
    let mut temp = [false; N];
    for i in 0..N {
        temp[i] = original_cw[inverse_pi1[i]];
    }

    // Compute inverse_pi2
    let mut inverse_pi2 = [0; N];
    for i in 0..N {
        inverse_pi2[bit_matrix.permutation_index(i)] = i;
    }

    // Reapply pi2 using the inverse permutation
    for i in 0..N {
        back_to_systematic[i] = temp[inverse_pi2[i]];
    }

    println!("Systematic Codeword: {:?}", systematic_cw);
    println!("Original Codeword: {:?}", original_cw);
    println!("Back to Systematic: {:?}", back_to_systematic);

    // Print the permutations
    println!("pi1: {:?}", pi1);
    println!("inverse_pi1: {:?}", inverse_pi1);
    println!("pi2: {:?}", bit_matrix.pi1());

    // Verify the conversion is correct
    assert_eq!(back_to_systematic, systematic_cw, "Basis conversion failed");
}

/// Generate LLRs according to a Bi-AWGN channel with noise variance 1/gamma
fn generate_llrs(codeword: &[bool; N], gamma: f32) -> [f32; N] {
    let mut llrs = [0.0; N];
    let noise_std = 1.0 / gamma.sqrt();
    let mut rng = rand::rng();

    for i in 0..N {
        // Map true (1) to -1, false (0) to +1
        let s_i = if codeword[i] { -1.0 } else { 1.0 };

        // Generate Gaussian noise with mean 0 and variance 1/gamma
        let w_i: f32 = rng.sample::<f32, StandardNormal>(StandardNormal) * noise_std;

        // Received signal: r = s + w
        let r_i = s_i + w_i;

        // Compute LLR: 2 * gamma * r_i
        llrs[i] = 2.0 * gamma * r_i;
    }

    llrs
}

#[test]
fn test_exp4_f32_accuracy() {
    unsafe {
        let inputs = [-88.0_f32, -7.232225_f32, 0.0_f32, 87.999_f32];
        let input_vec = vld1q_f32(inputs.as_ptr());
        let approx_vec = exp4_f32(input_vec);

        // Transmute the NEON register to a [f32; 4]
        let approx_arr: [f32; 4] = transmute(approx_vec);

        // Compare each lane to the built-in expf
        for i in 0..4 {
            let exact = inputs[i].exp();
            let rel_error = (approx_arr[i] - exact).abs() / exact.abs().max(1e-4);
            assert!(
                rel_error < 1e-5,
                "Lane {i}: input={} approx={} exact={} rel_err={}",
                inputs[i],
                approx_arr[i],
                exact,
                rel_error
            );
        }
    }
}

/// Generate test TEPs with various configurations
fn generate_test_teps(k: u8, rng: &mut StdRng) -> Vec<TestErrorPattern> {
    let mut teps = Vec::new();

    // Empty TEP
    teps.push(TestErrorPattern::new());

    // Single position TEPs
    for i in 0..5 {
        let mut tep = TestErrorPattern::new();
        tep.positions_mut()[0] = i;
        tep.set_num_positions(1);
        teps.push(tep);
    }

    // Two position TEPs
    for i in 0..3 {
        for j in (i + 1)..4 {
            let mut tep = TestErrorPattern::new();
            tep.positions_mut()[0] = i;
            tep.positions_mut()[1] = j;
            tep.set_num_positions(2);
            teps.push(tep);
        }
    }

    // Random TEPs with 3-6 positions
    for num_pos in 3..=6 {
        for _ in 0..5 {
            let mut tep = TestErrorPattern::new();
            let mut used = std::collections::HashSet::new();

            for i in 0..num_pos {
                let mut pos;
                loop {
                    pos = rng.random_range(0..k);
                    if !used.contains(&pos) {
                        used.insert(pos);
                        break;
                    }
                }
                tep.positions_mut()[i as usize] = pos;
            }

            // Sort positions for consistency
            tep.positions_mut()[..num_pos as usize].sort_unstable();
            tep.set_num_positions(num_pos);
            teps.push(tep);
        }
    }

    teps
}

fn create_test_matrix(k: usize) -> BitMatrix {
    // Cascading 91 - k CRC bits produces a generator of dimension k.
    let cascade_index = 91 - k;
    let generator = CascadedGeneratorMatrix::new(cascade_index)
        .unwrap_or_else(|_| panic!("Could not build generator for k = {}", k));

    BitMatrix::from_cascaded_generator(&generator)
}

/// Builds an N × k matrix with an identity block above zero rows.
fn make_identity_bitmatrix(k: usize) -> BitMatrix {
    const N: usize = 174;
    assert!((77..=91).contains(&k), "k must be between 77 and 91");

    // Build row‐major: N rows, each row is length k.
    // For i in 0..k, row i has a true in column i; for i ≥ k, entire row is `false`.
    let mut rows: Vec<Vec<bool>> = Vec::with_capacity(N);
    for i in 0..N {
        let mut row = vec![false; k];
        if i < k {
            row[i] = true; // identity portion
        }
        rows.push(row);
    }

    BitMatrix::from_row_major(&rows, k)
}

#[test]
fn test_batch_vs_single_tep_processing() {
    // Test with different k values in valid range [77, 91]
    for k in [77u8, 83, 91] {
        println!("Testing with k = {}", k);

        let mut rng = StdRng::seed_from_u64(42);

        // Create test data
        let mut matrix = create_test_matrix(k as usize);
        let mut alpha = [0.0f32; N];
        let mut llrs = [0.0f32; N];

        // Generate random but deterministic alpha and LLR values
        for i in 0..N {
            alpha[i] = rng.random_range(0.1..5.0);
            llrs[i] = if rng.random_bool(0.5) {
                rng.random_range(0.1..3.0)
            } else {
                rng.random_range(-3.0..-0.1)
            };
        }

        let gamma = 2.0f32;
        let mut decoder = FrameDecoder::new(alpha, gamma, k, &mut matrix, llrs, 2);

        // Generate test TEPs
        let test_teps = generate_test_teps(k, &mut rng);

        // Track global TEP count
        let mut global_tep_count = 0u32;

        // Process TEPs in batches of 4
        for chunk in test_teps.chunks(4) {
            let mut batch: [&TestErrorPattern; 4] = [&DUMMY_TEP; 4];
            for (i, tep) in chunk.iter().enumerate() {
                batch[i] = tep;
            }

            // Single TEP processing results
            let mut single_results = Vec::new();
            for i in 0..chunk.len() {
                let tep = batch[i];

                // Calculate individual values
                let log_pe = decoder.calculate_log_pe(tep);
                let debw = FrameDecoder::calculate_debw(tep, &decoder.alpha_mrb);
                let p_pro = decoder.calculate_p_pro(tep);

                // Use global count for consistency with batch function
                let should_process = p_pro > decoder.p_pro_t || global_tep_count == 0;

                let (distance, codeword) = if should_process {
                    let cw = decoder.generate_codeword_v2(tep);
                    let cw_bool = FrameDecoder::convert_bitpacked_codeword_to_bool(cw);
                    let dist = FrameDecoder::calculate_weighted_hamming(&cw_bool, &decoder.c0_mrb, &decoder.alpha_mrb);
                    (dist, cw)
                } else {
                    (f32::INFINITY, [0u64; 3])
                };

                single_results.push((log_pe, debw, p_pro, should_process, distance, codeword));
                global_tep_count += 1;
            }

            // Batch processing - use the global count from start of this chunk
            let batch_start_count = global_tep_count - chunk.len() as u32;
            let (batch_should_process, batch_distances, batch_codewords, _) =
                decoder.process_tep_batch_fused(batch, chunk.len(), batch_start_count);

            // Also test individual batch functions
            let batch_log_pe = decoder.calculate_log_pe_4(batch);
            let batch_debw = FrameDecoder::calculate_debw_4(batch, &decoder.alpha_mrb);
            let batch_p_pro = decoder.calculate_p_pro_4(batch, &batch_log_pe);

            // Compare results
            for i in 0..chunk.len() {
                let (
                    single_log_pe,
                    single_debw,
                    single_p_pro,
                    single_should_process,
                    single_dist,
                    single_cw,
                ) = &single_results[i];

                // Test individual batch functions
                assert!(
                    (batch_log_pe[i] - single_log_pe).abs() < 1e-5,
                    "log_pe mismatch at {}: batch={:.6}, single={:.6}",
                    i,
                    batch_log_pe[i],
                    single_log_pe
                );

                assert!(
                    (batch_debw[i] - single_debw).abs() < 1e-5,
                    "debw mismatch at {}: batch={:.6}, single={:.6}",
                    i,
                    batch_debw[i],
                    single_debw
                );

                assert!(
                    (batch_p_pro[i] - single_p_pro).abs() < 1e-5,
                    "p_pro mismatch at {}: batch={:.6}, single={:.6}",
                    i,
                    batch_p_pro[i],
                    single_p_pro
                );

                // Test fused function results
                assert_eq!(
                    batch_should_process[i], *single_should_process,
                    "should_process mismatch at {}: batch={}, single={}, p_pro={:.6}, p_pro_t={:.6}, batch_start_count={}",
                    i, batch_should_process[i], single_should_process, single_p_pro, decoder.p_pro_t, batch_start_count
                );

                if *single_should_process {
                    // F16 lookup tables have different precision than scalar boolean calculation
                    // Allow 0.1% relative error which is reasonable for F16 vs scalar precision
                    let relative_error =
                        (batch_distances[i] - single_dist).abs() / single_dist.max(1.0);
                    assert!(
                        relative_error < 1e-3, // 0.1% relative tolerance for F16 vs scalar precision
                        "distance mismatch at {}: batch={:.6}, single={:.6}, relative_error={:.6}",
                        i,
                        batch_distances[i],
                        single_dist,
                        relative_error
                    );

                    assert_eq!(batch_codewords[i], *single_cw, "codeword mismatch at {}", i);
                }
            }
        }
    }
}

#[test]
fn test_edge_cases() {
    let k = 83u8;
    let mut matrix = create_test_matrix(k as usize);
    let alpha = [1.0f32; N];
    let llrs = [0.5f32; N];
    let gamma = 1.0f32;

    let mut decoder = FrameDecoder::new(alpha, gamma, k, &mut matrix, llrs, 2);

    // Test with all empty TEPs
    let empty_teps = [&DUMMY_TEP; 4];
    let (should_process, _, _, _) = decoder.process_tep_batch_fused(empty_teps, 4, 0);

    // All empty TEPs should be processed when num_teps_processed = 0
    assert!(should_process.iter().all(|&x| x));

    // Test with maximum positions
    let mut max_tep = TestErrorPattern::new();
    for i in 0..8 {
        max_tep.positions_mut()[i] = i as u8;
    }
    max_tep.set_num_positions(8);

    let max_teps = [&max_tep; 4];
    let (_, _, codewords, _) = decoder.process_tep_batch_fused(max_teps, 4, 100);

    // Verify codewords are different from baseline
    for cw in &codewords {
        assert_ne!(*cw, decoder.baseline_codeword);
    }
}

#[test]
fn test_log_pe_debw_and_p_pro_scalar_vs_batch_vs_fused() {
    const K: usize = 77;
    const N: usize = 174;
    // 1) Build generator matrix
    let mut matrix = create_test_matrix(K);

    //    so that all alpha[i] ≥ 0.1 and vary from 0.1 .. 1.0
    let mut alpha = [0.0_f32; N];
    for i in 0..N {
        alpha[i] = (i as f32 / (N as f32)) + 0.1_f32;
    }

    // 3) Construct dummy llrs (all zeros) and choose gamma = 2.5
    let llrs = [0.0_f32; N];
    let gamma = 2.5_f32;
    let k_u8 = K as u8;

    // 4) Initialize the decoder
    let mut decoder = FrameDecoder::new(alpha, gamma, k_u8, &mut matrix, llrs, 2);

    // 5) Create a few TestErrorPatterns to exercise different shapes:
    //    - Zero positions
    //    - One position
    //    - Three positions
    //    - Eight positions (the maximum)
    let mut teps: Vec<TestErrorPattern> = Vec::new();

    // (a) zero positions
    {
        let mut t = TestErrorPattern::new();
        t.set_num_positions(0);
        *t.positions_mut() = [0; 8]; // doesn't matter
        t.set_log_pe_value(decoder.calculate_log_pe(&t));
        teps.push(t);
    }

    // (b) one position (e.g. position = 5)
    {
        let mut t = TestErrorPattern::new();
        t.set_num_positions(1);
        t.positions_mut()[0] = 5;
        t.set_log_pe_value(decoder.calculate_log_pe(&t));
        teps.push(t);
    }

    // (c) three positions (e.g. [2, 10, 50])
    {
        let mut t = TestErrorPattern::new();
        t.set_num_positions(3);
        t.positions_mut()[0] = 2;
        t.positions_mut()[1] = 10;
        t.positions_mut()[2] = 50;
        t.set_log_pe_value(decoder.calculate_log_pe(&t));
        teps.push(t);
    }

    // (d) eight positions (e.g. [0, 1, 2, 3, 4, 5, 6, 7])
    {
        let mut t = TestErrorPattern::new();
        t.set_num_positions(8);
        for i in 0..8 {
            t.positions_mut()[i] = i as u8;
        }
        t.set_log_pe_value(decoder.calculate_log_pe(&t));
        teps.push(t);
    }

    // 6) For each TEP, compare scalar vs. batch vs. fused outputs:
    for tep in &teps {
        // 6a) Scalar `log_pe`:
        let log_pe_scalar = decoder.calculate_log_pe(tep);

        // 6b) Scalar `debw`:
        let debw_scalar = FrameDecoder::calculate_debw(tep, &decoder.alpha_mrb);

        // 6c) Batch‐of‐4 `log_pe_4` (all four slots point to the same TEP):
        let batch_in = [tep, tep, tep, tep];
        let log_pe_4 = decoder.calculate_log_pe_4(batch_in);
        // Check that each lane matches `log_pe_scalar`
        for i in 0..4 {
            assert!(
                (log_pe_4[i] - log_pe_scalar).abs() < 1e-6,
                "log_pe_4 lane {} = {}, but scalar = {} for TEP {:?}",
                i,
                log_pe_4[i],
                log_pe_scalar,
                tep.positions()
            );
        }

        // 6d) Batch‐of‐4 `debw_4`:
        let debw_4 = FrameDecoder::calculate_debw_4(batch_in, &decoder.alpha_mrb);
        for i in 0..4 {
            assert!(
                (debw_4[i] - debw_scalar).abs() < 1e-6,
                "debw_4 lane {} = {}, but scalar = {} for TEP {:?}",
                i,
                debw_4[i],
                debw_scalar,
                tep.positions()
            );
        }

        // 6e) Scalar `p_pro` (needs both `log_pe_scalar` and `debw_scalar`):
        //     We can call the existing `calculate_p_pro` (which calls debw internally),
        //     or replicate it manually:
        let p_pro_scalar = decoder.calculate_p_pro(tep);

        // 6f) Batch‐of‐4 `p_pro_4` (requiring `log_pe_4`):
        let p_pro_4 = decoder.calculate_p_pro_4(batch_in, &log_pe_4);
        for i in 0..4 {
            assert!(
                (p_pro_4[i] - p_pro_scalar).abs() < 1e-6,
                "p_pro_4 lane {} = {}, but scalar = {} for TEP {:?}",
                i,
                p_pro_4[i],
                p_pro_scalar,
                tep.positions()
            );
        }

        // 6g) Fused “log_pe & p_pro” in one shot:
        let (lp_fused, pp_fused) = decoder.calculate_log_pe_and_p_pro_4(batch_in);
        for i in 0..4 {
            assert!(
                (lp_fused[i] - log_pe_scalar).abs() < 1e-6,
                "Fused log_pe lane {} = {}, but scalar = {} for TEP {:?}",
                i,
                lp_fused[i],
                log_pe_scalar,
                tep.positions()
            );
            assert!(
                (pp_fused[i] - p_pro_scalar).abs() < 1e-6,
                "Fused p_pro lane {} = {}, but scalar = {} for TEP {:?}",
                i,
                pp_fused[i],
                p_pro_scalar,
                tep.positions()
            );
        }
    }
}

/// Evaluates the scalar Q-function for each lane.
fn q_function_4x_scalar(input: &[f32; 4]) -> [f32; 4] {
    [
        crate::pb_osd::probability::q_function(input[0]),
        crate::pb_osd::probability::q_function(input[1]),
        crate::pb_osd::probability::q_function(input[2]),
        crate::pb_osd::probability::q_function(input[3]),
    ]
}

/// Evaluates the exponential-polynomial Q approximation for each lane.
fn q_function_4x_old_approx(input: &[f32; 4]) -> [f32; 4] {
    const A: f32 = Q_PARAM_A;
    const B: f32 = Q_PARAM_B;
    const C: f32 = Q_PARAM_C;
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        let x = input[i];
        let poly = x.mul_add(A * x + B, C);
        out[i] = poly.exp();
    }
    out
}

/// A small “almost equal” comparison for floats
fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

#[test]
fn test_q_function_with_coefs_matches_original() {
    // Pick a few test vectors to exercise positive, negative, zero, large
    let test_inputs: &[[f32; 4]] = &[
        [0.0, 0.0, 0.0, 0.0],
        [0.1, -0.1, 0.5, -0.5],
        [1.0, 2.0, 3.0, -4.0],
        [-1.2345, 0.9876, 5.4321, -6.5432],
        [10.0, -10.0, 0.001, -0.001],
    ];

    // Load the NEON coefficients exactly once
    let coefs = FrameDecoder::load_q_coefs();

    for input in test_inputs.iter() {
        // Compute the “old” way (scalar + exp)
        let expected = q_function_4x_old_approx(input);

        let got = FrameDecoder::q_function_4x_with_coefs(coefs, input);

        // Compare lane by lane with a small tolerance
        for i in 0..4 {
            assert!(
                approx_eq(expected[i], got[i], 1e-6),
                "Mismatch at lane {}: expected {:.9}, got {:.9}",
                i,
                expected[i],
                got[i]
            );
        }
    }
}

#[test]
fn test_q_function_with_random_inputs() {
    // For extra confidence, test a handful of random vectors
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0xdead_beef);

    let coefs = FrameDecoder::load_q_coefs();

    for _ in 0..50 {
        // Generate random floats in [-5.0, 5.0]
        let mut input = [0.0f32; 4];
        for i in 0..4 {
            input[i] = rng.random_range(-5.0..5.0);
        }

        let expected = q_function_4x_old_approx(&input);
        let got = FrameDecoder::q_function_4x_with_coefs(coefs, &input);

        for i in 0..4 {
            assert!(
                approx_eq(expected[i], got[i], 1e-6),
                "Random test: lane {} mismatch: input={:.6}, expected {:.9}, got {:.9}",
                i,
                input[i],
                expected[i],
                got[i]
            );
        }
    }
}

