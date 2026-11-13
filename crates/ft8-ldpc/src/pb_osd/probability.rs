//! Probability calculations for PB-OSD:
//! - Error probabilities P(e) for Test Error Patterns
//! - Promising probabilities P_pro for search space pruning  
//! - Success probabilities P_suc for early termination
//! - Q-function approximations for normal distribution calculations
//!
//! The probabilities follow C. Yue et al., "Probability-Based Ordered-Statistics
//! Decoding for Short Block Codes", IEEE Communications Letters, 2021. The
//! Q-function approximation constants are from M. López-Benítez and F. Casadevall,
//! IEEE Transactions on Communications, vol. 59, no. 4, 2011.

use super::tep::TestErrorPattern;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
use ft8_ldpc_core::protocol::N;
#[cfg(target_arch = "aarch64")]
use std::mem::transmute;

/// Constants for Q-function approximation
/// Source: López-Benítez & Casadevall (2011) min-SSE
pub const Q_PARAM_A: f32 = -0.3842;
pub const Q_PARAM_B: f32 = -0.7640;
pub const Q_PARAM_C: f32 = -0.6964;

/// Complementary standard-normal probability with the decoder's f32 input scaling.
#[inline]
pub(crate) fn q_function(x: f32) -> f32 {
    crate::pb_osd::normal::q_decoder_f32(x)
}

/// Evaluate erfc(y)/2 with locally derived Remez polynomials before f32 rounding.
#[inline]
pub(super) fn q_scaled(y: f64) -> f64 {
    crate::pb_osd::normal::half_erfc_f64(y)
}

/// Fast approximation of the Q-function using López-Benítez & Casadevall approximation
/// Note: This approximation has accuracy issues for negative inputs and should be avoided
#[inline]
pub(crate) fn q_function_fast_approx(x: f32) -> f32 {
    x.mul_add(Q_PARAM_A * x + Q_PARAM_B, Q_PARAM_C).exp()
}

/// Evaluate the Q-function for four values.
#[inline]
pub(crate) fn q_function_4x(input: &[f32; 4]) -> [f32; 4] {
    [
        q_function(input[0]),
        q_function(input[1]),
        q_function(input[2]),
        q_function(input[3]),
    ]
}

/// Fast vectorized Q-function approximation for 4 values simultaneously
/// Note: This uses the López-Benítez & Casadevall approximation with accuracy issues
#[inline]
fn q_function_4x_fast_approx_poly(input: &[f32; 4]) -> [f32; 4] {
    const A: f32 = Q_PARAM_A;
    const B: f32 = Q_PARAM_B;
    const C: f32 = Q_PARAM_C;

    let mut poly_arr = [0.0; 4];
    for i in 0..4 {
        let x = input[i];
        poly_arr[i] = x.mul_add(A * x + B, C);
    }
    poly_arr
}

/// Fast vectorized Q-function approximation — NEON exp path
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn q_function_4x_fast_approx(input: &[f32; 4]) -> [f32; 4] {
    use crate::pb_osd::simd::exp4_f32;
    let poly_arr = q_function_4x_fast_approx_poly(input);
    unsafe {
        let x_vec = vld1q_f32(poly_arr.as_ptr());
        let y_vec = exp4_f32(x_vec);
        transmute(y_vec)
    }
}

/// Fast vectorized Q-function approximation — scalar exp path
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub(crate) fn q_function_4x_fast_approx(input: &[f32; 4]) -> [f32; 4] {
    let poly_arr = q_function_4x_fast_approx_poly(input);
    [
        poly_arr[0].exp(),
        poly_arr[1].exp(),
        poly_arr[2].exp(),
        poly_arr[3].exp(),
    ]
}

/// Probability calculation context for a decoder instance
#[derive(Clone)]
pub struct ProbabilityCalculator {
    /// Logarithm of bit error probabilities
    pub log_p: [f32; N],
    /// Logarithm of (1 - bit error probabilities)  
    pub log_1_minus_p: [f32; N],
    /// Sum of all log(1-p) values for efficiency
    pub sum_log_1_minus_p: f32,
    /// Expected value E[α] for normal approximations
    pub e_p_alpha: f32,
    /// Average parity bit error probability
    pub parity_p_avg: f32,
    /// Reliability values in MRB order
    pub alpha_mrb: [f32; N],
    /// Channel hard decisions in MRB order.
    pub c0_mrb: [bool; N],
    /// Encoded baseline codeword for P_suc calculations.
    pub baseline_codeword_bool: [bool; N],
    /// Message length
    pub k: u8,
    /// Number of parity bits
    pub nmk: usize,
    /// Log of 2^(n-k) for probability normalization
    pub log_two_pow_nmk: f32,
    /// Minimum weighted distance for comparisons
    pub d_min_w: f32,
}

impl ProbabilityCalculator {
    /// Calculate logarithm of error probability P(e) for a Test Error Pattern
    #[inline]
    pub fn calculate_log_pe(&self, tep: &TestErrorPattern) -> f32 {
        let mut log_pe = self.sum_log_1_minus_p;

        for i in 0..tep.num_positions() {
            let pos = tep.positions()[i as usize] as usize;
            if pos < self.k as usize {
                // Remove log(1-p[pos]) and add log(p[pos])
                log_pe -= unsafe { *self.log_1_minus_p.get_unchecked(pos) };
                log_pe += unsafe { *self.log_p.get_unchecked(pos) };
            }
        }

        log_pe
    }

    /// Calculate log P(e) for 4 TEPs simultaneously
    #[inline]
    pub fn calculate_log_pe_4(&self, teps: &[TestErrorPattern; 4]) -> [f32; 4] {
        let mut results = [self.sum_log_1_minus_p; 4];

        for tep_idx in 0..4 {
            let tep = &teps[tep_idx];
            for i in 0..tep.num_positions() {
                let pos = tep.positions()[i as usize] as usize;
                if pos < self.k as usize {
                    results[tep_idx] -= unsafe { *self.log_1_minus_p.get_unchecked(pos) };
                    results[tep_idx] += unsafe { *self.log_p.get_unchecked(pos) };
                }
            }
        }

        results
    }

    /// Calculate both log P(e) and P_pro for 4 TEPs simultaneously
    #[inline]
    pub fn calculate_log_pe_and_p_pro_4(
        &self,
        teps: &[TestErrorPattern; 4],
    ) -> ([f32; 4], [f32; 4]) {
        let log_pe_values = self.calculate_log_pe_4(teps);
        let p_pro_values = self.calculate_p_pro_4(teps);
        (log_pe_values, p_pro_values)
    }

    /// Calculate promising probability P_pro for a Test Error Pattern
    #[inline]
    pub fn calculate_p_pro(&self, tep: &TestErrorPattern) -> f32 {
        let log_pe = self.calculate_log_pe(tep);
        let pe = log_pe.exp();

        // Calculate d_eBW (expected weighted distance for this TEP)
        let mut d_ebw = 0.0;
        for i in 0..tep.num_positions() {
            let pos = tep.positions()[i as usize] as usize;
            if pos < self.k as usize {
                d_ebw += unsafe { *self.alpha_mrb.get_unchecked(pos) };
            }
        }

        // β = max parity errors for TEP to beat current best
        let beta = ((self.d_min_w - d_ebw) / self.e_p_alpha).floor();

        if beta < 0.0 {
            return 0.0;
        }

        // Normal approximation arguments
        let n = self.nmk as f32;
        let p = self.parity_p_avg;

        let sqrt_np_1minusp = (n * p * (1.0 - p)).sqrt();
        let sqrt_n = n.sqrt();

        let arg1 = (beta + 0.5 - n * p) / sqrt_np_1minusp;
        let arg2 = (2.0 * beta + 1.0 - n) / sqrt_n;

        let q1 = q_function(arg1);
        let q2 = q_function(arg2);

        pe * (1.0 - q1) + (1.0 - pe) * (1.0 - q2)
    }

    /// Calculate P_pro for 4 TEPs simultaneously
    #[inline]
    pub fn calculate_p_pro_4(&self, teps: &[TestErrorPattern; 4]) -> [f32; 4] {
        let log_pe_values = self.calculate_log_pe_4(teps);
        let mut pe_values = [0.0f32; 4];
        let mut d_ebw_values = [0.0f32; 4];

        // Calculate P(e) and d_eBW for each TEP
        for (i, &log_pe) in log_pe_values.iter().enumerate() {
            pe_values[i] = log_pe.exp();

            let tep = &teps[i];
            for j in 0..tep.num_positions() {
                let pos = tep.positions()[j as usize] as usize;
                if pos < self.k as usize {
                    d_ebw_values[i] += unsafe { *self.alpha_mrb.get_unchecked(pos) };
                }
            }
        }

        // Calculate beta values
        let mut beta_values = [0.0f32; 4];
        for i in 0..4 {
            beta_values[i] = ((self.d_min_w - d_ebw_values[i]) / self.e_p_alpha).floor();
        }

        // Calculate Q-function arguments
        let n = self.nmk as f32;
        let p = self.parity_p_avg;
        let sqrt_np_1minusp = (n * p * (1.0 - p)).sqrt();
        let sqrt_n = n.sqrt();

        let mut arg1_values = [0.0f32; 4];
        let mut arg2_values = [0.0f32; 4];

        for i in 0..4 {
            arg1_values[i] = (beta_values[i] + 0.5 - n * p) / sqrt_np_1minusp;
            arg2_values[i] = (2.0 * beta_values[i] + 1.0 - n) / sqrt_n;
        }

        let q1_values = q_function_4x(&arg1_values);
        let q2_values = q_function_4x(&arg2_values);

        let mut results = [0.0f32; 4];
        for i in 0..4 {
            if beta_values[i] < 0.0 {
                results[i] = 0.0;
            } else {
                results[i] = pe_values[i] * (1.0 - q1_values[i])
                    + (1.0 - pe_values[i]) * (1.0 - q2_values[i]);
            }
        }

        results
    }

    /// Calculate success probability P_suc for early termination
    ///
    /// This calculates the probability that the given Test Error Pattern (TEP)
    /// will lead to the maximum likelihood (ML) codeword. Used for early termination
    /// in the PB-OSD algorithm.
    #[inline]
    pub fn calculate_p_suc_e(&self, tep: &TestErrorPattern, cw: &[bool; 174]) -> f32 {
        // Calculate difference vector d_e = cw XOR c0_mrb (baseline codeword)
        let d_e = self.calculate_d_e(cw);

        // Extract error probability from TEP
        let log_pe = tep.log_pe_value();
        let pe = log_pe.exp();

        // Calculate parity probability product for positions k..N-1
        let mut log_pi_product = 0.0;
        for i in (self.k as usize)..N {
            let d_e_i_is_one = d_e[i];
            let log_prob = if d_e_i_is_one {
                unsafe { *self.log_p.get_unchecked(i) }
            } else {
                unsafe { *self.log_1_minus_p.get_unchecked(i) }
            };
            log_pi_product += log_prob;
        }

        // Calculate stable log ratio: log((1-P(e))/P(e))
        let stable_log_ratio = if pe >= 1.0 - f32::EPSILON {
            // Special case: pe very close to 1, log(1-pe) would be unstable
            -20.0 // Very negative value, meaning very low success probability
        } else {
            (-pe).ln_1p() - log_pe
        };

        // Final P_suc calculation using the formula:
        // P_suc = 1 / (1 + [(1-P(e)) * 2^(k-n)] / [P(e) * Pr(d_{e,P})])
        let log_2_pow_k_minus_n = -self.log_two_pow_nmk; // ln(2^(k-n))
        let diff = stable_log_ratio + log_2_pow_k_minus_n - log_pi_product;
        let p_suc_e = 1.0 / (1.0 + diff.exp());

        // Clamp to valid probability range and check for stability
        let result = p_suc_e.clamp(0.0, 1.0);

        debug_assert!(
            result.is_finite() && !result.is_nan(),
            "P_suc calculation resulted in invalid value: {result}, inputs: pe={pe}, log_pi_product={log_pi_product}, diff={diff}"
        );

        result
    }

    /// Calculate difference vector d_e = cw XOR baseline_codeword (properly encoded baseline)
    #[inline]
    fn calculate_d_e(&self, cw: &[bool; 174]) -> [bool; 174] {
        let mut d_e = [false; 174];
        for i in 0..174 {
            d_e[i] = cw[i] ^ self.baseline_codeword_bool[i];
        }
        d_e
    }
}
