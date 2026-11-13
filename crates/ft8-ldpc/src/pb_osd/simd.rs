//! ARM NEON exponential and Q-function approximations for PB-OSD.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use std::mem::transmute;

/// Approximates e^x for four lanes using range reduction and an Estrin polynomial.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn exp4_f32(x: float32x4_t) -> float32x4_t {
    const INV_LN2: f32 = std::f32::consts::LOG2_E;
    const LN2: f32 = std::f32::consts::LN_2;
    let p = super::exp_coefficients::EXP_COEFFICIENTS_ESTRIN;

    // Truncating x / ln(2) leaves the polynomial argument in [-ln(2), ln(2)].
    let inv_ln2_vec = vdupq_n_f32(INV_LN2);
    let ln2_vec = vdupq_n_f32(LN2);

    let fx = vmulq_f32(x, inv_ln2_vec);
    let n = vcvtq_s32_f32(fx);

    let n_f = vcvtq_f32_s32(n);

    let x_reduced = vsubq_f32(x, vmulq_f32(n_f, ln2_vec));

    // Coefficients are stored in Estrin evaluation order.
    let p0 = vdupq_n_f32(p[0]);
    let p1 = vdupq_n_f32(p[1]);
    let p2 = vdupq_n_f32(p[2]);
    let p3 = vdupq_n_f32(p[3]);
    let p4 = vdupq_n_f32(p[4]);
    let p5 = vdupq_n_f32(p[5]);
    let p6 = vdupq_n_f32(p[6]);
    let p7 = vdupq_n_f32(p[7]);

    let a = vfmaq_f32(p0, p4, x_reduced);
    let b = vfmaq_f32(p2, p6, x_reduced);
    let c = vfmaq_f32(p1, p5, x_reduced);
    let d = vfmaq_f32(p3, p7, x_reduced);

    let x2 = vmulq_f32(x_reduced, x_reduced);

    let a = vfmaq_f32(a, b, x2);
    let c = vfmaq_f32(c, d, x2);

    let x4 = vmulq_f32(x2, x2);

    let poly = vfmaq_f32(a, c, x4);

    // A zero fraction and biased exponent encode 2^n for normal exponents.

    let offset127 = vdupq_n_s32(127);
    let mut n_biased = vaddq_s32(n, offset127);

    // The endpoint exponents encode zero and positive infinity.
    n_biased = vmaxq_s32(n_biased, vdupq_n_s32(0));
    n_biased = vminq_s32(n_biased, vdupq_n_s32(255));

    let n_shifted = vshlq_n_s32(n_biased, 23);
    let pow2n = vreinterpretq_f32_s32(n_shifted);

    vmulq_f32(poly, pow2n)
}

/// Scalar fallback for exp4_f32 on non-ARM architectures
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn exp4_f32_scalar(values: &[f32; 4]) -> [f32; 4] {
    [
        values[0].exp(),
        values[1].exp(),
        values[2].exp(),
        values[3].exp(),
    ]
}

/// Broadcasts Q-function polynomial coefficients for reuse across NEON evaluations.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn load_q_coefs() -> (float32x4_t, float32x4_t, float32x4_t) {
    use crate::pb_osd::probability::{Q_PARAM_A, Q_PARAM_B, Q_PARAM_C};
    unsafe {
        let a_vec: float32x4_t = vdupq_n_f32(Q_PARAM_A);
        let b_vec: float32x4_t = vdupq_n_f32(Q_PARAM_B);
        let c_vec: float32x4_t = vdupq_n_f32(Q_PARAM_C);
        (a_vec, b_vec, c_vec)
    }
}

/// Evaluates exp(C + x * (A * x + B)) for four lanes with preloaded coefficients.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn q_function_4x_with_coefs(
    coefs: (float32x4_t, float32x4_t, float32x4_t),
    input: &[f32; 4],
) -> [f32; 4] {
    let (a_vec, b_vec, c_vec) = coefs;
    unsafe {
        let x = vld1q_f32(input.as_ptr());

        let ax = vmulq_f32(a_vec, x);
        let ax_plus_b = vaddq_f32(ax, b_vec);
        let x_times_ax_plus_b = vmulq_f32(x, ax_plus_b);
        let poly = vaddq_f32(c_vec, x_times_ax_plus_b);

        let result = exp4_f32(poly);

        transmute(result)
    }
}
