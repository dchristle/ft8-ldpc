//! Normal-tail evaluation in f64 with an f32 interface for the decoder.

#[path = "q_coefficients.rs"]
mod coefficients;
use coefficients::{Q_CENTRAL, Q_COEFFICIENTS};

/// Standard-normal upper-tail probability, retaining tails through f64 underflow.
#[inline]
pub(crate) fn q_f64(x: f64) -> f64 {
    half_erfc_f64(x / std::f64::consts::SQRT_2)
}

/// Normal-tail probability with the decoder's f32 scaling and rounding.
#[inline]
pub(crate) fn q_decoder_f32(x: f32) -> f32 {
    let y = (x / 2.0_f32.sqrt()) as f64;
    if y.abs() >= 10.5 {
        return if y < 0.0 { 1.0 } else { 0.0 };
    }
    half_erfc_f64(y) as f32
}

#[inline]
pub(crate) fn half_erfc_f64(y: f64) -> f64 {
    let a = y.abs();
    if a.is_nan() {
        return y;
    }
    if a >= 28.0 {
        return if y < 0.0 { 1.0 } else { 0.0 };
    }
    let tail = if a < 1.0 {
        let z = 2.0 * a * a - 1.0;
        let mut p = Q_CENTRAL[10];
        for i in (0..10).rev() {
            p = p.mul_add(z, Q_CENTRAL[i]);
        }
        (-a).mul_add(p, 0.5)
    } else if a < 10.5 {
        let segment = (2.0 * a) as usize;
        let z = 4.0 * a - (2 * segment + 1) as f64;
        let c = &Q_COEFFICIENTS[segment - 2];
        let mut p = c[10];
        for i in (0..10).rev() {
            p = p.mul_add(z, c[i]);
        }
        p * (-a * a).exp()
    } else {
        far_tail(a)
    };
    if y < 0.0 { 1.0 - tail } else { tail }
}

#[inline]
fn far_tail(a: f64) -> f64 {
    // Repeated integration by parts gives alternating terms
    // (-1)^n (2n-1)!! / (2*a*a)^n for exp(a*a)*erfc(a)*a*sqrt(pi).
    let inverse_square = 0.5 / (a * a);
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1..32 {
        term *= -((2 * n - 1) as f64) * inverse_square;
        sum += term;
        if term.abs() <= 0.25 * f64::EPSILON * sum.abs() {
            break;
        }
    }
    let square = a * a;
    let square_error = a.mul_add(a, -square);
    let exponential = (-square).exp() * (1.0 - square_error);
    (0.25 * std::f64::consts::FRAC_2_SQRT_PI / a) * sum * exponential
}

#[cfg(test)]
mod tests {
    use super::{q_decoder_f32, q_f64};

    #[test]
    fn independent_reference_covers_normal_and_subnormal_tails() {
        let mut count = 0;
        let mut max_relative_fit = 0.0f64;
        let mut max_relative_extended = 0.0f64;
        let mut max_subnormal_ulps = 0u64;
        for row in include_bytes!("../../tools/pb_osd/q64_reference.bin").chunks_exact(24) {
            let x = f64::from_le_bytes(row[..8].try_into().unwrap());
            let reference = f64::from_le_bytes(row[8..16].try_into().unwrap());
            let negative = f64::from_le_bytes(row[16..24].try_into().unwrap());
            let actual = q_f64(x);
            let relative = if x <= 14.0 { 1e-13 } else { 1e-12 };
            let allowance = (relative * reference).max(f64::from_bits(1));
            assert!((actual-reference).abs() <= allowance, "x={x}, got {actual:e}, expected {reference:e}");
            assert!((q_f64(-x)-negative).abs() <= 1e-13 * negative);
            if reference.is_normal() {
                let error = ((actual-reference)/reference).abs();
                if x <= 14.0 {
                    max_relative_fit = max_relative_fit.max(error);
                } else {
                    max_relative_extended = max_relative_extended.max(error);
                }
            } else {
                max_subnormal_ulps = max_subnormal_ulps.max(actual.to_bits().abs_diff(reference.to_bits()));
            }
            count += 1;
        }
        println!("{count} independent references: max relative error <=14: {max_relative_fit:e}; larger normal tails: {max_relative_extended:e}; subnormal error: {max_subnormal_ulps} ulp");
        assert!(count > 4000);
        assert!(q_f64(16.0) > 0.0);
        assert!(q_f64(38.0) > 0.0);
        assert_eq!(q_decoder_f32(16.0), 0.0);
    }

    #[test]
    fn endpoints_and_sampled_domain_are_monotone() {
        assert_eq!(q_f64(0.0), 0.5);
        assert_eq!(q_f64(-0.0), 0.5);
        assert!(q_f64(f64::NAN).is_nan());
        assert_eq!(q_f64(f64::INFINITY), 0.0);
        assert_eq!(q_f64(f64::NEG_INFINITY), 1.0);
        let mut previous = 0.5;
        for i in 0..=100_000 {
            let x = i as f64 / 2500.0;
            let q = q_f64(x);
            assert!(q >= 0.0 && q <= previous, "nonmonotone at {x}");
            previous = q;
        }
        for a in [1.0f32, 10.5, 28.0] {
            let x = a * 2.0_f32.sqrt();
            let mut previous = 1.0f32;
            for bits in x.to_bits()-4..=x.to_bits()+4 {
                let q = q_decoder_f32(f32::from_bits(bits));
                assert!(q <= previous);
                previous = q;
            }
        }
    }

    #[test]
    fn decoder_wrapper_preserves_f32_argument_scaling() {
        assert_eq!(q_decoder_f32(0.75).to_bits(), 0x3e68_10ff);
        assert_ne!(q_decoder_f32(0.75).to_bits(), (q_f64(0.75) as f32).to_bits());
    }
}
