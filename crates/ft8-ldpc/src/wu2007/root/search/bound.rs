//! Conservative information-distance bounds for first preprocessing.

/// The systematic baseline agrees with every information hard decision, so
/// flipping position i contributes |LLR[i]|. Redundancy costs are nonnegative.
/// The cheapest one/two distinct suffix positions bound every pair completion.
pub(super) struct InformationBound {
    weights: [f64; 91],
    required_first: [f64; 91],
    useful: bool,
    #[cfg(test)]
    suffix: [f64; 92],
}

// Each input to PackedWeightedHammingScorer passes through at most four
// nibble additions, one low+high addition, and 22 chunk-fold additions. With
// round-to-nearest binary32, u = EPSILON/2, its contribution is at least
// (1-u)^27 times its exact value, hence at least (1-27u). Nonnegative sums
// that are subnormal are exact (multiples of the smallest subnormal); overflow
// only increases the result. This factor is smaller than 1-27u. The binary64
// suffix sum is rounded downward. The rejection threshold is then rounded
// upward at every operation, so rejecting cannot exclude a smaller scorer
// result, even with widely separated exponents.
// Keep this proof in sync with the scorer's arithmetic, not its SIMD width.
const ROUNDING_FACTOR: f64 = 1.0 - 64.0 * f32::EPSILON as f64;

#[inline(always)]
fn add_down(a: f64, b: f64) -> f64 {
    (a + b).next_down().max(0.0)
}

impl InformationBound {
    pub(super) fn new(alpha: &[f32], order: usize, distance: f32) -> Option<Self> {
        if !(3..=4).contains(&order)
            || alpha.len() < order
            || alpha.len() > 91
            || !distance.is_finite()
            || distance < 0.0
            || alpha.iter().any(|a| !a.is_finite() || *a < 0.0)
        {
            return None;
        }
        let mut bound = Self {
            weights: [0.0; 91],
            required_first: [f64::INFINITY; 91],
            useful: false,
            #[cfg(test)]
            suffix: [f64::INFINITY; 92],
        };
        let mut suffix = [f64::INFINITY; 92];
        let (mut first, mut second) = (f64::INFINITY, f64::INFINITY);
        for i in (0..alpha.len()).rev() {
            let a = alpha[i] as f64;
            bound.weights[i] = a;
            if a < first {
                second = first;
                first = a;
            } else if a < second {
                second = a;
            }
            suffix[i] = if order == 3 {
                first
            } else {
                add_down(first, second)
            };
        }
        // Rearrange factor*(a0+a1+suffix) >= distance. Rounding every
        // operation toward +infinity makes this a sufficient condition and
        // keeps all rounding work out of the pair loop.
        let target = (distance as f64 / ROUNDING_FACTOR).next_up();
        let mut largest_first = alpha[0] as f64;
        for i1 in 1..alpha.len() - (order - 2) {
            let required = ((target - suffix[i1 + 1]).next_up() - bound.weights[i1]).next_up();
            bound.required_first[i1] = if distance == 0.0 { 0.0 } else { required };
            bound.useful |= largest_first >= bound.required_first[i1];
            largest_first = largest_first.max(bound.weights[i1]);
        }
        #[cfg(test)]
        {
            bound.suffix = suffix;
        }
        Some(bound)
    }

    pub(super) fn can_prune(&self) -> bool {
        self.useful
    }

    #[cfg(test)]
    fn lower(&self, i0: usize, i1: usize) -> f64 {
        let pair = add_down(self.weights[i0], self.weights[i1]);
        (add_down(pair, self.suffix[i1 + 1]) * ROUNDING_FACTOR)
            .next_down()
            .max(0.0)
    }

    #[inline(always)]
    pub(super) fn rejects(&self, i0: usize, i1: usize) -> bool {
        self.weights[i0] >= self.required_first[i1]
    }
}

/// Number of patterns used by exhaustive test oracles.
#[cfg(test)]
fn pattern_count(k: usize, order: usize) -> Option<usize> {
    if order > k {
        return Some(0);
    }
    (0..order).try_fold(1usize, |count, i| {
        count.checked_mul(k - i).map(|n| n / (i + 1))
    })
}

#[cfg(test)]
mod tests;
