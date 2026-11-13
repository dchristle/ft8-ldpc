//! Exact nibble sums and chunk-ordered binary32 folds.
use crate::wu2007::packed::*;
use ft8_ldpc_core::protocol::N;

const CHUNKS: usize = 22;
const TABLE_LEN: usize = CHUNKS * 256;

#[derive(Clone)]
pub struct PackedWeightedHammingScorer {
    table: Box<[f32; TABLE_LEN]>,
    reference: PackedCodeword,
}

impl PackedWeightedHammingScorer {
    pub fn new(reliabilities: &[f32; N], reference_codeword: PackedCodeword) -> Self {
        let mut scorer = Self {
            table: vec![0.; TABLE_LEN].into_boxed_slice().try_into().unwrap(),
            reference: reference_codeword,
        };
        scorer.update(reliabilities, reference_codeword);
        scorer
    }

    pub fn update(&mut self, reliabilities: &[f32; N], reference_codeword: PackedCodeword) {
        self.reference = reference_codeword;
        for c in 0..CHUNKS {
            let mut low = [0f32; 16];
            let mut high = [0f32; 16];
            for bit in 0..4 {
                for p in 0..16 {
                    if p & (1 << bit) != 0 {
                        low[p] += reliabilities[8 * c + bit];
                        if 8 * c + 4 + bit < N {
                            high[p] += reliabilities[8 * c + 4 + bit];
                        }
                    }
                }
            }
            for (h, &sum) in high.iter().enumerate() {
                let row = &mut self.table[256 * c + 16 * h..256 * c + 16 * h + 16];
                for (chunk, low) in row
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(low.as_chunks::<4>().0)
                {
                    *chunk = low.map(|value| value + sum);
                }
            }
        }
    }

    #[cfg(test)]
    pub fn make_diff(&self, codeword: &PackedCodeword) -> PackedCodeword {
        xor_codewords(codeword, &self.reference)
    }

    #[inline]
    pub fn distance_from_diff(&self, diff: &PackedCodeword) -> f32 {
        let mut distance = 0f32;
        for c in 0..CHUNKS {
            let byte = ((diff[c / 8] >> ((c % 8) * 8)) & 255) as usize;
            distance += self.table[256 * c + byte];
        }
        distance
    }

    #[cfg(test)]
    pub fn calculate_weighted_hamming_f32(&self, codeword: &PackedCodeword) -> f32 {
        self.distance_from_diff(&self.make_diff(codeword))
    }

    pub fn calculate_batch_weighted_hamming_f32_from_diffs_portable_simd(
        &self,
        diffs: &[PackedCodeword; 4],
    ) -> [f32; 4] {
        let mut distances = [0f32; 4];
        // The 22 steps are expanded so word and byte offsets are constants;
        // each lane still performs the ascending-chunk left fold.
        macro_rules! chunks {
            ($($c:literal),* $(,)?) => {$(
                let values: [f32; 4] = std::array::from_fn(|i| {
                    let byte = ((diffs[i][$c / 8] >> (($c % 8) * 8)) & 255) as usize;
                    self.table[256 * $c + byte]
                });
                distances = std::array::from_fn(|i| distances[i] + values[i]);
            )*};
        }
        chunks!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21);
        distances
    }
}

/// Computes every table entry independently without shared partial sums.
#[cfg(test)]
pub(crate) fn fill_reference(reliabilities: &[f32; N]) -> Vec<f32> {
    let mut table = vec![0f32; TABLE_LEN];
    for c in 0..CHUNKS {
        for p in 0..256 {
            let mut low = 0f32;
            for j in 0..4 {
                if p & (1 << j) != 0 {
                    low += reliabilities[8 * c + j];
                }
            }
            let mut high = 0f32;
            for j in 0..4 {
                if p & (1 << (4 + j)) != 0 && 8 * c + 4 + j < N {
                    high += reliabilities[8 * c + 4 + j];
                }
            }
            table[256 * c + p] = low + high;
        }
    }
    table
}

/// Computes four independent single-candidate folds.
#[cfg(test)]
pub(crate) fn batch_reference(
    scorer: &PackedWeightedHammingScorer,
    diffs: &[PackedCodeword; 4],
) -> [f32; 4] {
    diffs.map(|diff| scorer.distance_from_diff(&diff))
}

#[cfg(test)]
impl PackedWeightedHammingScorer {
    /// Refills the table with independently computed entries.
    pub(crate) fn update_reference(
        &mut self,
        reliabilities: &[f32; N],
        reference_codeword: PackedCodeword,
    ) {
        self.reference = reference_codeword;
        self.table.copy_from_slice(&fill_reference(reliabilities));
    }
}

#[cfg(test)]
pub(crate) mod test_inputs {
    //! Deterministic generators shared by the smoke tests.
    use super::*;

    pub struct SplitMix64(pub u64);
    impl SplitMix64 {
        pub fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        /// Uniform value in [0, 1) with 24 random bits.
        pub fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
        pub fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Reliability vectors covering zeros, ties, wide magnitudes and partial final chunks.
    pub fn alpha_vectors() -> Vec<[f32; N]> {
        let mut rng = SplitMix64(0x7a1f_0001);
        let mut out: Vec<[f32; N]> = Vec::new();
        out.push([0.; N]);
        out.push([1.5; N]);
        out.push([0.1; N]);
        out.push(std::array::from_fn(|i| (1u32 << (i % 20)) as f32));
        for _ in 0..3 {
            out.push(std::array::from_fn(|_| rng.unit() * 7.3 + 0.01));
        }
        out.push(std::array::from_fn(|_| rng.below(5) as f32));
        out.push(std::array::from_fn(|_| rng.below(9) as f32 / 4.));
        for _ in 0..3 {
            out.push(std::array::from_fn(|_| {
                let exponent = rng.below(41) as i32 - 20;
                (0.5 + rng.unit() * 0.5) * (exponent as f32).exp2()
            }));
        }
        for _ in 0..2 {
            out.push(std::array::from_fn(|_| {
                if rng.below(5) == 0 {
                    0.
                } else {
                    rng.unit() * 4.1 + 0.2
                }
            }));
        }
        let mut zero_tail: [f32; N] = std::array::from_fn(|_| rng.unit() * 3.3 + 0.1);
        zero_tail[172] = 0.;
        zero_tail[173] = 0.;
        out.push(zero_tail);
        for _ in 0..3 {
            // Decreasing within the information part and the redundancy part.
            let mut v: [f32; N] = std::array::from_fn(|_| rng.unit() * 6.1 + 0.05);
            v[..91].sort_by(|a, b| b.total_cmp(a));
            v[91..].sort_by(|a, b| b.total_cmp(a));
            out.push(v);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::test_inputs::*;
    use super::*;

    /// Compares optimized table entries and scorer reuse with independently computed entries.
    #[test]
    fn fill_matches_the_scalar_reference_and_reuse_leaks_nothing() {
        let alphas = alpha_vectors();
        let mut reused = PackedWeightedHammingScorer::new(&[0.; N], [0; 3]);
        let allocation = reused.table.as_ptr();
        let mut running_sum_differs = false;
        let mut saw_zero = false;
        let mut saw_zero_tail = false;
        for alpha in &alphas {
            let expected = fill_reference(alpha);
            let reference_word = [3, 5, 7];
            let fresh = PackedWeightedHammingScorer::new(alpha, reference_word);
            reused.update(alpha, reference_word);
            assert_eq!(reused.table.as_ptr(), allocation);
            for scorer in [&fresh, &reused] {
                assert_eq!(scorer.reference, reference_word);
                for (i, (&got, &want)) in scorer.table.iter().zip(&expected).enumerate() {
                    assert_eq!(got.to_bits(), want.to_bits(), "entry {i}");
                }
            }
            running_sum_differs |= (0..CHUNKS).any(|c| {
                (0..256).any(|p| {
                    let running = (0..8)
                        .filter(|&j| p & (1 << j) != 0 && 8 * c + j < N)
                        .fold(0f32, |sum, j| sum + alpha[8 * c + j]);
                    running.to_bits() != expected[256 * c + p].to_bits()
                })
            });
            saw_zero |= alpha.iter().all(|&a| a == 0.);
            saw_zero_tail |= alpha[172] == 0. && alpha[173] == 0. && alpha[0] != 0.;
        }
        assert!(alphas.len() >= 16);
        assert!(
            running_sum_differs,
            "the inputs cannot detect a running-sum fill"
        );
        assert!(saw_zero && saw_zero_tail);
    }

    /// Compares the four-lane pass with four single folds.
    #[test]
    fn four_lane_pass_matches_four_single_folds() {
        let mut rng = SplitMix64(0x7a1f_0006);
        let mut all_different = false;
        for alpha in alpha_vectors() {
            let scorer = PackedWeightedHammingScorer::new(&alpha, [0; 3]);
            let dense = |rng: &mut SplitMix64| -> PackedCodeword {
                [rng.next(), rng.next(), rng.next() & low_mask(N - 128)]
            };
            let single = |bit: usize| {
                let mut diff = [0; 3];
                set_codeword_bit(&mut diff, bit);
                diff
            };
            let mut batches: Vec<[PackedCodeword; 4]> = vec![
                [[0; 3]; 4],
                [single(0), single(63), single(64), single(127)],
                [single(128), single(173), [0; 3], single(1)],
            ];
            batches.push(std::array::from_fn(|_| dense(&mut rng)));
            let shared = dense(&mut rng);
            batches.push([shared, dense(&mut rng), shared, dense(&mut rng)]);
            for diffs in batches {
                let got =
                    scorer.calculate_batch_weighted_hamming_f32_from_diffs_portable_simd(&diffs);
                let want = batch_reference(&scorer, &diffs);
                assert_eq!(got.map(f32::to_bits), want.map(f32::to_bits));
                all_different |=
                    (0..4).all(|a| (a + 1..4).all(|b| got[a].to_bits() != got[b].to_bits()));
            }
        }
        assert!(all_different);
    }

    #[test]
    fn make_diff_and_weighted_distance_use_the_stored_reference() {
        let alpha: [f32; N] = std::array::from_fn(|i| (i % 7 + 1) as f32 / 8.);
        let scorer = PackedWeightedHammingScorer::new(&alpha, [1, 2, 3]);
        let x = [5, 6, 7];
        assert_eq!(scorer.make_diff(&x), [4, 4, 4]);
        assert_eq!(
            scorer.calculate_weighted_hamming_f32(&x).to_bits(),
            scorer.distance_from_diff(&[4, 4, 4]).to_bits()
        );
        let zero = PackedWeightedHammingScorer::new(&[0.; N], [0; 3]);
        assert_eq!(zero.distance_from_diff(&[0; 3]).to_bits(), 0f32.to_bits());
    }
}
