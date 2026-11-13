//! Weighted Hamming and test-error-pattern distances for PB-OSD.
//!
//! `OptimizedWeightedHammingTables` is the packed table scorer: 22 lookup tables of
//! 256 `f32` entries, one per 8-bit slice of the 174-bit codeword, summed in a fixed order.

use super::tep::TestErrorPattern;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
use ft8_ldpc_core::protocol::N;

/// Distance calculation context containing precomputed values
pub struct DistanceCalculator {
    /// Reliability values in MRB order
    pub alpha_mrb: [f32; N],
    /// Baseline codeword in bit-packed format
    pub c0_mrb_bitpacked: [u64; 3],
}

impl DistanceCalculator {
    /// Create a new distance calculator with precomputed values
    pub fn new(alpha_mrb: [f32; N], c0_mrb_bitpacked: [u64; 3]) -> Self {
        Self {
            alpha_mrb,
            c0_mrb_bitpacked,
        }
    }

    /// Calculate Distance-Enhanced Bit-Weighted (DEBW) distance for a single TEP
    ///
    /// This sums the reliability values (alpha) for all error positions in the TEP.
    #[inline]
    pub fn calculate_debw(&self, tep: &TestErrorPattern) -> f32 {
        let mut distance = 0.0;
        for i in 0..tep.num_positions() {
            let pos = tep.positions()[i as usize] as usize;
            distance += unsafe { *self.alpha_mrb.get_unchecked(pos) };
        }
        distance
    }

    /// Calculate DEBW distance for 4 TEPs simultaneously (vectorized)
    #[inline]
    pub fn calculate_debw_4(&self, teps: [&TestErrorPattern; 4]) -> [f32; 4] {
        let mut distances = [0.0f32; 4];

        for tep_idx in 0..4 {
            let tep = teps[tep_idx];
            for i in 0..tep.num_positions() {
                let pos = tep.positions()[i as usize] as usize;
                distances[tep_idx] += unsafe { *self.alpha_mrb.get_unchecked(pos) };
            }
        }

        distances
    }

    /// Calculates DEBW with ARM NEON gathers.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub fn calculate_debw_simd(&self, tep: &TestErrorPattern) -> f32 {
        if tep.num_positions() == 0 {
            return 0.0;
        }

        unsafe {
            let mut sum_vec = vdupq_n_f32(0.0);
            let positions = tep.positions();

            // Process up to 4 positions at a time with SIMD
            let full_groups = (tep.num_positions() / 4) as usize;
            for group in 0..full_groups {
                let base_idx = group * 4;

                // Load 4 positions and gather corresponding alpha values
                let mut alpha_values = [0.0f32; 4];
                for i in 0..4 {
                    let pos = positions[base_idx + i] as usize;
                    alpha_values[i] = *self.alpha_mrb.get_unchecked(pos);
                }

                let alpha_vec = vld1q_f32(alpha_values.as_ptr());
                sum_vec = vaddq_f32(sum_vec, alpha_vec);
            }

            // Handle remaining positions (scalar)
            let mut scalar_sum = vaddvq_f32(sum_vec);
            let remaining_start = full_groups * 4;
            for i in remaining_start..(tep.num_positions() as usize) {
                let pos = positions[i] as usize;
                scalar_sum += *self.alpha_mrb.get_unchecked(pos);
            }

            scalar_sum
        }
    }

    /// Scalar fallback for DEBW SIMD calculation
    #[cfg(not(target_arch = "aarch64"))]
    #[inline]
    pub fn calculate_debw_simd(&self, tep: &TestErrorPattern) -> f32 {
        self.calculate_debw(tep)
    }

    /// Calculate weighted Hamming distance using boolean array representation
    #[inline]
    pub fn calculate_weighted_hamming(&self, cw: &[bool; N], c0_mrb: &[bool; N]) -> f32 {
        let mut distance = 0.0;
        for i in 0..N {
            if cw[i] != c0_mrb[i] {
                distance += unsafe { *self.alpha_mrb.get_unchecked(i) };
            }
        }
        distance
    }

    /// Fast weighted Hamming distance using bit-packed representation
    #[inline]
    pub fn calculate_weighted_hamming_fast(&self, cw: &[u64; 3]) -> f32 {
        Self::calculate_weighted_hamming_bitpacked(cw, &self.c0_mrb_bitpacked, &self.alpha_mrb)
    }

    /// Calculates weighted Hamming distance from packed bit differences.
    #[inline]
    pub fn calculate_weighted_hamming_bitpacked(
        cw: &[u64; 3],
        c0_mrb: &[u64; 3],
        alpha_mrb: &[f32; N],
    ) -> f32 {
        let mut distance = 0.0;

        for chunk_idx in 0..3 {
            let diff = cw[chunk_idx] ^ c0_mrb[chunk_idx];
            if diff != 0 {
                let mut remaining = diff;
                while remaining != 0 {
                    let bit_pos = remaining.trailing_zeros() as usize;
                    let global_pos = chunk_idx * 64 + bit_pos;
                    if global_pos < N {
                        distance += unsafe { *alpha_mrb.get_unchecked(global_pos) };
                    }
                    remaining &= remaining - 1; // Clear the lowest set bit
                }
            }
        }

        distance
    }

    /// Parallel weighted Hamming distance for better instruction-level parallelism
    #[inline]
    pub fn calculate_weighted_hamming_parallel(&self, cw: &[u64; 3]) -> f32 {
        let c0_mrb = &self.c0_mrb_bitpacked;
        let alpha_mrb = &self.alpha_mrb;

        // Process all three chunks in parallel
        let diff0 = cw[0] ^ c0_mrb[0];
        let diff1 = cw[1] ^ c0_mrb[1];
        let diff2 = cw[2] ^ c0_mrb[2];

        let mut distance0 = 0.0;
        let mut distance1 = 0.0;
        let mut distance2 = 0.0;

        // Process chunk 0 (bits 0-63)
        if diff0 != 0 {
            let mut remaining = diff0;
            while remaining != 0 {
                let bit_pos = remaining.trailing_zeros() as usize;
                distance0 += unsafe { *alpha_mrb.get_unchecked(bit_pos) };
                remaining &= remaining - 1;
            }
        }

        // Process chunk 1 (bits 64-127)
        if diff1 != 0 {
            let mut remaining = diff1;
            while remaining != 0 {
                let bit_pos = remaining.trailing_zeros() as usize;
                let global_pos = 64 + bit_pos;
                distance1 += unsafe { *alpha_mrb.get_unchecked(global_pos) };
                remaining &= remaining - 1;
            }
        }

        // Process chunk 2 (bits 128-173)
        if diff2 != 0 {
            let mut remaining = diff2;
            while remaining != 0 {
                let bit_pos = remaining.trailing_zeros() as usize;
                let global_pos = 128 + bit_pos;
                if global_pos < N {
                    distance2 += unsafe { *alpha_mrb.get_unchecked(global_pos) };
                }
                remaining &= remaining - 1;
            }
        }

        distance0 + distance1 + distance2
    }

    /// Calculates weighted Hamming distance with unrolled bit processing.
    #[inline]
    pub fn calculate_weighted_hamming_unrolled(&self, cw: &[u64; 3]) -> f32 {
        let c0_mrb = &self.c0_mrb_bitpacked;
        let alpha_mrb = &self.alpha_mrb;

        let mut distance = 0.0;

        // Unroll the outer loop for all three chunks
        macro_rules! process_chunk {
            ($chunk_idx:expr) => {
                let diff = cw[$chunk_idx] ^ c0_mrb[$chunk_idx];
                if diff != 0 {
                    let mut remaining = diff;
                    let base_offset = $chunk_idx * 64;

                    while remaining != 0 {
                        let bit_pos = remaining.trailing_zeros() as usize;
                        let global_pos = base_offset + bit_pos;

                        if global_pos < N {
                            distance += unsafe { *alpha_mrb.get_unchecked(global_pos) };
                        }

                        remaining &= remaining - 1; // Clear lowest set bit
                    }
                }
            };
        }

        process_chunk!(0);
        process_chunk!(1);
        process_chunk!(2);

        distance
    }

    /// Bitwise weighted Hamming distance calculation
    #[inline]
    pub fn calculate_weighted_hamming_bitpacked_bitwise(
        cw: &[u64; 3],
        c0_mrb: &[u64; 3],
        alpha_mrb: &[f32; N],
    ) -> f32 {
        let mut sum = 0.0;
        let mut bit_index = 0;

        for (&cw_word, &ref_word) in cw.iter().zip(c0_mrb.iter()) {
            let diff = cw_word ^ ref_word;
            for i in 0..64 {
                let bit = (diff >> i) & 1;
                sum += alpha_mrb[bit_index + i] * (bit as f32);
            }
            bit_index += 64;
        }
        sum
    }

    /// Calculates weighted Hamming distance with unrolled bit processing.
    #[inline(always)]
    pub fn calculate_weighted_hamming_bitpacked_bitwise_unrolled(
        cw: &[u64; 3],
        c0_mrb: &[u64; 3],
        alpha_mrb: &[f32; N],
    ) -> f32 {
        let mut sum0 = 0.0;
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;

        // First u64 - all 64 bits are used
        let diff0 = cw[0] ^ c0_mrb[0];
        for i in 0..64 {
            let bit = (diff0 >> i) & 1;
            sum0 += alpha_mrb[i] * (bit as f32);
        }

        // Second u64 - all 64 bits are used
        let diff1 = cw[1] ^ c0_mrb[1];
        for i in 0..64 {
            let bit = (diff1 >> i) & 1;
            sum1 += alpha_mrb[64 + i] * (bit as f32);
        }

        // Third u64 - only first 46 bits are used (174 - 128 = 46)
        let diff2 = cw[2] ^ c0_mrb[2];
        for i in 0..46 {
            let bit = (diff2 >> i) & 1;
            sum2 += alpha_mrb[128 + i] * (bit as f32);
        }

        sum0 + sum1 + sum2
    }
}

// 22 chunks × 256 byte-patterns each.
const LOOKUP_TABLE_LEN: usize = 22 * 256;

pub struct OptimizedWeightedHammingTables {
    // Heap-allocated fixed-size F32 lookup table.  Box<[f32; N]> gives the
    // compiler a compile-time-known length so it can prove index bounds in the
    // hot distance loop without a runtime length load.
    lookup_table_f32: Box<[f32; LOOKUP_TABLE_LEN]>,
    c0_mrb_bitpacked: [u64; 3],
}

impl OptimizedWeightedHammingTables {
    /// Allocate and fill the tables.  Call once when creating the decoder.
    pub fn new(alpha_mrb: &[f32; 174], c0_mrb_bitpacked: [u64; 3]) -> Self {
        let mut tables = Self {
            lookup_table_f32: Box::new([0.0f32; LOOKUP_TABLE_LEN]),
            c0_mrb_bitpacked,
        };
        tables.fill(alpha_mrb);
        tables
    }

    /// Recompute tables in-place for new LLRs / hard-decision vector.
    /// Reuses the existing heap allocation — no allocator calls on the hot path.
    pub fn update(&mut self, alpha_mrb: &[f32; 174], c0_mrb_bitpacked: [u64; 3]) {
        self.c0_mrb_bitpacked = c0_mrb_bitpacked;
        self.fill(alpha_mrb);
    }

    /// Precompute diff = cw XOR c0_mrb_bitpacked, for use with distance_from_diff.
    #[inline(always)]
    pub fn make_diff(&self, cw: &[u64; 3]) -> [u64; 3] {
        [
            cw[0] ^ self.c0_mrb_bitpacked[0],
            cw[1] ^ self.c0_mrb_bitpacked[1],
            cw[2] ^ self.c0_mrb_bitpacked[2],
        ]
    }

    /// Distance from a pre-XOR'd diff array (skips the XOR step of calculate_weighted_hamming_f32).
    #[inline]
    pub fn distance_from_diff(&self, diff: &[u64; 3]) -> f32 {
        let mut total = 0.0f32;
        for chunk in 0..8usize {
            let pattern = ((diff[0] >> (chunk * 8)) & 0xFF) as usize;
            total += self.lookup_table_f32[chunk * 256 + pattern];
        }
        for chunk in 0..8usize {
            let pattern = ((diff[1] >> (chunk * 8)) & 0xFF) as usize;
            total += self.lookup_table_f32[(8 + chunk) * 256 + pattern];
        }
        for chunk in 0..6usize {
            let pattern = ((diff[2] >> (chunk * 8)) & 0xFF) as usize;
            total += self.lookup_table_f32[(16 + chunk) * 256 + pattern];
        }
        total
    }

    fn fill(&mut self, alpha_mrb: &[f32; 174]) {
        // table[p] sums the reliabilities of the bits set in p. Splitting the low
        // nibble into a 16-entry table confines small-stride butterfly work; the
        // high-bit fill and final addition access contiguous entries.
        for chunk in 0..22usize {
            let offset = chunk * 256;
            let alpha_base: usize;
            let n_bits: usize;
            if chunk < 8 {
                alpha_base = chunk * 8;
                n_bits = 8;
            } else if chunk < 16 {
                alpha_base = 64 + (chunk - 8) * 8;
                n_bits = 8;
            } else {
                // Third word (bits 128-173): 6 chunks, last has only 6 valid bits.
                let pos = chunk - 16;
                alpha_base = 128 + pos * 8;
                n_bits = if chunk < 21 { 8 } else { 6 }; // chunk 21: bits 168-173
            }

            // Stage 1: build mini table for bits 0..low_bits (max 4).
            let low_bits = n_bits.min(4);
            let mut mini = [0.0f32; 16];
            for bit in 0..low_bits {
                let a = alpha_mrb[alpha_base + bit];
                let stride = 1usize << bit;
                let mut base = stride;
                while base < 16 {
                    for p in 0..stride {
                        mini[base + p] += a;
                    }
                    base += stride * 2;
                }
            }

            // Stage 2: high-bit butterfly on main table (stride ≥ 16 → NEON-vectorised).
            self.lookup_table_f32[offset..offset + 256].fill(0.0);
            for bit in 4..n_bits {
                let a = alpha_mrb[alpha_base + bit];
                let stride = 1usize << bit; // ≥ 16 entries per contiguous group
                let mut base = stride;
                while base < 256 {
                    for p in 0..stride {
                        self.lookup_table_f32[offset + base + p] += a;
                    }
                    base += stride * 2;
                }
            }

            // Stage 3: tile mini into the main table (sequential, auto-vectorised).
            for p in 0..256usize {
                self.lookup_table_f32[offset + p] += mini[p & 15];
            }
        }
    }

    /// Calculate weighted Hamming distance using F32 lookup tables
    #[inline]
    pub fn calculate_weighted_hamming_f32(&self, cw: &[u64; 3]) -> f32 {
        let diff0 = cw[0] ^ self.c0_mrb_bitpacked[0];
        let diff1 = cw[1] ^ self.c0_mrb_bitpacked[1];
        let diff2 = cw[2] ^ self.c0_mrb_bitpacked[2];

        let mut total = 0.0;

        // Process first word (8 chunks).  chunk ∈ 0..8, pattern ∈ 0..256 →
        // index ≤ 7*256+255 = 2047 < LOOKUP_TABLE_LEN; bounds check elided.
        for chunk in 0..8usize {
            let pattern = ((diff0 >> (chunk * 8)) & 0xFF) as usize;
            total += self.lookup_table_f32[chunk * 256 + pattern];
        }

        // Process second word (8 chunks).  index ≤ 15*256+255 = 4095.
        for chunk in 0..8usize {
            let pattern = ((diff1 >> (chunk * 8)) & 0xFF) as usize;
            total += self.lookup_table_f32[(8 + chunk) * 256 + pattern];
        }

        // Process third word (6 chunks).  index ≤ 21*256+255 = 5631.
        for chunk in 0..6usize {
            let pattern = ((diff2 >> (chunk * 8)) & 0xFF) as usize;
            total += self.lookup_table_f32[(16 + chunk) * 256 + pattern];
        }

        total
    }

    /// Batch process 4 codewords at once using NEON SIMD
    ///
    /// # Safety
    ///
    /// This function requires `neon` CPU feature to be available.
    /// Caller must ensure the target platform supports this feature.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn calculate_batch_weighted_hamming_f32(&self, cws: &[[u64; 3]; 4]) -> [f32; 4] {
        self.calculate_batch_weighted_hamming_f32_neon_128bit_optimized(cws)
    }

    /// Calculates weighted Hamming distance using 128-bit NEON vectors.
    ///
    /// # Safety
    ///
    /// This function requires `neon` CPU feature to be available.
    /// Caller must ensure the target platform supports this feature.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn calculate_batch_weighted_hamming_f32_neon_128bit_optimized(
        &self,
        cws: &[[u64; 3]; 4],
    ) -> [f32; 4] {
        use core::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vst1q_f32};

        // Calculate diffs
        let diffs = [
            [
                cws[0][0] ^ self.c0_mrb_bitpacked[0],
                cws[0][1] ^ self.c0_mrb_bitpacked[1],
                cws[0][2] ^ self.c0_mrb_bitpacked[2],
            ],
            [
                cws[1][0] ^ self.c0_mrb_bitpacked[0],
                cws[1][1] ^ self.c0_mrb_bitpacked[1],
                cws[1][2] ^ self.c0_mrb_bitpacked[2],
            ],
            [
                cws[2][0] ^ self.c0_mrb_bitpacked[0],
                cws[2][1] ^ self.c0_mrb_bitpacked[1],
                cws[2][2] ^ self.c0_mrb_bitpacked[2],
            ],
            [
                cws[3][0] ^ self.c0_mrb_bitpacked[0],
                cws[3][1] ^ self.c0_mrb_bitpacked[1],
                cws[3][2] ^ self.c0_mrb_bitpacked[2],
            ],
        ];

        let mut totals = vdupq_n_f32(0.0);

        // Process word 0 (8 chunks)
        for chunk in 0..8usize {
            let chunk_base = chunk * 256;
            let patterns = [
                ((diffs[0][0] >> (chunk * 8)) & 0xFF) as usize,
                ((diffs[1][0] >> (chunk * 8)) & 0xFF) as usize,
                ((diffs[2][0] >> (chunk * 8)) & 0xFF) as usize,
                ((diffs[3][0] >> (chunk * 8)) & 0xFF) as usize,
            ];

            let values = [
                self.lookup_table_f32[chunk_base + patterns[0]],
                self.lookup_table_f32[chunk_base + patterns[1]],
                self.lookup_table_f32[chunk_base + patterns[2]],
                self.lookup_table_f32[chunk_base + patterns[3]],
            ];

            totals = vaddq_f32(totals, vld1q_f32(values.as_ptr()));
        }

        // Process word 1 (8 chunks)
        for chunk in 0..8usize {
            let chunk_base = (8 + chunk) * 256;
            let patterns = [
                ((diffs[0][1] >> (chunk * 8)) & 0xFF) as usize,
                ((diffs[1][1] >> (chunk * 8)) & 0xFF) as usize,
                ((diffs[2][1] >> (chunk * 8)) & 0xFF) as usize,
                ((diffs[3][1] >> (chunk * 8)) & 0xFF) as usize,
            ];

            let values = [
                self.lookup_table_f32[chunk_base + patterns[0]],
                self.lookup_table_f32[chunk_base + patterns[1]],
                self.lookup_table_f32[chunk_base + patterns[2]],
                self.lookup_table_f32[chunk_base + patterns[3]],
            ];

            totals = vaddq_f32(totals, vld1q_f32(values.as_ptr()));
        }

        // Process word 2 (6 chunks)
        for chunk in 0..6usize {
            let chunk_base = (16 + chunk) * 256;
            let patterns = [
                ((diffs[0][2] >> (chunk * 8)) & 0xFF) as usize,
                ((diffs[1][2] >> (chunk * 8)) & 0xFF) as usize,
                ((diffs[2][2] >> (chunk * 8)) & 0xFF) as usize,
                ((diffs[3][2] >> (chunk * 8)) & 0xFF) as usize,
            ];

            let values = [
                self.lookup_table_f32[chunk_base + patterns[0]],
                self.lookup_table_f32[chunk_base + patterns[1]],
                self.lookup_table_f32[chunk_base + patterns[2]],
                self.lookup_table_f32[chunk_base + patterns[3]],
            ];

            totals = vaddq_f32(totals, vld1q_f32(values.as_ptr()));
        }

        // Extract results
        let mut results = [0.0f32; 4];
        vst1q_f32(results.as_mut_ptr(), totals);
        results
    }

    /// Batch process 4 pre-diffed arrays using NEON SIMD.
    ///
    /// Like `calculate_batch_weighted_hamming_f32` but skips the XOR with
    /// `c0_mrb_bitpacked` — caller has already computed `diff = cw XOR c0`.
    /// Use this with arrays produced by `make_diff` or computed incrementally.
    ///
    /// Accumulates four independent distances into one `float32x4_t`, emitting
    /// one `fadd.4s` per 8-bit chunk (22 chunks total).  Explicit intrinsics are
    /// required: LLVM's SLP vectoriser does NOT auto-vectorise the equivalent
    /// safe Rust because the four lookup-table indices differ per lane (scatter
    /// reads), breaking the uniform-stride access pattern SLP requires.
    /// Verified by assembly inspection: safe version emits scalar `fadd s` × 4
    /// per chunk; this version emits a single `fadd.4s` — ~4× fewer instructions.
    ///
    /// # Safety
    /// Caller must be on an AArch64 target (guaranteed by the `#[cfg]` guard).
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn calculate_batch_weighted_hamming_f32_from_diffs(
        &self,
        diffs: &[[u64; 3]; 4],
    ) -> [f32; 4] {
        use core::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vst1q_f32};

        let mut totals = vdupq_n_f32(0.0);

        // Process word 0 (8 chunks)
        for chunk in 0..8usize {
            let chunk_base = chunk * 256;
            let values = [
                self.lookup_table_f32[chunk_base + ((diffs[0][0] >> (chunk * 8)) & 0xFF) as usize],
                self.lookup_table_f32[chunk_base + ((diffs[1][0] >> (chunk * 8)) & 0xFF) as usize],
                self.lookup_table_f32[chunk_base + ((diffs[2][0] >> (chunk * 8)) & 0xFF) as usize],
                self.lookup_table_f32[chunk_base + ((diffs[3][0] >> (chunk * 8)) & 0xFF) as usize],
            ];
            totals = vaddq_f32(totals, vld1q_f32(values.as_ptr()));
        }

        // Process word 1 (8 chunks)
        for chunk in 0..8usize {
            let chunk_base = (8 + chunk) * 256;
            let values = [
                self.lookup_table_f32[chunk_base + ((diffs[0][1] >> (chunk * 8)) & 0xFF) as usize],
                self.lookup_table_f32[chunk_base + ((diffs[1][1] >> (chunk * 8)) & 0xFF) as usize],
                self.lookup_table_f32[chunk_base + ((diffs[2][1] >> (chunk * 8)) & 0xFF) as usize],
                self.lookup_table_f32[chunk_base + ((diffs[3][1] >> (chunk * 8)) & 0xFF) as usize],
            ];
            totals = vaddq_f32(totals, vld1q_f32(values.as_ptr()));
        }

        // Process word 2 (6 chunks)
        for chunk in 0..6usize {
            let chunk_base = (16 + chunk) * 256;
            let values = [
                self.lookup_table_f32[chunk_base + ((diffs[0][2] >> (chunk * 8)) & 0xFF) as usize],
                self.lookup_table_f32[chunk_base + ((diffs[1][2] >> (chunk * 8)) & 0xFF) as usize],
                self.lookup_table_f32[chunk_base + ((diffs[2][2] >> (chunk * 8)) & 0xFF) as usize],
                self.lookup_table_f32[chunk_base + ((diffs[3][2] >> (chunk * 8)) & 0xFF) as usize],
            ];
            totals = vaddq_f32(totals, vld1q_f32(values.as_ptr()));
        }

        let mut results = [0.0f32; 4];
        vst1q_f32(results.as_mut_ptr(), totals);
        results
    }
}

/// Utility functions for distance calculations
pub mod utils {
    use super::*;

    /// Count the number of differing bits between two bit-packed codewords
    #[inline]
    pub fn hamming_distance_bitpacked(a: &[u64; 3], b: &[u64; 3]) -> u32 {
        let mut count = 0;
        for i in 0..3 {
            count += (a[i] ^ b[i]).count_ones();
        }
        count
    }

    /// Convert boolean array to bit-packed representation
    #[inline]
    pub fn pack_bool_array(bits: &[bool; N]) -> [u64; 3] {
        let mut packed = [0u64; 3];

        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                let chunk = i / 64;
                let bit_pos = i % 64;
                packed[chunk] |= 1u64 << bit_pos;
            }
        }

        packed
    }

    /// Convert bit-packed representation to boolean array
    #[inline]
    pub fn unpack_to_bool_array(packed: &[u64; 3]) -> [bool; N] {
        let mut bits = [false; N];

        for chunk in 0..3 {
            let mut word = packed[chunk];
            while word != 0 {
                let bit_pos = word.trailing_zeros() as usize;
                let global_pos = chunk * 64 + bit_pos;
                if global_pos < N {
                    bits[global_pos] = true;
                }
                word &= word - 1; // Clear the lowest set bit
            }
        }

        bits
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;
    use super::*;

    #[test]
    fn test_debw_calculation() {
        let alpha_mrb = [1.0; N];
        let c0_mrb_bitpacked = [0; 3];
        let calc = DistanceCalculator::new(alpha_mrb, c0_mrb_bitpacked);

        let tep = TestErrorPattern::with_positions(&[0, 5, 10]);
        let distance = calc.calculate_debw(&tep);

        assert_eq!(distance, 3.0); // Sum of alpha values at positions 0, 5, 10
    }

    #[test]
    fn test_weighted_hamming_consistency() {
        let alpha_mrb = [1.0; N]; // All weights = 1.0 for simplicity
        let c0_mrb_bitpacked = [0; 3];
        let calc = DistanceCalculator::new(alpha_mrb, c0_mrb_bitpacked);

        // Create test codewords
        let mut cw1 = [false; N];
        let mut cw2 = [false; N];
        cw1[0] = true;
        cw1[5] = true;
        cw2[0] = false;
        cw2[5] = true;

        let packed_cw1 = pack_bool_array(&cw1);
        let packed_cw2 = pack_bool_array(&cw2);

        let distance_bool = calc.calculate_weighted_hamming(&cw1, &cw2);
        let distance_packed = DistanceCalculator::calculate_weighted_hamming_bitpacked(
            &packed_cw1,
            &packed_cw2,
            &alpha_mrb,
        );

        assert_eq!(distance_bool, distance_packed);
    }

    #[test]
    fn test_pack_unpack_consistency() {
        let mut original = [false; N];
        original[0] = true;
        original[63] = true;
        original[64] = true;
        original[173] = true;

        let packed = pack_bool_array(&original);
        let unpacked = unpack_to_bool_array(&packed);

        assert_eq!(original, unpacked);
    }
}
