//! Four-lane first-preprocessing kernels.
use super::*;

#[inline(always)]
fn survivors<const M: usize, const LANES: usize>(p: u64, x: &[u64; LANES], theta: usize) -> u8 {
    if theta < M {
        return 0;
    }
    if LANES == 1 {
        return u8::from((p ^ x[0]).count_ones() as usize <= theta - M);
    }
    let counts = x.map(|value| (p ^ value).count_ones());
    let limit = theta - M;
    if counts.iter().copied().min().unwrap() as usize > limit {
        return 0;
    }
    let mut mask = 0;
    for (lane, count) in counts.into_iter().enumerate() {
        mask |= u8::from(count as usize <= limit) << lane;
    }
    mask
}

impl Decoder {
    // Orders 2 and 3 check the flag before a block and every nonempty-block
    // lane. Order 2 carries its leading diff; order 3 rebuilds per survivor.
    #[inline(always)]
    fn pre1_vector_block<const M: usize, const LANES: usize>(
        &mut self,
        state: &mut SearchState,
        baseline: PackedCodeword,
        leading: [u8; 6],
        p: u64,
        j: usize,
        theta: usize,
        examined: &mut usize,
        rejected: &mut usize,
    ) -> bool {
        if state.diagnostics.candidate_cap_hit {
            return false;
        }
        let block = self.prefix_words[j..].first_chunk::<LANES>().unwrap();
        let mask = survivors::<M, LANES>(p, block, theta);
        if mask == 0 {
            *examined += LANES;
            *rejected += LANES;
            return true;
        }
        core::hint::cold_path();
        for lane in 0..LANES {
            if state.diagnostics.candidate_cap_hit {
                return false;
            }
            *examined += 1;
            if mask & (1 << lane) == 0 {
                *rejected += 1;
                continue;
            }
            let mut positions = leading;
            positions[M - 1] = (j + lane) as u8;
            let diff = if M == 2 {
                // baseline is d0 for order 2, to keep the carried XOR.
                xor_delta(baseline, self.basis.encoded_basis_deltas[j + lane])
            } else {
                self.diff_from_positions(baseline, &positions[..M])
            };
            self.reserve_and_score(state, diff, M as u8, positions);
        }
        true
    }

    // Keep each traversal separate from decode: this limits register pressure
    // and leaves baseline/position construction behind the cold survivor branch.
    #[inline(never)]
    pub(in crate::wu2007::root) fn pre1_order2_vector(
        &mut self,
        state: &mut SearchState,
        baseline: PackedCodeword,
        k: usize,
        b: u64,
        theta: usize,
    ) {
        assert!(k <= self.prefix_words.len());
        let (mut ex, mut rej) = (0, 0);
        'stage: for i0 in 0..k - 1 {
            let p0 = b ^ self.prefix_words[i0];
            let d0 = xor_delta(baseline, self.basis.encoded_basis_deltas[i0]);
            let leading = [i0 as u8, 0, 0, 0, 0, 0];
            let mut j = i0 + 1;
            while j + 4 <= k {
                if !self
                    .pre1_vector_block::<2, 4>(state, d0, leading, p0, j, theta, &mut ex, &mut rej)
                {
                    break 'stage;
                }
                j += 4;
            }
            if j + 2 <= k {
                if !self
                    .pre1_vector_block::<2, 2>(state, d0, leading, p0, j, theta, &mut ex, &mut rej)
                {
                    break 'stage;
                }
                j += 2;
            }
            if j < k {
                if !self
                    .pre1_vector_block::<2, 1>(state, d0, leading, p0, j, theta, &mut ex, &mut rej)
                {
                    break 'stage;
                }
            }
        }
        Self::pre1_counts(state, ex, rej);
    }

    #[inline(never)]
    pub(in crate::wu2007::root) fn pre1_order3_vector(
        &mut self,
        state: &mut SearchState,
        baseline: PackedCodeword,
        k: usize,
        b: u64,
        theta: usize,
    ) {
        assert!(k <= self.prefix_words.len());
        let (mut ex, mut rej) = (0, 0);
        'stage: for i0 in 0..k - 2 {
            let p0 = b ^ self.prefix_words[i0];
            for i1 in i0 + 1..k - 1 {
                let p01 = p0 ^ self.prefix_words[i1];
                let leading = [i0 as u8, i1 as u8, 0, 0, 0, 0];
                let mut j = i1 + 1;
                while j + 4 <= k {
                    if !self.pre1_vector_block::<3, 4>(
                        state, baseline, leading, p01, j, theta, &mut ex, &mut rej,
                    ) {
                        break 'stage;
                    }
                    j += 4;
                }
                while j < k {
                    if !self.pre1_vector_block::<3, 1>(
                        state, baseline, leading, p01, j, theta, &mut ex, &mut rej,
                    ) {
                        break 'stage;
                    }
                    j += 1;
                }
            }
        }
        Self::pre1_counts(state, ex, rej);
    }

    #[inline(always)]
    fn pre1_order4_block<const LANES: usize>(
        &mut self,
        state: &mut SearchState,
        baseline: PackedCodeword,
        leading: [u8; 6],
        p: u64,
        j: usize,
        theta: usize,
        examined: &mut usize,
        rejected: &mut usize,
    ) -> bool {
        let block = self.prefix_words[j..].first_chunk::<LANES>().unwrap();
        let mask = survivors::<4, LANES>(p, block, theta);
        if mask == 0 {
            *examined += LANES;
            *rejected += LANES;
            return true;
        }
        core::hint::cold_path();
        // Exactly one rebuild per surviving block, none in the outer loops.
        let d = self.diff_from_positions(baseline, &leading[..3]);
        for lane in 0..LANES {
            *examined += 1;
            if mask & (1 << lane) == 0 {
                *rejected += 1;
                continue;
            }
            let mut positions = leading;
            positions[3] = (j + lane) as u8;
            self.reserve_and_score(
                state,
                xor_delta(d, self.basis.encoded_basis_deltas[j + lane]),
                4,
                positions,
            );
            // Rejected patterns cannot exhaust the candidate cap.
            if state.diagnostics.candidate_cap_hit {
                return false;
            }
        }
        true
    }

    #[inline(never)]
    pub(in crate::wu2007::root) fn pre1_order4_vector(
        &mut self,
        state: &mut SearchState,
        baseline: PackedCodeword,
        k: usize,
        b: u64,
        theta: usize,
    ) {
        assert!(k <= self.prefix_words.len());
        let (mut ex, mut rej) = (0, 0);
        'stage: for i0 in 0..k - 3 {
            let p0 = b ^ self.prefix_words[i0];
            for i1 in i0 + 1..k - 2 {
                let p01 = p0 ^ self.prefix_words[i1];
                for i2 in i1 + 1..k - 1 {
                    let p012 = p01 ^ self.prefix_words[i2];
                    let leading = [i0 as u8, i1 as u8, i2 as u8, 0, 0, 0];
                    let mut j = i2 + 1;
                    while j + 4 <= k {
                        if !self.pre1_order4_block::<4>(
                            state, baseline, leading, p012, j, theta, &mut ex, &mut rej,
                        ) {
                            break 'stage;
                        }
                        j += 4;
                    }
                    if j + 2 <= k {
                        if !self.pre1_order4_block::<2>(
                            state, baseline, leading, p012, j, theta, &mut ex, &mut rej,
                        ) {
                            break 'stage;
                        }
                        j += 2;
                    }
                    if j < k {
                        if !self.pre1_order4_block::<1>(
                            state, baseline, leading, p012, j, theta, &mut ex, &mut rej,
                        ) {
                            break 'stage;
                        }
                    }
                }
            }
        }
        Self::pre1_counts(state, ex, rej);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn masks_match_scalar_at_every_population_count_and_threshold() {
        for m in [2, 3, 4] {
            for count in 0..=64 {
                let p = low_mask(count);
                let x = [0, !0, 0x5555_5555_5555_5555, 0xaaaa_aaaa_aaaa_aaaa];
                for theta in (0..=70).chain([usize::MAX]) {
                    let expected = x.iter().enumerate().fold(0, |mask, (lane, x)| {
                        mask | (u8::from(m + (p ^ x).count_ones() as usize <= theta) << lane)
                    });
                    macro_rules! check {
                        ($m:literal) => {{
                            assert_eq!(survivors::<$m, 4>(p, &x, theta), expected);
                            assert_eq!(survivors::<$m, 2>(p, &[x[0], x[1]], theta), expected & 3);
                            assert_eq!(survivors::<$m, 1>(p, &[x[0]], theta), expected & 1);
                        }};
                    }
                    match m {
                        2 => check!(2),
                        3 => check!(3),
                        _ => check!(4),
                    }
                }
            }
        }
    }

    /// Compares the 4-lane and 2-lane helpers with the scalar reference for every
    /// θ from 0 to 70 and `usize::MAX`, each m, and each lane count.
    #[test]
    fn survivor_masks_match_the_scalar_reference() {
        use crate::wu2007::root::search::reference::survivor_mask_reference;
        let mut rng = crate::wu2007::scoring::test_inputs::SplitMix64(0x7700);
        let mut inputs: Vec<(u64, [u64; 4])> = vec![
            (0, [0; 4]),
            (!0, [!0; 4]),
            (0x0123_4567_89ab_cdef, [0x0123_4567_89ab_cdef; 4]),
            (0, [0, 1, 3, 7]),
            (0, [0, !0, 0x5555_5555_5555_5555, 0xff]),
        ];
        for flips in 0..=8 {
            let p = rng.next();
            let lane = |rng: &mut crate::wu2007::scoring::test_inputs::SplitMix64| {
                (0..flips).fold(p, |x, _| x ^ (1 << rng.below(64)))
            };
            inputs.push((
                p,
                [
                    lane(&mut rng),
                    lane(&mut rng),
                    lane(&mut rng),
                    lane(&mut rng),
                ],
            ));
        }
        for _ in 0..6 {
            inputs.push((rng.next(), [rng.next(), rng.next(), rng.next(), rng.next()]));
        }
        let (mut empty, mut full, mut mixed) = (false, false, false);
        for (p, x) in inputs {
            for theta in (0..=70).chain([usize::MAX]) {
                macro_rules! check {
                    ($m:literal) => {{
                        let four = survivors::<$m, 4>(p, &x, theta);
                        assert_eq!(four, survivor_mask_reference(p, &x, $m, theta));
                        let two = survivors::<$m, 2>(p, &[x[0], x[1]], theta);
                        assert_eq!(two, survivor_mask_reference(p, &x[..2], $m, theta));
                        empty |= four == 0;
                        full |= four == 0b1111;
                        mixed |= four != 0 && four != 0b1111;
                    }};
                }
                check!(2);
                check!(3);
                check!(4);
            }
        }
        assert!(empty && full && mixed);
    }
}
