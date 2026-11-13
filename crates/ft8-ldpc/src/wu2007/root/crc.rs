//! CRC syndromes for candidates formed by XORing systematic basis rows.

use super::*;
use ft8_ldpc_core::crc::crc14_syndrome_packed_message;

#[derive(Default)]
pub(super) struct CandidateCrc {
    cached: Option<BasisSyndromes>,
}

struct BasisSyndromes {
    baseline: u16,
    rows: [u16; 91],
}

impl CandidateCrc {
    pub(super) fn check(
        &mut self,
        basis: &SystematicBasis,
        candidate: &PackedCodeword,
        positions: &[u8],
    ) -> bool {
        let direct = |word: &PackedCodeword| {
            crc14_syndrome_packed_message(extract_natural_message_with_crc(
                word,
                &basis.natural_to_sys,
            ))
        };
        // Leave the cache empty for baseline early stops and searches that
        // never need another CRC check after metric pruning.
        if positions.is_empty() {
            return direct(candidate) == 0;
        }
        let cached = self.cached.get_or_insert_with(|| {
            let rows = std::array::from_fn(|q| direct(&basis.encoded_basis_deltas[q]));
            // The first checked candidate need not be the baseline. Undo its
            // flipped rows in syndrome space to recover the baseline syndrome.
            let baseline = positions
                .iter()
                .fold(direct(candidate), |s, &p| s ^ rows[p as usize]);
            BasisSyndromes { baseline, rows }
        });
        positions
            .iter()
            .fold(cached.baseline, |s, &p| s ^ cached.rows[p as usize])
            == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wu2007::root::test_support::random_codeword;
    use crate::wu2007::scoring::test_inputs::SplitMix64;

    #[test]
    fn cached_syndromes_match_direct_checks_for_every_cascade_and_order() {
        let mut rng = SplitMix64(0xc7c1_4ba5);
        for cascade_bits in 0..=14 {
            let mut decoder = Decoder::new(Config {
                cascade_bits,
                ..Config::default()
            })
            .unwrap();
            let k = 91 - cascade_bits;
            let llr = std::array::from_fn(|_| rng.unit() * 8. - 4.);
            decoder.rebuild_basis(&llr, k).unwrap();
            let basis = &decoder.basis;
            let natural = random_codeword(&mut rng, cascade_bits);
            let mut baseline = [0; 3];
            for (s, &n) in basis.sys_to_natural.iter().enumerate() {
                if get_codeword_bit(&natural, n as usize) {
                    set_codeword_bit(&mut baseline, s);
                }
            }
            let mut crc = CandidateCrc::default();
            assert!(crc.check(basis, &baseline, &[]));
            assert!(
                crc.cached.is_none(),
                "baseline must not prepare row syndromes"
            );

            // Start a second cache at a non-baseline candidate, as a stage
            // runner can do. Both caches must recover the same baseline.
            let mut without_baseline = CandidateCrc::default();
            for order in 1..=6 {
                for _ in 0..32 {
                    let mut positions = [0; 6];
                    let mut candidate = baseline;
                    for p in &mut positions[..order] {
                        *p = rng.below(k) as u8;
                        xor_in_place(&mut candidate, &basis.encoded_basis_deltas[*p as usize]);
                    }
                    let expected = verify_crc14_packed_message(extract_natural_message_with_crc(
                        &candidate,
                        &basis.natural_to_sys,
                    ));
                    assert_eq!(crc.check(basis, &candidate, &positions[..order]), expected);
                    assert_eq!(
                        without_baseline.check(basis, &candidate, &positions[..order]),
                        expected
                    );
                }
            }
            assert_eq!(crc.cached.as_ref().unwrap().baseline, 0);
            assert_eq!(without_baseline.cached.as_ref().unwrap().baseline, 0);
        }
    }
}
