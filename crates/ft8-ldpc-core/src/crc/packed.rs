//! Fixed-length CRC-14 syndromes for LSB-first packed messages.

use super::constants::{CRC_LEN, MESSAGE_WITH_CRC_BITS, PAYLOAD_BITS};
use super::core::add_crc14;

const MESSAGE_BYTES: usize = MESSAGE_WITH_CRC_BITS.div_ceil(8);

// Each table maps one byte directly to its contribution to computed CRC XOR
// stored CRC. Derive the payload contributions from the const encoder so the
// 96-bit zero padding and MSB-first CRC convention have a single definition.
static BYTE_SYNDROMES: [[u16; 256]; MESSAGE_BYTES] = byte_syndromes();

const fn byte_syndromes() -> [[u16; 256]; MESSAGE_BYTES] {
    let mut columns = [0u16; MESSAGE_WITH_CRC_BITS];
    let mut bit = 0;
    while bit < PAYLOAD_BITS {
        let mut payload = [false; PAYLOAD_BITS];
        payload[bit] = true;
        let encoded = add_crc14(&payload);
        let mut crc_bit = 0;
        while crc_bit < CRC_LEN {
            columns[bit] |= (encoded[PAYLOAD_BITS + crc_bit] as u16) << (CRC_LEN - 1 - crc_bit);
            crc_bit += 1;
        }
        bit += 1;
    }
    while bit < MESSAGE_WITH_CRC_BITS {
        columns[bit] = 1 << (MESSAGE_WITH_CRC_BITS - 1 - bit);
        bit += 1;
    }

    let mut tables = [[0; 256]; MESSAGE_BYTES];
    let mut byte = 0;
    while byte < MESSAGE_BYTES {
        let mut value = 0;
        while value < 256 {
            let mut offset = 0;
            while offset < 8 {
                let bit = byte * 8 + offset;
                if bit < MESSAGE_WITH_CRC_BITS && value & (1 << offset) != 0 {
                    tables[byte][value] ^= columns[bit];
                }
                offset += 1;
            }
            value += 1;
        }
        byte += 1;
    }
    tables
}

/// Returns computed CRC XOR stored CRC for a 91-bit LSB-first packed message.
///
/// Zero means the CRC is valid. The syndrome is linear under XOR, uses the
/// stored CRC's MSB-first numeric order, and ignores unused high bits.
#[inline(always)]
pub fn crc14_syndrome_packed_message(message: [u64; 2]) -> u16 {
    let mut syndrome = 0;
    // Constant word and byte offsets allow all twelve lookups to execute
    // independently, without a serial polynomial-reduction recurrence.
    macro_rules! bytes {
        ($($byte:literal),* $(,)?) => {$(
            syndrome ^= BYTE_SYNDROMES[$byte]
                [((message[$byte / 8] >> (8 * ($byte % 8))) & 255) as usize];
        )*};
    }
    bytes!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    syndrome
}

/// Verifies a 91-bit LSB-first packed message-with-CRC, ignoring unused high bits.
#[inline(always)]
pub fn verify_crc14_packed_message(message: [u64; 2]) -> bool {
    crc14_syndrome_packed_message(message) == 0
}

/// Verifies the CRC in the first 91 bits of an LSB-first packed codeword.
#[inline(always)]
pub fn verify_crc14_bitpacked(cw: [u64; 3]) -> bool {
    verify_crc14_packed_message([cw[0], cw[1]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crc::{compute_crc14_from_bits, verify_crc14};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn bitwise_syndrome(packed: [u64; 2]) -> u16 {
        let message: [bool; MESSAGE_WITH_CRC_BITS] =
            std::array::from_fn(|i| (packed[i / 64] >> (i % 64)) & 1 != 0);
        let stored = message[PAYLOAD_BITS..]
            .iter()
            .fold(0, |crc, &bit| (crc << 1) | bit as u16);
        let expected = compute_crc14_from_bits(&message[..PAYLOAD_BITS]) ^ stored;
        assert_eq!(verify_crc14_packed_message(packed), verify_crc14(&message));
        expected
    }

    #[test]
    fn packed_syndromes_match_bitwise_crc_and_are_linear() {
        for bit in 0..128 {
            let mut message = [0; 2];
            message[bit / 64] = 1 << (bit % 64);
            assert_eq!(
                crc14_syndrome_packed_message(message),
                bitwise_syndrome(message)
            );
        }
        let mut rng = StdRng::seed_from_u64(0xc7c1_4007);
        for _ in 0..4096 {
            let a: [u64; 2] = rng.random();
            let b: [u64; 2] = rng.random();
            let combined = [a[0] ^ b[0], a[1] ^ b[1]];
            let syndrome = crc14_syndrome_packed_message(a);
            assert_eq!(syndrome, bitwise_syndrome(a));
            assert_eq!(
                crc14_syndrome_packed_message(combined),
                syndrome ^ crc14_syndrome_packed_message(b)
            );
        }
    }

    #[test]
    fn packed_verifiers_ignore_unused_bits() {
        let payload = std::array::from_fn(|i| i % 3 == 0);
        let message = add_crc14(&payload);
        let mut packed = [0; 3];
        for (i, bit) in message.into_iter().enumerate() {
            packed[i / 64] |= (bit as u64) << (i % 64);
        }
        packed[1] |= u64::MAX << (MESSAGE_WITH_CRC_BITS - 64);
        packed[2] = u64::MAX;
        assert_eq!(crc14_syndrome_packed_message([packed[0], packed[1]]), 0);
        assert!(verify_crc14_bitpacked(packed));
        for bit in 0..MESSAGE_WITH_CRC_BITS {
            let mut broken = packed;
            broken[bit / 64] ^= 1 << (bit % 64);
            assert!(!verify_crc14_bitpacked(broken), "bit {bit}");
        }
    }
}
