//! Compile-time CRC-14 computation for FT8 and FT4 payloads.
//!
//! Uses polynomial 0x2757 without its leading term, no reflection, an initial
//! remainder of zero and 96-bit augmented processing.

use crate::crc::constants::{CRC_LEN, MESSAGE_WITH_CRC_BITS, PAYLOAD_BITS};

const CRC_WIDTH: usize = 14;
const CRC_POLYNOMIAL: u16 = 0x2757;
pub(crate) const BYTE_LEN: usize = 12;
pub(crate) const BIT_LEN: usize = BYTE_LEN * 8;

/// Computes CRC-14 over exactly `bit_length` MSB-first bits without adding padding.
///
/// # Panics
/// Panics if `bit_length` exceeds the number of bits in `data`.
#[inline]
pub const fn compute_crc14(data: &[u8], bit_length: usize) -> u16 {
    let mut rem: u16 = 0;

    let mut bit_idx = 0;
    while bit_idx < bit_length {
        let byte_idx = bit_idx / 8;
        let bit_in_byte = 7 - (bit_idx % 8);

        let bit = (data[byte_idx] >> bit_in_byte) & 1;

        rem = (rem << 1) | (bit as u16);

        if (rem >> CRC_WIDTH) != 0 {
            rem ^= CRC_POLYNOMIAL;
        }

        rem &= (1 << CRC_WIDTH) - 1;
        bit_idx += 1;
    }

    rem
}

/// Appends the 14 CRC bits, most significant first, to a 77-bit payload.
pub const fn add_crc14(message: &[bool; PAYLOAD_BITS]) -> [bool; MESSAGE_WITH_CRC_BITS] {
    let mut bytes = [0u8; BYTE_LEN];
    let mut i = 0;
    while i < PAYLOAD_BITS {
        if message[i] {
            bytes[i / 8] |= 1 << (7 - (i % 8));
        }
        i += 1;
    }

    let crc = compute_crc14(&bytes, BIT_LEN);

    let mut result = [false; MESSAGE_WITH_CRC_BITS];
    let mut i = 0;
    while i < PAYLOAD_BITS {
        result[i] = message[i];
        i += 1;
    }

    let mut i = 0;
    while i < CRC_LEN {
        result[PAYLOAD_BITS + i] = (crc & (1 << (CRC_LEN - 1 - i))) != 0;
        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAYLOAD: &str =
        "00000000000000000000000000100000010011011111110011011100100010100001010000001";
    const CRC: &str = "01010101111001";

    fn binary_str_to_bools<const N: usize>(s: &str) -> [bool; N] {
        let mut result = [false; N];
        for (i, c) in s.chars().enumerate() {
            result[i] = c == '1';
        }
        result
    }

    fn binary_str_to_u16(s: &str) -> u16 {
        s.chars()
            .fold(0u16, |acc, c| (acc << 1) | (c == '1') as u16)
    }

    #[test]
    fn test_compute_crc14_known_values() {
        let msg = PAYLOAD;
        let expected_crc = CRC;

        let mut bytes = [0u8; 12];
        let msg_bits = binary_str_to_bools::<PAYLOAD_BITS>(msg);
        for (i, &bit) in msg_bits.iter().enumerate() {
            if bit {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }

        let crc = compute_crc14(&bytes, 96);
        let expected = binary_str_to_u16(expected_crc);

        assert_eq!(
            crc, expected,
            "CRC computation mismatch: got {:014b}, expected {:014b}",
            crc, expected
        );
    }

    #[test]
    fn test_add_crc14_consistency() {
        let msg = PAYLOAD;
        let msg_bits = binary_str_to_bools::<PAYLOAD_BITS>(msg);

        let with_crc = add_crc14(&msg_bits);

        assert_eq!(&with_crc[..PAYLOAD_BITS], &msg_bits[..]);

        let mut bytes = [0u8; 12];
        for (i, &bit) in msg_bits.iter().enumerate() {
            if bit {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let expected_crc = compute_crc14(&bytes, 96);

        let actual_crc = with_crc[PAYLOAD_BITS..]
            .iter()
            .fold(0u16, |acc, &bit| (acc << 1) | (bit as u16));

        assert_eq!(actual_crc, expected_crc);
    }

    #[test]
    fn test_const_fn_compatibility() {
        const TEST_MSG: [bool; PAYLOAD_BITS] = [false; PAYLOAD_BITS];
        const _: [bool; MESSAGE_WITH_CRC_BITS] = add_crc14(&TEST_MSG);

        const TEST_BYTES: [u8; 12] = [0; 12];
        const _: u16 = compute_crc14(&TEST_BYTES, 96);
    }
}
