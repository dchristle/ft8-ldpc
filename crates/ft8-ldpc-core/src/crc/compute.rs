//! Runtime CRC-14 helpers for FT8 and FT4 payload bit arrays.
//!
//! Uses polynomial 0x2757 without its leading term, no reflection, an initial
//! remainder of zero and 96-bit augmented processing for 77-bit payloads.

#[cfg(test)]
use crate::crc::compute_crc14;
use crate::crc::constants::{CRC_LEN, MESSAGE_WITH_CRC_BITS, PAYLOAD_BITS};
use crate::crc::core::BIT_LEN;

const CRC_WIDTH: usize = 14;
const CRC_POLYNOMIAL: u16 = 0x2757;

/// Computes CRC-14 after padding the supplied bits with zeros to 96 bits.
///
/// # Panics
/// Panics if more than 96 bits are supplied.
#[inline(always)]
pub fn compute_crc14_from_bits(bits: &[bool]) -> u16 {
    let mut padded_bits = [false; BIT_LEN];
    padded_bits[..bits.len()].copy_from_slice(bits);

    let mut rem: u16 = 0;
    const MASK: u16 = (1 << CRC_WIDTH) - 1;

    for &bit in &padded_bits {
        rem = (rem << 1) | (bit as u16);

        let needs_xor = rem >> CRC_WIDTH;
        rem ^= CRC_POLYNOMIAL & needs_xor.wrapping_neg();

        rem &= MASK;
    }

    rem
}

#[inline(always)]
fn compute_crc14_from_bitpacked(cw: [u64; 3]) -> u16 {
    let mut rem: u16 = 0;
    let mask: u16 = (1 << CRC_WIDTH) - 1;

    for i in 0..BIT_LEN {
        let bit = if i < PAYLOAD_BITS {
            let word_index = i / 64;
            let bit_index = i % 64;
            ((cw[word_index] >> bit_index) & 1) != 0
        } else {
            false
        };

        rem = (rem << 1) | (bit as u16);

        let needs_xor = rem >> CRC_WIDTH;
        rem ^= CRC_POLYNOMIAL & needs_xor.wrapping_neg();
        rem &= mask;
    }

    rem
}

#[cfg(test)]
use crate::crc::add_crc14;

/// Verifies CRC on a 91-bit message-with-CRC.
#[inline(always)]
pub fn verify_crc14(message: &[bool; MESSAGE_WITH_CRC_BITS]) -> bool {
    let computed_crc = compute_crc14_from_bits(&message[..PAYLOAD_BITS]);

    let actual_crc = message[PAYLOAD_BITS..]
        .iter()
        .fold(0u16, |acc, &bit| (acc << 1) | (bit as u16));

    computed_crc == actual_crc
}

#[cfg(test)]
fn verify_crc14_bytes(message: &[bool; MESSAGE_WITH_CRC_BITS]) -> bool {
    let mut bytes = [0u8; 12];
    for (i, &bit) in message[..PAYLOAD_BITS].iter().enumerate() {
        if bit {
            bytes[i / 8] |= 1 << (7 - (i % 8));
        }
    }

    let computed_crc = compute_crc14(&bytes, BIT_LEN);

    let mut actual_crc = 0u16;
    for i in 0..CRC_LEN {
        if message[PAYLOAD_BITS + i] {
            actual_crc |= 1 << (CRC_LEN - 1 - i);
        }
    }

    computed_crc == actual_crc
}

/// Verifies the CRC in the first 91 bits of an LSB-first packed codeword.
#[inline(always)]
pub fn verify_crc14_bitpacked(cw: [u64; 3]) -> bool {
    verify_crc14_packed_message([cw[0], cw[1]])
}

/// Verifies a 91-bit LSB-first packed message-with-CRC, ignoring unused high bits.
#[inline(always)]
pub fn verify_crc14_packed_message(message: [u64; 2]) -> bool {
    let computed_crc = compute_crc14_from_bitpacked([message[0], message[1], 0]);
    let mut actual_crc = 0u16;
    for i in PAYLOAD_BITS..(PAYLOAD_BITS + CRC_LEN) {
        let word_index = i / 64;
        let bit_index = i % 64;
        let bit = ((message[word_index] >> bit_index) & 1) != 0;
        actual_crc = (actual_crc << 1) | (bit as u16);
    }

    computed_crc == actual_crc
}

/// Verifies the CRC in the first 91 bits without checking LDPC parity.
pub fn verify_codeword_crc14(cw: &[bool; 174]) -> bool {
    let mut message_with_crc: [bool; 91] = [false; 91];
    message_with_crc.copy_from_slice(&cw[..91]);
    verify_crc14(&message_with_crc)
}

#[cfg(test)]
fn extract_crc14(message: &[bool; MESSAGE_WITH_CRC_BITS]) -> [bool; CRC_LEN] {
    let mut result = [false; CRC_LEN];
    result.copy_from_slice(&message[PAYLOAD_BITS..MESSAGE_WITH_CRC_BITS]);
    result
}

#[cfg(test)]
#[inline(always)]
fn extract_crc(crc_bits: &[bool]) -> u16 {
    crc_bits
        .iter()
        .fold(0u16, |acc, &bit| (acc << 1) | (bit as u16))
}

#[cfg(test)]
fn binary_str_to_bools<const N: usize>(s: &str) -> [bool; N] {
    let mut result = [false; N];
    for (i, c) in s.chars().enumerate() {
        result[i] = c == '1';
    }
    result
}

#[cfg(test)]
fn binary_str_to_u16(s: &str) -> u16 {
    s.chars()
        .fold(0u16, |acc, c| (acc << 1) | (c == '1') as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAYLOAD: &str =
        "00000000000000000000000000100000010011011111110011011100100010100001010000001";
    const CRC: &str = "01010101111001";

    #[test]
    fn test_known_good_message_1() {
        let msg = PAYLOAD;
        let expected_crc = CRC;

        let mut bytes = [0u8; 12];
        let msg_bits = binary_str_to_bools::<PAYLOAD_BITS>(msg);
        for (i, &bit) in msg_bits.iter().enumerate() {
            if bit {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }

        let crc = compute_crc14(&bytes, BIT_LEN);
        let expected = binary_str_to_u16(expected_crc);

        assert_eq!(
            crc, expected,
            "CRC computation mismatch: got {:014b}, expected {:014b}",
            crc, expected
        );
    }

    #[test]
    fn test_known_good_message_2() {
        let msg = "11010100010001000000001111010100001011010001000100110011001111111000000010000";
        let crc = "11001100101011";

        let mut full_msg = [false; MESSAGE_WITH_CRC_BITS];
        full_msg[..PAYLOAD_BITS].copy_from_slice(&binary_str_to_bools::<PAYLOAD_BITS>(msg));
        full_msg[PAYLOAD_BITS..].copy_from_slice(&binary_str_to_bools::<CRC_LEN>(crc));

        assert!(verify_crc14(&full_msg));
    }

    #[test]
    fn test_known_good_message_3() {
        let msg = "11010111110111010111001110010011111011011100000011001100100110111001100111110";
        let crc = "10000010000110";

        let mut full_msg = [false; MESSAGE_WITH_CRC_BITS];
        full_msg[..PAYLOAD_BITS].copy_from_slice(&binary_str_to_bools::<PAYLOAD_BITS>(msg));
        full_msg[PAYLOAD_BITS..].copy_from_slice(&binary_str_to_bools::<CRC_LEN>(crc));

        assert!(verify_crc14(&full_msg));
    }

    #[test]
    fn test_verify_valid_crc() {
        let msg = PAYLOAD;
        let crc = CRC;

        let mut full_msg = [false; MESSAGE_WITH_CRC_BITS];
        full_msg[..PAYLOAD_BITS].copy_from_slice(&binary_str_to_bools::<PAYLOAD_BITS>(msg));
        full_msg[PAYLOAD_BITS..].copy_from_slice(&binary_str_to_bools::<CRC_LEN>(crc));

        assert!(verify_crc14(&full_msg));
        assert!(verify_crc14_bytes(&full_msg));
    }

    #[test]
    fn test_verify_invalid_crc() {
        let msg = PAYLOAD;

        let mut full_msg = [false; MESSAGE_WITH_CRC_BITS];
        full_msg[..PAYLOAD_BITS].copy_from_slice(&binary_str_to_bools::<PAYLOAD_BITS>(msg));
        full_msg[PAYLOAD_BITS..].copy_from_slice(&binary_str_to_bools::<CRC_LEN>(CRC));
        full_msg[MESSAGE_WITH_CRC_BITS - 1] ^= true;

        assert!(!verify_crc14(&full_msg));
    }

    #[test]
    fn test_extract_matches_add() {
        let msg = PAYLOAD;
        let msg_bits = binary_str_to_bools::<PAYLOAD_BITS>(msg);

        let with_crc = add_crc14(&msg_bits);
        let extracted = extract_crc14(&with_crc);

        assert_eq!(&with_crc[PAYLOAD_BITS..], &extracted);
    }

    #[test]
    fn test_extract_crc_packs_bits() {
        let crc_bits = binary_str_to_bools::<CRC_LEN>(CRC);
        assert_eq!(extract_crc(&crc_bits), binary_str_to_u16(CRC));
    }

    #[test]
    fn test_verify_packed_message_matches_bool_message() {
        let msg = PAYLOAD;
        let msg_bits = binary_str_to_bools::<PAYLOAD_BITS>(msg);
        let with_crc = add_crc14(&msg_bits);

        let mut packed = [0u64; 2];
        for (i, bit) in with_crc.iter().copied().enumerate() {
            if bit {
                packed[i / 64] |= 1u64 << (i % 64);
            }
        }

        assert_eq!(verify_crc14_packed_message(packed), verify_crc14(&with_crc));
    }
}
