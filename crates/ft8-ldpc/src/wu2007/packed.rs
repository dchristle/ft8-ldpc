//! Packed operations with bit zero in the least significant position.
use ft8_ldpc_core::protocol::N;
pub const CODEWORD_WORDS: usize = 3;
pub const MESSAGE_WORDS: usize = 2;
pub const REDUNDANCY_WORDS: usize = 2;
pub type PackedCodeword = [u64; CODEWORD_WORDS];
pub type PackedMessage = [u64; MESSAGE_WORDS];
pub type PackedRedundancy = [u64; REDUNDANCY_WORDS];

#[doc(hidden)]
pub fn low_mask(bits: usize) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}
#[doc(hidden)]
pub fn active_message_mask(len: usize) -> PackedMessage {
    [low_mask(len), low_mask(len.saturating_sub(64))]
}
#[doc(hidden)]
pub fn prefix_mask_red(len: usize) -> PackedRedundancy {
    active_message_mask(len)
}
#[doc(hidden)]
pub fn mask_message_words(words: PackedMessage, mask: PackedMessage) -> PackedMessage {
    [words[0] & mask[0], words[1] & mask[1]]
}
#[doc(hidden)]
pub fn mask_red(diff: PackedRedundancy, mask: PackedRedundancy) -> PackedRedundancy {
    mask_message_words(diff, mask)
}
#[doc(hidden)]
pub fn extract_prefix(words: &PackedCodeword, len: usize) -> PackedMessage {
    mask_message_words([words[0], words[1]], active_message_mask(len))
}
#[doc(hidden)]
pub fn message_with_crc_from_codeword(codeword: &PackedCodeword) -> PackedMessage {
    extract_prefix(codeword, 91)
}
#[doc(hidden)]
pub fn extract_slice(words: &PackedCodeword, start: usize, len: usize) -> PackedRedundancy {
    let word = start / 64;
    let shift = start % 64;
    let at = |i: usize| words.get(i).copied().unwrap_or(0);
    let a = if shift == 0 {
        at(word)
    } else {
        (at(word) >> shift) | (at(word + 1) << (64 - shift))
    };
    let b = if shift == 0 {
        at(word + 1)
    } else {
        (at(word + 1) >> shift) | (at(word + 2) << (64 - shift))
    };
    mask_red([a, b], prefix_mask_red(len))
}
#[doc(hidden)]
pub fn get_message_bit(words: &PackedMessage, pos: usize) -> bool {
    words[pos / 64] & (1u64 << (pos % 64)) != 0
}
#[doc(hidden)]
pub fn get_codeword_bit(words: &PackedCodeword, pos: usize) -> bool {
    words[pos / 64] & (1u64 << (pos % 64)) != 0
}
#[doc(hidden)]
pub fn set_codeword_bit(words: &mut PackedCodeword, pos: usize) {
    words[pos / 64] |= 1u64 << (pos % 64);
}
#[doc(hidden)]
#[inline(always)]
pub fn xor_delta(base: PackedCodeword, delta: PackedCodeword) -> PackedCodeword {
    [base[0] ^ delta[0], base[1] ^ delta[1], base[2] ^ delta[2]]
}
#[doc(hidden)]
#[cfg(test)]
pub fn xor_codewords(a: &PackedCodeword, b: &PackedCodeword) -> PackedCodeword {
    xor_delta(*a, *b)
}
#[doc(hidden)]
pub fn xor_in_place(target: &mut PackedCodeword, delta: &PackedCodeword) {
    *target = xor_delta(*target, *delta);
}
#[doc(hidden)]
pub fn xor_red(base: PackedRedundancy, delta: PackedRedundancy) -> PackedRedundancy {
    [base[0] ^ delta[0], base[1] ^ delta[1]]
}
#[doc(hidden)]
pub fn popcount_red(diff: PackedRedundancy) -> usize {
    (diff[0].count_ones() + diff[1].count_ones()) as usize
}
#[doc(hidden)]
pub fn popcount_codeword(diff: PackedCodeword) -> u16 {
    (diff[0].count_ones() + diff[1].count_ones() + (diff[2] & low_mask(N - 128)).count_ones())
        as u16
}
#[doc(hidden)]
pub fn inverse_permute(
    candidate_sys: &PackedCodeword,
    sys_to_natural: &[u16; N],
) -> PackedCodeword {
    let mut out = [0; 3];
    for (s, &n) in sys_to_natural.iter().enumerate() {
        let bit = (candidate_sys[s / 64] >> (s % 64)) & 1;
        out[n as usize / 64] |= bit << (n % 64);
    }
    out
}
#[doc(hidden)]
pub fn extract_natural_message_with_crc(
    candidate_sys: &PackedCodeword,
    natural_to_sys: &[u16; N],
) -> PackedMessage {
    let mut out = [0; 2];
    for n in 0..91 {
        let s = natural_to_sys[n] as usize;
        let bit = (candidate_sys[s / 64] >> (s % 64)) & 1;
        out[n / 64] |= bit << (n % 64);
    }
    out
}
