//! FT8 frame and tone dimensions, and payload, CRC and LDPC dimensions shared
//! by FT8 and FT4.

// LDPC code dimensions.

/// LDPC codeword length in bits.
pub const N: usize = 174;

/// LDPC message length in bits, including the 77-bit payload and 14-bit CRC.
pub const K: usize = 91;

/// LDPC parity length in bits.
pub const M: usize = N - K;

// Message and CRC dimensions.

/// FT8 payload length in bits.
pub const PAYLOAD_BITS: usize = 77;

/// FT8 CRC length in bits.
pub const CRC_LEN: usize = 14;

/// FT8 payload plus CRC length in bits.
pub const MESSAGE_WITH_CRC_BITS: usize = PAYLOAD_BITS + CRC_LEN;

// Symbol and tone dimensions.

/// Total symbols in an FT8 transmission.
pub const FT8_NSYM: usize = 79;

/// FT8 data symbols carrying the 174-bit LDPC codeword.
pub const FT8_DATA_SYM: usize = 58;

/// FT8 synchronization symbols across the three Costas arrays.
pub const FT8_SYNC_SYM: usize = 21;

/// Number of tones in FT8 8-FSK modulation.
pub const FT8_NUM_TONES: usize = 8;
