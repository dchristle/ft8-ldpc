//! CRC-14 calculation and checking for the 77-bit FT8 and FT4 payload.

mod compute;
pub mod constants;
mod core;

pub use compute::{
    compute_crc14_from_bits, verify_codeword_crc14, verify_crc14, verify_crc14_bitpacked,
    verify_crc14_packed_message,
};
pub use core::{add_crc14, compute_crc14};
