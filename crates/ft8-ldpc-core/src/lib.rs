//! CRC-14, LDPC encoding, generator matrices, and protocol constants for FT8 and FT4.
//!
//! [`fec::encode_174_91`] appends CRC and parity bits to a 77-bit payload, while
//! [`fec::BaseGeneratorMatrix`] accepts all 91 message and CRC bits.
//! Packed arrays use bit `i % 64` of word `i / 64` for logical bit `i`, unless
//! an API explicitly specifies byte-oriented MSB-first input.

#![deny(missing_docs)]

pub mod crc;
pub mod fec;
pub mod protocol;
