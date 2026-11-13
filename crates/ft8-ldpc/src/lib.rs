//! LDPC decoding for the FT8 and FT4 (174,91) code.
//!
//! Each decoder accepts 174 log-likelihood ratios in natural codeword order.
//! A positive LLR favors bit 0 and a negative LLR favors bit 1, corresponding
//! to BPSK symbols `(-1)^bit`. At exactly zero (including negative zero),
//! Wu2007 selects bit 0; PB-OSD and min-sum select bit 1.
//!
//! Decoder-specific result types describe convergence, CRC selection, and
//! distance units. Configuration uses public fields, `Default`, and a
//! `validate` method.

#![deny(missing_docs)]

#[cfg(feature = "fast-min-sum")]
pub mod fast_min_sum;
#[cfg(feature = "pb-osd")]
pub mod pb_osd;
#[cfg(feature = "wu2007")]
pub mod wu2007;
