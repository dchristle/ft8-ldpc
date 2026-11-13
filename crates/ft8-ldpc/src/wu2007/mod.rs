//! Ordered-statistics decoding with main search and two preprocessing stages.
//!
//! See the [crate-level LLR convention](crate) for input sign and zero decisions.
//! Reuse a decoder for subsequent codewords; with candidate collection disabled,
//! decoding allocates no heap memory after construction.
//!
//! ```
//! use ft8_ldpc::wu2007::{Config, Decoder, Status};
//!
//! let mut decoder = Decoder::new(Config::default()).expect("valid configuration");
//! let result = decoder.decode(&[4.0_f32; 174]);
//! assert_eq!(result.status, Status::Success);
//! assert!(result.crc_valid);
//! assert_eq!(result.payload, [0; 2]);
//! ```
#![deny(missing_docs)]

mod basis;
mod config;
mod decoder;
mod packed;
mod result;
mod root;
mod scoring;

pub use basis::RedundancyOrder;
pub use config::{Config, ConfigError, SelectionPolicy};
pub use result::{Candidate, DecodeResult, Diagnostics, Status, CANDIDATE_POSITION_CAPACITY};
pub use root::Decoder;
