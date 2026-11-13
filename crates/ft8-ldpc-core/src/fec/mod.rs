//! Generator matrices, CRC cascading and packed GF(2) algebra for the (174,91)
//! LDPC code shared by FT8 and FT4.

pub mod bit_matrix;
pub mod constants;
pub mod crc_tables;
pub mod generator;

pub use bit_matrix::{BitMatrix, BitMatrixSource};
pub use generator::{
    encode_174_91, BaseGeneratorMatrix, CascadedGeneratorMatrix, GeneratorError,
    BASE_GENERATOR_MATRIX,
};
