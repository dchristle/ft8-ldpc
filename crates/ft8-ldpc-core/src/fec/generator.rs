//! Generator matrices for the (174,91) LDPC code shared by FT8 and FT4.
//!
//! # Generators
//!
//! - [`BaseGeneratorMatrix`]: Standard (174,91) LDPC generator matrix in systematic form
//! - [`CascadedGeneratorMatrix`]: Generator with 0 to 14 CRC bits folded into the LDPC encoding
//!
//! # CRC cascading
//!
//! A cascaded generator mixes the last `n` CRC-14 bits (0 ≤ n ≤ 14) into the LDPC code. Each
//! mixed CRC bit is a fixed XOR of payload bits, so the CRC and the LDPC encoder combine into one
//! effective generator matrix over GF(2):
//! - Input size becomes (91 - n) bits instead of the full 91 bits
//! - The generator computes the mixed CRC bits during encoding
//! - The first (14 - n) CRC bits remain inputs, used only for error detection
//! - With `n = 0` nothing is mixed in, and the generator is the plain (174,91) systematic one
//!
//! The CRC matrices are built at compile time. The 15 combined generators are built on first
//! use as bit-packed GF(2) products of the LDPC generator and the CRC matrix.

use crate::crc::add_crc14;
use crate::fec::constants::GENERATOR_PARITY_EQUATIONS;
use crate::fec::crc_tables::packed_crc_generator;
use crate::protocol::{K, M, N, PAYLOAD_BITS};
use once_cell::sync::Lazy;

#[allow(dead_code)]
const PAD_LEN: usize = 5;
#[allow(dead_code)]
const PADDED_LEN: usize = PAYLOAD_BITS + PAD_LEN;

/// Systematic LDPC generator for the 91-bit message-with-CRC input.
pub const BASE_GENERATOR_MATRIX: BaseGeneratorMatrix = BaseGeneratorMatrix::new();

/// Generator matrix for the (174,91) LDPC code.
///
/// Uses the column-vector convention `c = Gm`, with `G = [I; P]` of size
/// 174×91. Only the 83×91 parity matrix `P` is stored; the first 91 codeword
/// bits are copied from the message.
pub struct BaseGeneratorMatrix {
    // For each parity bit (83), store which message bits (0..91) contribute to it
    parity_equations: [[bool; K]; M],
    packed_equations: [[u64; 2]; M],
    h_packed: [[u64; 3]; M],
}

impl Default for BaseGeneratorMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseGeneratorMatrix {
    /// Constructs the fixed systematic generator and its parity-check matrix.
    pub const fn new() -> Self {
        let parity_equations = GENERATOR_PARITY_EQUATIONS;

        // Pack bool arrays into u64s during initialization
        let mut packed_equations = [[0u64; 2]; M];
        let mut i = 0;
        while i < M {
            let mut j = 0;
            while j < K {
                if parity_equations[i][j] {
                    packed_equations[i][j / 64] |= 1u64 << (j % 64);
                }
                j += 1;
            }
            i += 1;
        }

        let mut h_packed = [[0u64; 3]; M];

        // In this column-vector convention, H = [P | I]; negation is identity over GF(2).
        i = 0;
        while i < M {
            let mut j = 0;
            while j < K {
                if parity_equations[i][j] {
                    h_packed[i][j / 64] ^= 1u64 << (j % 64);
                }
                j += 1;
            }
            i += 1;
        }

        // Second part: 83×83 identity matrix
        // Position starts at bit 91 in our packed representation
        i = 0;
        while i < M {
            let word_idx = (K + i) / 64;
            let bit_idx = (K + i) % 64;
            h_packed[i][word_idx] |= 1u64 << bit_idx;
            i += 1;
        }

        Self {
            parity_equations,
            packed_equations,
            h_packed,
        }
    }

    /// Encodes 91 message and CRC bits into a systematic 174-bit LDPC codeword.
    pub fn encode(&self, message: &[bool; K]) -> [bool; N] {
        self.encode_with_packed_parity(message)
    }

    /// Encodes each parity bit by summing its message contributions over GF(2).
    #[cfg(test)]
    #[allow(dead_code)]
    fn encode_simple(&self, message: &[bool; K]) -> [bool; N] {
        let mut codeword = [false; N];

        // Systematic part
        codeword[..K].copy_from_slice(message);

        // Parity bits
        for i in 0..M {
            let mut bit = false;
            for j in 0..K {
                bit ^= message[j] & self.parity_equations[i][j];
            }
            codeword[K + i] = bit;
        }

        codeword
    }

    /// Encodes Boolean input and output arrays using packed parity masks internally.
    #[inline]
    pub fn encode_with_packed_parity(&self, message: &[bool; K]) -> [bool; N] {
        let mut codeword = [false; N];

        // Systematic part
        codeword[..K].copy_from_slice(message);

        let mut msg_packed = [0u64; 2];

        // Process 64 bits at a time
        let msg_slice = message.as_ref();
        for (i, &bit) in msg_slice.iter().enumerate().take(64.min(K)) {
            if bit {
                msg_packed[0] |= 1u64 << i;
            }
        }

        for (i, &bit) in msg_slice.iter().enumerate().take(K).skip(64) {
            if bit {
                msg_packed[1] |= 1u64 << (i - 64);
            }
        }

        // Compute two parity bits per iteration.
        let half = M / 2;
        for i in 0..half {
            let idx0 = i * 2;
            let idx1 = idx0 + 1;

            // Compute two parity bits at once
            let bit0 = ((msg_packed[0] & self.packed_equations[idx0][0]).count_ones()
                + (msg_packed[1] & self.packed_equations[idx0][1]).count_ones())
                & 1
                != 0;
            let bit1 = ((msg_packed[0] & self.packed_equations[idx1][0]).count_ones()
                + (msg_packed[1] & self.packed_equations[idx1][1]).count_ones())
                & 1
                != 0;

            codeword[K + idx0] = bit0;
            codeword[K + idx1] = bit1;
        }

        // Handle odd case
        if M & 1 != 0 {
            let i = M - 1;
            let bit = ((msg_packed[0] & self.packed_equations[i][0]).count_ones()
                + (msg_packed[1] & self.packed_equations[i][1]).count_ones())
                & 1
                != 0;
            codeword[K + i] = bit;
        }

        codeword
    }

    /// Get a reference to the parity equations for internal use
    #[allow(dead_code)]
    pub(crate) const fn parity_equations(&self) -> &[[bool; K]; M] {
        &self.parity_equations
    }

    /// Get a specific bit from the parity equations matrix
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) const fn get_parity_equation_bit(&self, row: usize, col: usize) -> bool {
        self.parity_equations[row][col]
    }

    /// Get a reference to the packed equations for internal use
    #[allow(dead_code)]
    pub(crate) fn packed_equations(&self) -> &[[u64; 2]; M] {
        &self.packed_equations
    }

    /// Checks LDPC parity in natural codeword order without verifying the CRC.
    pub fn is_valid_codeword(&self, received: &[bool; N]) -> bool {
        let mut received_packed = [0u64; 3];
        for i in 0..N {
            if received[i] {
                received_packed[i / 64] |= 1u64 << (i % 64);
            }
        }

        // Compute syndrome: H * received^T should be zero
        for i in 0..M {
            let mut syndrome_bit = false;
            for (j, &received_word) in received_packed.iter().enumerate() {
                syndrome_bit ^= (received_word & self.h_packed[i][j]).count_ones() & 1 != 0;
            }
            if syndrome_bit {
                return false; // Non-zero syndrome bit means invalid codeword
            }
        }
        true
    }
}

/// Appends CRC-14 to a 77-bit payload and encodes the resulting 91 bits.
pub fn encode_174_91(payload: &[bool; 77]) -> [bool; N] {
    let message_with_crc = add_crc14(payload);
    BASE_GENERATOR_MATRIX.encode(&message_with_crc)
}

/// Converts bytes back into a fixed-size array of bools.
/// The bytes are unpacked in MSB-first order.
#[allow(dead_code)]
fn bytes_to_bool_array<const N: usize>(bytes: &[u8], num_bits: usize) -> [bool; N] {
    let mut result = [false; N];
    for i in 0..num_bits.min(N) {
        result[i] = (bytes[i / 8] & (1 << (7 - (i % 8)))) != 0;
    }
    result
}

#[allow(dead_code)]
fn bool_slice_to_bytes(bits: &[bool]) -> Vec<u8> {
    let num_bytes = bits.len().div_ceil(8);
    let mut bytes = vec![0u8; num_bytes];

    assert!(
        bits.len() <= num_bytes * 8,
        "Bit length {} exceeds capacity of {} bytes",
        bits.len(),
        num_bytes
    );

    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8); // MSB first
            assert!(
                byte_idx < bytes.len(),
                "Byte index {} out of bounds for {} bytes",
                byte_idx,
                bytes.len()
            );
            bytes[byte_idx] |= 1 << bit_idx;
        }
    }
    bytes
}

/// Extract CRC value from binary array
#[allow(dead_code)]
pub(crate) fn extract_crc_bits_to_u16(crc_bits: &[bool]) -> u16 {
    crc_bits
        .iter()
        .fold(0u16, |acc, &bit| (acc << 1) | u16::from(bit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    const CODEWORD: &str = concat!(
        "00000000000100111100101100010101010010001010111100111101011100010010001000000",
        "00000100000100",
        "01000101111001111000100011010011001010000110100011000010000001010010010101001000111"
    );

    #[test]
    fn test_ldpc_encoding() {
        let generator = BaseGeneratorMatrix::new();

        let message: [bool; K] = std::array::from_fn(|i| CODEWORD.as_bytes()[i] == b'1');
        let expected: [bool; N] = std::array::from_fn(|i| CODEWORD.as_bytes()[i] == b'1');

        let cw = generator.encode(&message);
        assert_eq!(cw, expected, "Encoding doesn't match the expected codeword");
    }

    #[test]
    fn test_encode_methods_match() {
        let matrix = BaseGeneratorMatrix::new();
        let mut rng = rand::rng();

        for test_num in 0..10 {
            let mut message = [false; K];
            for bit in message.iter_mut() {
                *bit = rng.random_bool(0.5);
            }

            let encoded_optimized = matrix.encode(&message);
            let encoded_simple = matrix.encode_simple(&message);

            assert_eq!(
                encoded_optimized, encoded_simple,
                "Encoding mismatch for test vector {}",
                test_num
            );

            assert_eq!(encoded_optimized.len(), 174);
            assert_eq!(encoded_simple.len(), 174);
            assert_eq!(&encoded_optimized[..K], &message);
            assert_eq!(&encoded_simple[..K], &message);
        }
    }

    #[test]
    fn test_encode_with_packed_parity_comprehensive() {
        let matrix = BaseGeneratorMatrix::new();
        let mut rng = rand::rng();

        // Test 1: Edge cases
        let edge_cases = [
            [false; K], // All zeros
            [true; K],  // All ones
        ];

        for (i, message) in edge_cases.iter().enumerate() {
            let encoded_original = matrix.encode(message);
            let encoded_packed = matrix.encode_with_packed_parity(message);

            assert_eq!(
                encoded_original, encoded_packed,
                "Edge case {} failed: original vs packed mismatch",
                i
            );

            // Verify systematic property
            assert_eq!(&encoded_original[..K], message);
            assert_eq!(&encoded_packed[..K], message);
        }

        // Test 2: Alternating patterns
        let mut alternating = [false; K];
        for i in 0..K {
            alternating[i] = i.is_multiple_of(2);
        }
        let encoded_original = matrix.encode(&alternating);
        let encoded_packed = matrix.encode_with_packed_parity(&alternating);
        assert_eq!(
            encoded_original, encoded_packed,
            "Alternating pattern failed: original vs packed mismatch"
        );

        // Test 3: Random vectors (100 test cases)
        for test_num in 0..100 {
            let mut message = [false; K];
            for bit in message.iter_mut() {
                *bit = rng.random_bool(0.5);
            }

            let encoded_original = matrix.encode(&message);
            let encoded_packed = matrix.encode_with_packed_parity(&message);

            assert_eq!(
                encoded_original, encoded_packed,
                "Random test vector {} failed: original vs packed mismatch",
                test_num
            );

            // Verify systematic property
            assert_eq!(&encoded_original[..K], &message);
            assert_eq!(&encoded_packed[..K], &message);

            // Verify length
            assert_eq!(encoded_original.len(), N);
            assert_eq!(encoded_packed.len(), N);

            // Verify both produce valid codewords
            assert!(
                matrix.is_valid_codeword(&encoded_original),
                "Original encoding produced invalid codeword for test {}",
                test_num
            );
            assert!(
                matrix.is_valid_codeword(&encoded_packed),
                "Packed encoding produced invalid codeword for test {}",
                test_num
            );
        }

        let message: [bool; K] = std::array::from_fn(|i| CODEWORD.as_bytes()[i] == b'1');

        let encoded_original = matrix.encode(&message);
        let encoded_packed = matrix.encode_with_packed_parity(&message);

        assert_eq!(
            encoded_original, encoded_packed,
            "Expected-codeword test failed: encoding methods disagree"
        );
    }

    #[test]
    fn test_valid_codeword() {
        let code = BaseGeneratorMatrix::new();

        // Generate a valid codeword
        let message = [false; K]; // all-zero message
        let codeword = code.encode(&message);

        assert!(
            code.is_valid_codeword(&codeword),
            "Generated codeword should be valid"
        );

        // Corrupt one bit and verify detection
        let mut corrupted = codeword;
        corrupted[0] ^= true;
        assert!(
            !code.is_valid_codeword(&corrupted),
            "Single bit error should be detected"
        );
    }
}

const CRC_MAX: usize = 14;

static CASCADED_MATRICES: Lazy<[Box<[[u64; 2]]>; CRC_MAX + 1]> = Lazy::new(|| {
    let mut matrices = Vec::with_capacity(CRC_MAX + 1);
    for num_cascaded_bits in 0..=CRC_MAX {
        let crc_generator = get_packed_crc_generator(num_cascaded_bits);
        let matrix =
            fast_matrix_multiply(&PACKED_LDPC_GENERATOR, crc_generator, K - num_cascaded_bits);
        matrices.push(matrix);
    }
    matrices.try_into().unwrap()
});

const PACKED_LDPC_GENERATOR: [[u64; 2]; N] = {
    // Initialize the identity (first K rows)
    let mut generator = [[0u64; 2]; N];
    let mut i = 0;
    while i < K {
        generator[i][i / 64] |= 1u64 << (i % 64);
        i += 1;
    }
    // Initialize the parity portion (last N-K rows)
    let mut i = 0;
    while i < (N - K) {
        let mut j = 0;
        while j < K {
            if BASE_GENERATOR_MATRIX.parity_equations[i][j] {
                generator[K + i][j / 64] |= 1u64 << (j % 64);
            }
            j += 1;
        }
        i += 1;
    }
    generator
};

/// Failure to construct a generator matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeneratorError {
    /// The requested number of cascaded CRC bits exceeds the supported maximum.
    CascadeBitsOutOfRange {
        /// Requested number of cascaded CRC bits.
        value: usize,
        /// Maximum supported number of cascaded CRC bits.
        max: usize,
    },
}

impl std::fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CascadeBitsOutOfRange { max, .. } => {
                write!(f, "Cannot cascade more than {max} bits")
            }
        }
    }
}

impl std::error::Error for GeneratorError {}

/// Generator matrix with 0 to 14 CRC bits folded into the LDPC code.
///
/// CascadedGeneratorMatrix stores a 174×91 (N×k) matrix when num_cascaded_bits = 0,
/// in row-major format. Note that the *full* generator matrix, including the
/// systematic portion (an identity matrix) for the first k bits, is stored.
#[derive(Debug, Clone)]
pub struct CascadedGeneratorMatrix {
    /// Number of CRC bits cascaded into the generator matrix.
    num_cascaded_bits: usize,
    /// The actual generator matrix, represented as a packed bit array.
    /// Each row has 2 u64 values (128 bits) which is more than enough for
    /// the maximum 91 columns needed.
    generator: Box<[[u64; 2]]>,
}

impl CascadedGeneratorMatrix {
    /// Constructs a generator that computes the final `num_cascaded_bits` CRC bits.
    ///
    /// # Errors
    /// Returns [`GeneratorError::CascadeBitsOutOfRange`] for values greater than 14.
    pub fn new(num_cascaded_bits: usize) -> Result<Self, GeneratorError> {
        if num_cascaded_bits > 14 {
            return Err(GeneratorError::CascadeBitsOutOfRange {
                value: num_cascaded_bits,
                max: 14,
            });
        }
        let generator = CASCADED_MATRICES[num_cascaded_bits].clone();

        Ok(Self {
            num_cascaded_bits,
            generator,
        })
    }

    /// Encodes payload bits followed by any CRC bits not folded into the generator.
    ///
    /// # Panics
    /// Panics unless the input contains exactly `91 - self.num_cascaded_bits()` bits.
    #[inline]
    pub fn encode(&self, message: &[bool]) -> Vec<bool> {
        let input_size = K - self.num_cascaded_bits;
        assert_eq!(
            message.len(),
            input_size,
            "Message length {} must match matrix input dimension {}",
            message.len(),
            input_size
        );

        // Pack message into u64 words
        let mut msg_packed = [0u64; 2];
        for (i, &bit) in message.iter().enumerate() {
            if bit {
                msg_packed[i / 64] |= 1u64 << (i % 64);
            }
        }

        // Initialize result vector
        let mut result = vec![false; N];

        // For each row in the generator matrix
        for (i, row) in self.generator.iter().enumerate() {
            // Compute dot product using population count
            let bit =
                ((row[0] & msg_packed[0]).count_ones() + (row[1] & msg_packed[1]).count_ones()) & 1;

            result[i] = bit != 0;
        }

        result
    }

    /// Encodes the first `91 - self.num_cascaded_bits()` LSB-first input bits into 174 packed bits.
    #[inline]
    pub fn encode_packed(&self, message_packed: [u64; 2]) -> [u64; 3] {
        let mut result = [0u64; 3]; // 174 bits requires 3 u64s

        // For each row in the generator matrix
        for i in 0..N {
            let row = &self.generator[i];

            // Compute dot product using population count
            let bit = ((row[0] & message_packed[0]).count_ones()
                + (row[1] & message_packed[1]).count_ones())
                & 1;

            // Set the corresponding bit in the result
            if bit != 0 {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }

        result
    }

    /// Returns the number of CRC bits computed by the generator.
    pub fn num_cascaded_bits(&self) -> usize {
        self.num_cascaded_bits
    }

    /// Returns 174 codeword rows with LSB-first packed message columns.
    pub fn generator(&self) -> &[[u64; 2]] {
        &self.generator
    }

    /// Reads a codeword row and message column, with `row < 174` and `col < 91 - self.num_cascaded_bits()`.
    #[inline]
    pub fn bit(&self, row: usize, col: usize) -> bool {
        let word_idx = col / 64;
        let bit_idx = col % 64;
        (self.generator[row][word_idx] & (1u64 << bit_idx)) != 0
    }
}

#[inline]
fn get_packed_crc_generator(num_cascaded_bits: usize) -> &'static [[u64; 2]; K] {
    packed_crc_generator(num_cascaded_bits)
}

/// Multiplies two packed matrices over GF(2), with each row stored in two `u64` words.
pub(crate) fn fast_matrix_multiply(
    a: &[[u64; 2]], // First matrix dimensions: rows_a × K (91 columns)
    b: &[[u64; 2]], // Second matrix dimensions: K × cols_b
    cols_b: usize,  // Number of columns in matrix b (and the result)
) -> Box<[[u64; 2]]> {
    let rows_a = a.len();

    // Calculate how many u64s needed per row of the result
    let words_per_row = cols_b.div_ceil(64);
    assert!(words_per_row <= 2, "Result matrix exceeds 128 columns");

    // Allocate result matrix with zeros
    let mut result = vec![[0u64; 2]; rows_a];

    // For each row in matrix a
    for i in 0..rows_a {
        // For each column in matrix b (process 64 at a time)
        for j_chunk in 0..words_per_row {
            let chunk_start = j_chunk * 64;
            let chunk_end = (chunk_start + 64).min(cols_b);
            let bits_in_chunk = chunk_end - chunk_start;

            if bits_in_chunk == 0 {
                continue;
            }

            // For each column position within this chunk
            for j_offset in 0..bits_in_chunk {
                let j = chunk_start + j_offset;

                // Compute dot product for this position
                let bit: bool;

                // For each row of b / column of a (91 bits)
                // Process 64 bits at a time using population count
                let dot0 = (a[i][0] & get_column_bits(b, j, 0)).count_ones() & 1;
                let dot1 = (a[i][1] & get_column_bits(b, j, 1)).count_ones() & 1;

                bit = (dot0 ^ dot1) != 0;

                // Set result bit
                if bit {
                    result[i][j_chunk] |= 1u64 << j_offset;
                }
            }
        }
    }

    result.into_boxed_slice()
}

/// Helper function to extract a column from packed matrix as a u64
///
/// This gets all bits in column j for word_offset (0 or 1)
#[inline]
fn get_column_bits(matrix: &[[u64; 2]], col: usize, word_offset: usize) -> u64 {
    let mut result = 0u64;
    let bit_in_word = col % 64;

    // If word_offset is 0, we process rows 0-63
    // If word_offset is 1, we process rows 64-127 (or fewer if matrix is smaller)
    let start_row = word_offset * 64;
    let end_row = ((word_offset + 1) * 64).min(matrix.len());

    for (i, row) in matrix.iter().enumerate().take(end_row).skip(start_row) {
        // If the bit at (i, col) is set, set the corresponding bit in result
        if (row[col / 64] & (1u64 << bit_in_word)) != 0 {
            // Map the row index back to 0-63 range for the result
            result |= 1u64 << (i - start_row);
        }
    }

    result
}

/// Converts a traditional Vec<Vec<bool>> matrix to a packed bit matrix
#[cfg(test)]
pub(crate) fn pack_matrix(matrix: &[Vec<bool>]) -> Box<[[u64; 2]]> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    assert!(cols <= 128, "Matrix too wide for packing (max 128 columns)");

    let mut packed = vec![[0u64; 2]; rows];

    for i in 0..rows {
        for j in 0..cols {
            if matrix[i][j] {
                packed[i][j / 64] |= 1u64 << (j % 64);
            }
        }
    }

    packed.into_boxed_slice()
}

/// Converts a packed bit matrix back to Vec<Vec<bool>>
#[cfg(test)]
pub(crate) fn unpack_matrix(packed: &[[u64; 2]], rows: usize, cols: usize) -> Vec<Vec<bool>> {
    let mut matrix = vec![vec![false; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            if (packed[i][j / 64] & (1u64 << (j % 64))) != 0 {
                matrix[i][j] = true;
            }
        }
    }

    matrix
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
mod cascaded_tests {
    use super::*;
    use crate::crc::add_crc14;
    use rand::Rng;

    const TEST_PAYLOAD_BITS: usize = 77;
    const TEST_N: usize = 174;
    const TEST_K: usize = 91;
    const PAYLOAD: &str =
        "00000000000000000000000000100000010011011111110011011100100010100001010000001";

    #[test]
    fn cascade_count_error_preserves_requested_value_and_limit() {
        assert!(CascadedGeneratorMatrix::new(14).is_ok());
        for value in [15, usize::MAX] {
            let error = CascadedGeneratorMatrix::new(value).unwrap_err();
            assert_eq!(
                error,
                GeneratorError::CascadeBitsOutOfRange { value, max: 14 }
            );
            assert_eq!(error.to_string(), "Cannot cascade more than 14 bits");
        }
    }

    #[test]
    fn test_cascaded_systematic_form() {
        let num_cascaded_bits = 0;
        let k = 91 - num_cascaded_bits;
        let gen = CascadedGeneratorMatrix::new(num_cascaded_bits).unwrap();

        // Test systematic property for each input bit
        for i in 0..91 {
            let mut message = vec![false; k];
            message[i] = true;

            let codeword = gen.encode(&message);
            assert!(codeword[i], "Systematic property failed at position {}", i);
        }
    }

    #[test]
    fn test_cascaded_generator_zero_bits() {
        let num_cascaded_bits = 0;
        let _k = 91 - num_cascaded_bits;
        let gen_cascaded = CascadedGeneratorMatrix::new(num_cascaded_bits).unwrap();

        // Test systematic part (first 77 rows should have exactly one 1 in their corresponding column)
        for i in 0..77 {
            for j in 0..91 {
                if i == j {
                    assert!(
                        gen_cascaded.bit(i, j),
                        "Missing 1 on diagonal at ({},{})",
                        i,
                        j
                    );
                } else {
                    assert!(
                        !gen_cascaded.bit(i, j),
                        "Unexpected 1 off diagonal in systematic part at ({},{})",
                        i,
                        j
                    );
                }
            }
        }

        // Test that non-cascaded CRC bit positions (columns 77-90) are systematic
        for j in 77..91 {
            for i in 0..91 {
                if i == j {
                    assert!(
                        gen_cascaded.bit(i, j),
                        "Missing 1 on diagonal in CRC part at ({},{})",
                        i,
                        j
                    );
                } else if i < 77 {
                    // Only check rows corresponding to message bits
                    assert!(
                        !gen_cascaded.bit(i, j),
                        "Unexpected 1 in CRC column at ({},{})",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_cascaded_encode() {
        // Test with different numbers of cascaded bits
        for num_cascaded_bits in [0, 4, 14] {
            let gen_cascaded = CascadedGeneratorMatrix::new(num_cascaded_bits).unwrap();

            // Create test message and get its correct CRC
            let msg = PAYLOAD;
            let msg_bits = binary_str_to_bools::<TEST_PAYLOAD_BITS>(msg);
            let with_crc = add_crc14(&msg_bits);

            // Create input for partial generator matrix:
            // - First 77 bits are the payload
            // - Next (14 - num_cascaded_bits) bits are from the CRC
            let input_size = 91 - num_cascaded_bits;
            let mut input = Vec::with_capacity(input_size);
            input.extend_from_slice(&msg_bits);
            input.extend_from_slice(
                &with_crc[TEST_PAYLOAD_BITS..TEST_PAYLOAD_BITS + (14 - num_cascaded_bits)],
            );
            assert_eq!(input.len(), input_size);

            let result = gen_cascaded.encode(&input);
            assert_eq!(result.len(), TEST_N, "Output should be 174 bits");
            // Verify all 91 bits match the original CRC computation
            for i in 0..91 {
                assert_eq!(
                    result[i], with_crc[i],
                    "Bit mismatch at position {} with {} cascaded bits",
                    i, num_cascaded_bits
                );
            }
        }
    }

    #[test]
    fn test_cascaded_generator_functionality() {
        // Test with different numbers of cascaded bits
        for num_cascaded_bits in [0, 4, 14] {
            let cascaded_gen = CascadedGeneratorMatrix::new(num_cascaded_bits).unwrap();

            let msg = PAYLOAD;
            let msg_bits = binary_str_to_bools::<TEST_PAYLOAD_BITS>(msg);
            let with_crc = add_crc14(&msg_bits);

            // Create input for cascaded generator:
            // - First 77 bits are the payload
            // - Next (14 - num_cascaded_bits) bits are from the CRC
            let mut input = Vec::with_capacity(91 - num_cascaded_bits);
            input.extend_from_slice(&msg_bits);
            input.extend_from_slice(
                &with_crc[TEST_PAYLOAD_BITS..TEST_PAYLOAD_BITS + (14 - num_cascaded_bits)],
            );

            // Get 174-bit codeword using cascaded generator
            let codeword_cascaded = cascaded_gen.encode(&input);

            // Verify correct dimensions
            assert_eq!(codeword_cascaded.len(), 174);

            // For validation, encode the traditional way using BaseGeneratorMatrix
            let codeword_traditional = BASE_GENERATOR_MATRIX.encode(&with_crc);

            // Verify both methods produce identical codewords
            assert_eq!(codeword_cascaded, codeword_traditional);
        }
    }

    #[test]
    fn test_max_cascaded_bits() {
        let gen = CascadedGeneratorMatrix::new(14).unwrap();

        let msg = PAYLOAD;
        let msg_bits = binary_str_to_bools::<TEST_PAYLOAD_BITS>(msg);
        let with_crc = add_crc14(&msg_bits);

        let input = msg_bits.to_vec(); // Only payload bits, no CRC bits
        let result = gen.encode(&input);

        assert_eq!(result.len(), TEST_N);
        for i in 0..91 {
            assert_eq!(result[i], with_crc[i], "Bit mismatch at position {}", i);
        }
    }

    #[test]
    fn test_systematic_property_with_cascaded_bits() {
        let num_cascaded_bits = 7;
        let gen = CascadedGeneratorMatrix::new(num_cascaded_bits).unwrap();

        let k = 91 - num_cascaded_bits;
        for i in 0..k {
            let mut message = vec![false; k];
            message[i] = true;

            let codeword = gen.encode(&message);
            assert!(codeword[i], "Systematic property failed at position {}", i);
        }
    }

    #[test]
    fn test_codeword_generation_with_zero_cascaded_bits() {
        let cascaded_gen = CascadedGeneratorMatrix::new(0).unwrap();

        // Test with multiple random messages
        let mut rng = rand::rng();
        for _ in 0..10 {
            // Generate a random 91-bit message-with-CRC
            let message: Vec<bool> = (0..91).map(|_| rng.random()).collect();

            // Encode using BaseGeneratorMatrix
            let mut message_array = [false; 91];
            message_array.copy_from_slice(&message);
            let codeword_base = BASE_GENERATOR_MATRIX.encode(&message_array);

            // Encode using CascadedGeneratorMatrix
            let codeword_cascaded = cascaded_gen.encode(&message);

            assert_eq!(
                codeword_cascaded, codeword_base,
                "Codewords generated by CascadedGeneratorMatrix and BaseGeneratorMatrix do not match"
            );

            // Validate codeword using BaseGeneratorMatrix's parity-check matrix
            let mut codeword_array = [false; TEST_N];
            codeword_array.copy_from_slice(&codeword_cascaded);
            assert!(
                BASE_GENERATOR_MATRIX.is_valid_codeword(&codeword_array),
                "Codeword generated by CascadedGeneratorMatrix failed validation"
            );
        }
    }

    #[test]
    fn test_systematic_property_for_all_cascaded_bits() {
        let mut rng = rand::rng();

        // Test for all valid num_cascaded_bits
        for num_cascaded_bits in 0..=14 {
            let k = 91 - num_cascaded_bits;
            let cascaded_gen = CascadedGeneratorMatrix::new(num_cascaded_bits).unwrap();

            // Generate a random k-bit message
            let message: Vec<bool> = (0..k).map(|_| rng.random()).collect();

            // Encode using CascadedGeneratorMatrix
            let codeword = cascaded_gen.encode(&message);

            // Verify systematic property
            assert_eq!(
                &codeword[..k],
                &message[..],
                "Systematic property failed for num_cascaded_bits = {}",
                num_cascaded_bits
            );
        }
    }

    #[test]
    fn test_codeword_validation_for_all_cascaded_bits() {
        let base = BaseGeneratorMatrix::new();
        let mut rng = rand::rng();

        // Test for all valid num_cascaded_bits
        for num_cascaded_bits in 0..=14 {
            let k = 91 - num_cascaded_bits;
            let cascaded_gen = CascadedGeneratorMatrix::new(num_cascaded_bits).unwrap();

            // Generate a random k-bit message
            let message: Vec<bool> = (0..k).map(|_| rng.random()).collect();

            // Encode using CascadedGeneratorMatrix
            let codeword = cascaded_gen.encode(&message);

            // Convert to fixed-length array for validation
            let mut codeword_array = [false; TEST_N];
            codeword_array.copy_from_slice(&codeword);

            // Validate codeword using BaseGeneratorMatrix's parity-check matrix
            assert!(
                base.is_valid_codeword(&codeword_array),
                "Codeword failed validation for num_cascaded_bits = {}",
                num_cascaded_bits
            );
        }
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let mut rng = rand::rng();
        let rows = 10;
        let cols = 15;

        // Create random boolean matrix
        let mut original = vec![vec![false; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                original[i][j] = rng.random();
            }
        }

        // Pack and unpack
        let packed = pack_matrix(&original);
        let unpacked = unpack_matrix(&packed, rows, cols);

        // Verify round-trip conversion
        assert_eq!(original, unpacked);
    }

    #[test]
    fn test_matrix_multiply_simple() {
        // Create simple test matrices
        let mut a = vec![[0u64; 2]; 3]; // 3x3 identity matrix
        let mut b = vec![[0u64; 2]; 3]; // 3x3 identity matrix

        // Set identity bits
        for i in 0..3 {
            a[i][0] |= 1u64 << i;
            b[i][0] |= 1u64 << i;
        }

        // Multiply
        let result = fast_matrix_multiply(&a, &b, 3);

        // Result should be identity matrix
        for i in 0..3 {
            for j in 0..3 {
                let bit = (result[i][j / 64] & (1u64 << (j % 64))) != 0;
                assert_eq!(bit, i == j, "Result[{},{}] should be {}", i, j, i == j);
            }
        }
    }

    #[test]
    fn test_get_column_bits() {
        // Create test matrix with specific pattern
        let mut matrix = vec![[0u64; 2]; 4];

        // Set bits in column 2
        matrix[0][0] |= 1u64 << 2; // Set bit (0,2)
        matrix[2][0] |= 1u64 << 2; // Set bit (2,2)

        // Get column bits
        let bits = get_column_bits(&matrix, 2, 0);

        // Should have bits 0 and 2 set (representing rows 0 and 2)
        assert_eq!(bits, 0b101);
    }

    #[test]
    fn test_packed_encode() {
        let gen = CascadedGeneratorMatrix::new(4).unwrap();

        // Create random message
        let mut rng = rand::rng();
        let input_size = TEST_K - 4;
        let message: Vec<bool> = (0..input_size).map(|_| rng.random()).collect();

        // Pack message
        let mut msg_packed = [0u64; 2];
        for (i, &bit) in message.iter().enumerate() {
            if bit {
                msg_packed[i / 64] |= 1u64 << (i % 64);
            }
        }

        // Encode using both methods
        let normal_result = gen.encode(&message);
        let packed_result = gen.encode_packed(msg_packed);

        // Unpack the packed result for comparison
        let mut unpacked = vec![false; TEST_N];
        for i in 0..TEST_N {
            unpacked[i] = (packed_result[i / 64] & (1u64 << (i % 64))) != 0;
        }

        assert_eq!(
            normal_result, unpacked,
            "Normal and packed encoding results don't match"
        );
    }
}
