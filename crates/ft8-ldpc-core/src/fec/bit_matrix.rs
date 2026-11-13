//! Generator matrices and Gaussian elimination for FT8's (174,k) linear block code.
//!
//! Generator storage is N×k, indexed by codeword coordinate then message bit.
//! Standard storage is its k×N transpose; systematic standard matrices have
//! the form `[I | P]`. These representations and their caches are updated
//! explicitly by the corresponding methods.
//
// Matrix convention for a binary linear block code C(n,k):
// - Generator matrix G is k×n (k rows, n columns)
// - Message vector b is 1×k
// - Codeword c = bG is 1×n
//
// For FT8's (174,k) code:
// - n = 174 (fixed)
// - k ∈ [77,91] (varies due to optional CRC cascading)
// - Column-major storage groups the entries swapped during reliability
//   ordering (π₁) and Gaussian elimination (π₂).
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
// Reliability ordering can place dependent columns among the first k positions.
// Gaussian elimination selects an independent basis and restores G = [I | P].
// Inverse coordinate permutations recover natural codeword order.
use crate::fec::CascadedGeneratorMatrix;
use crate::protocol::N;

/// Generator input in cascaded or unpacked Boolean form.
#[non_exhaustive]
pub enum BitMatrixSource {
    /// A generator with CRC bits folded into its encoding equations.
    Cascaded(CascadedGeneratorMatrix),
    /// An N×k matrix indexed by codeword coordinate, then message bit.
    Bool(Vec<Vec<bool>>),
}

impl From<CascadedGeneratorMatrix> for BitMatrixSource {
    fn from(gen: CascadedGeneratorMatrix) -> Self {
        BitMatrixSource::Cascaded(gen)
    }
}

impl From<Vec<Vec<bool>>> for BitMatrixSource {
    fn from(matrix: Vec<Vec<bool>>) -> Self {
        BitMatrixSource::Bool(matrix)
    }
}

/// Packed generator and standard matrices with explicitly refreshed derived caches.
#[derive(Clone)]
pub struct BitMatrix {
    k: usize,                        // Number of columns (77 to 91)
    generator: [[u64; 2]; N],        // N rows, each containing up to K = 91 bits
    pi1: [usize; N],                 // Tracks all 174 positions
    transformation: [[u64; 2]; 91],  // *Column-major order*. Tracks column XOR operations.
    standard_matrix: [[u64; 3]; 91], // k×N matrix in standard form, bit-packed by rows
    standard_matrix_t: [[u64; 2]; N],
    parity_matrix: [[u64; 2]; 91], // k x (N-k) parity rows, bit-packed
    standard_pi1: [usize; N],      // Tracks all 174 positions for standard_matrix
    standard_pi2: [usize; N],      // Tracks all 174 positions for standard_matrix
    standard_h: [[u64; 3]; 97],    // Parity-check matrix (n−k = 83-97 rows, 174 columns)
}

impl BitMatrix {
    /// Creates zeroed matrices and caches with identity coordinate mappings.
    ///
    /// # Panics
    /// Panics unless `k` is in `77..=91`.
    pub fn new(k: usize) -> Self {
        assert!((77..=91).contains(&k), "k must be in range [77,91]");
        let mut transformation = [[0, 0]; 91];
        for i in 0..k {
            let chunk = i / 64;
            let bit = i % 64;
            transformation[i][chunk] |= 1u64 << bit;
        }
        BitMatrix {
            k,
            generator: [[0u64; 2]; N],
            pi1: std::array::from_fn(|i| i),
            transformation,
            standard_matrix: [[0u64; 3]; 91],
            standard_matrix_t: [[0_u64; 2]; N],
            parity_matrix: [[0u64; 2]; 91],
            standard_pi1: std::array::from_fn(|i| i),
            standard_pi2: std::array::from_fn(|i| i),
            standard_h: [[0u64; 3]; 97],
        }
    }

    /// Creates a matrix from a cascaded generator or an N×k Boolean array.
    pub(crate) fn from_matrix(matrix: impl Into<BitMatrixSource>, k: usize) -> Self {
        match matrix.into() {
            BitMatrixSource::Cascaded(gen) => Self::from_cascaded_generator(&gen),
            BitMatrixSource::Bool(matrix) => Self::from_bool_matrix(&matrix, k),
        }
    }

    #[allow(clippy::ptr_arg)]
    pub(crate) fn from_bool_matrix(matrix: &Vec<Vec<bool>>, k: usize) -> Self {
        assert_eq!(matrix.len(), N, "Matrix must have N rows!");
        assert_eq!(matrix[0].len(), k, "Matrix must have k columns!");
        let mut bit_matrix = Self::new(k);

        // For each column in output
        for col in 0..k {
            // For each row in input
            for row in 0..N {
                if matrix[row][col] {
                    // Access input matrix [row][col]
                    bit_matrix.set_bit(row, col); // Store in column-major format
                }
            }
        }

        bit_matrix
    }

    /// Copies N×k generator storage, leaving standard storage and caches zeroed.
    pub fn from_cascaded_generator(gen: &CascadedGeneratorMatrix) -> Self {
        let k = 91 - gen.num_cascaded_bits();
        assert!((77..=91).contains(&k));

        let mut bit_matrix = Self::new(k);

        for col in 0..k {
            for row in 0..N {
                if gen.bit(row, col) {
                    // Using the helper method
                    bit_matrix.set_bit(row, col);
                }
            }
        }

        bit_matrix
    }

    // Accessor methods for private fields

    /// Returns the number of message input bits.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get a reference to the transformation matrix for internal use
    pub(crate) fn transformation(&self) -> &[[u64; 2]; 91] {
        &self.transformation
    }

    /// Get a reference to the standard matrix for internal use
    pub(crate) fn standard_matrix(&self) -> &[[u64; 3]; 91] {
        &self.standard_matrix
    }

    /// Installs systematic `[I | P]` rows and refreshes transpose, parity, and parity-check caches.
    ///
    /// The first `k` rows contain 174 LSB-first bits in the caller's codeword
    /// coordinate order; generator storage and coordinate mappings are unchanged.
    pub fn replace_standard_rows(&mut self, rows: [[u64; 3]; 91]) {
        let mut transposed = [[0u64; 2]; N];
        let mut parity = [[0u64; 2]; 91];

        for row in 0..self.k {
            parity[row] = Self::extract_parity(&rows[row], self.k);
            for (chunk_index, &word) in rows[row].iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let column = chunk_index * 64 + bits.trailing_zeros() as usize;
                    if column < N {
                        transposed[column][row / 64] |= 1u64 << (row % 64);
                    }
                    bits &= bits - 1;
                }
            }
        }

        let mut standard_h = [[0u64; 3]; 97];
        for h_row in 0..(N - self.k) {
            standard_h[h_row][0] = transposed[self.k + h_row][0];
            standard_h[h_row][1] = transposed[self.k + h_row][1];
            let identity_column = self.k + h_row;
            standard_h[h_row][identity_column / 64] |= 1u64 << (identity_column % 64);
        }

        self.standard_matrix = rows;
        self.standard_matrix_t = transposed;
        self.parity_matrix = parity;
        self.standard_h = standard_h;
    }

    /// Get a reference to the parity matrix for internal use
    pub(crate) fn parity_matrix(&self) -> &[[u64; 2]; 91] {
        &self.parity_matrix
    }

    /// Reads a cached parity row for message position `pos < k`, packed LSB-first from column `k`.
    ///
    /// Refresh this cache with [`Self::compute_parity_matrix`] after changing standard rows.
    #[inline(always)]
    pub fn parity_row(&self, pos: usize) -> (u64, u64) {
        debug_assert!(pos < 91, "Position {pos} out of bounds for parity matrix");
        (self.parity_matrix[pos][0], self.parity_matrix[pos][1])
    }

    /// Returns the generator row mapping from current positions to source coordinates.
    pub fn pi1(&self) -> &[usize; N] {
        &self.pi1
    }

    /// Returns the source coordinate for generator row `index < 174`.
    #[inline(always)]
    pub fn permutation_index(&self, index: usize) -> usize {
        debug_assert!(index < N, "Index {index} out of bounds for pi1");
        self.pi1[index]
    }

    /// Get a reference to the standard H matrix for internal use
    pub(crate) fn standard_h(&self) -> &[[u64; 3]; 97] {
        &self.standard_h
    }

    /// Reads a cached parity-check bit, with `row < 174 - k` and `col < 174`.
    ///
    /// Refresh this cache with [`Self::compute_standard_parity_checks`] after changing standard rows.
    #[inline(always)]
    pub fn parity_check_bit(&self, row: usize, col: usize) -> bool {
        debug_assert!(row < 97, "Row index {row} out of bounds");
        debug_assert!(col < 174, "Column index {col} out of bounds");

        let chunk = col / 64;
        let bit_pos = col % 64;
        (self.standard_h[row][chunk] >> bit_pos) & 1 != 0
    }

    /// Reads N×k generator storage, with `row < 174` and `col < k`.
    #[inline(always)]
    pub fn bit(&self, row: usize, col: usize) -> bool {
        let chunk = col / 64;
        let bit = col % 64;
        (self.generator[row][chunk] & (1u64 << bit)) != 0
    }

    #[inline(always)]
    pub(crate) fn set_bit(&mut self, row: usize, col: usize) {
        let chunk = col / 64;
        let bit = col % 64;
        self.generator[row][chunk] |= 1u64 << bit;
    }

    #[inline(always)]
    pub(crate) fn clear_bit(&mut self, row: usize, col: usize) {
        let chunk = col / 64;
        let bit = col % 64;
        self.generator[row][chunk] &= !(1u64 << bit);
    }

    #[inline(always)]
    pub(crate) fn flip_bit(&mut self, row: usize, col: usize) {
        let chunk = col / 64;
        let bit = col % 64;
        self.generator[row][chunk] ^= 1u64 << bit;
    }

    #[inline(always)]
    pub(crate) fn swap_rows(&mut self, row1: usize, row2: usize) {
        if row1 == row2 {
            return;
        }
        self.generator.swap(row1, row2);
        self.pi1.swap(row1, row2);
    }

    #[inline(always)]
    pub(crate) fn swap_columns(&mut self, col1: usize, col2: usize) {
        if col1 == col2 {
            return;
        }

        let chunk1 = col1 / 64;
        let bit1 = col1 % 64;
        let chunk2 = col2 / 64;
        let bit2 = col2 % 64;

        let mask1 = 1u64 << bit1;
        let mask2 = 1u64 << bit2;

        // For each row, need to swap elements of specific columns
        for row in 0..N {
            if chunk1 == chunk2 {
                // Bits in same u64 - can use single XOR
                let bits = (self.generator[row][chunk1] >> bit1) & 1
                    ^ (self.generator[row][chunk1] >> bit2) & 1;
                if bits == 1 {
                    self.generator[row][chunk1] ^= mask1 | mask2;
                }
            } else {
                // Bits in different u64s
                let bit1_val = (self.generator[row][chunk1] >> bit1) & 1;
                let bit2_val = (self.generator[row][chunk2] >> bit2) & 1;
                if bit1_val != bit2_val {
                    self.generator[row][chunk1] ^= mask1;
                    self.generator[row][chunk2] ^= mask2;
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn apply_col_permutation(&mut self, perm: &[usize]) {
        assert_eq!(perm.len(), self.k, "Permutation must be length k");

        for row in &mut self.generator {
            // Initialize new_packed to match the row's u64 array size (2 elements)
            let mut new_packed = [0u64; 2];

            // For each new column position, determine the original column
            for new_col in 0..self.k {
                let old_col = perm[new_col];

                // Check if the bit at (row, old_col) is set
                let old_chunk = old_col / 64;
                let old_bit = old_col % 64;
                if (row[old_chunk] & (1u64 << old_bit)) != 0 {
                    // Set the bit at (row, new_col) in new_packed
                    let new_chunk = new_col / 64;
                    let new_bit = new_col % 64;
                    new_packed[new_chunk] |= 1u64 << new_bit;
                }
            }

            // Update the row with permuted columns
            *row = new_packed;
        }
    }

    pub(crate) fn extract_parity_columns(&self) -> [[u64; 2]; 91] {
        let mut parity_cols = [[0u64; 2]; 91];
        let k = self.k;

        // For each message bit, extract its contributions to parity bits
        for msg_bit in 0..k {
            for parity_bit in 0..(174 - k) {
                let chunk = parity_bit / 64;
                let bit_pos = parity_bit % 64;

                // Check if the message bit participates in this parity equation
                if self.bit(parity_bit + k, msg_bit) {
                    parity_cols[msg_bit][chunk] |= 1u64 << bit_pos;
                }
            }
        }

        parity_cols
    }

    /// Refreshes the parity cache from standard columns `k..174`, packed LSB-first.
    pub fn compute_parity_matrix(&mut self) {
        let mut parity = [[0u64; 2]; 91];
        for r in 0..self.k {
            parity[r] = Self::extract_parity(&self.standard_matrix[r], self.k);
        }
        self.parity_matrix = parity;
    }

    /// Given a standard-matrix row (packed as [u64; 3]), extract the parity portion
    /// (i.e. columns k to N-1) into a [u64; 2] array.  
    /// Assumes that k > 64 (which holds for k in [77, 91]).
    fn extract_parity(row: &[u64; 3], k: usize) -> [u64; 2] {
        let parity_bits = N - k; // total number of parity bits
        let offset = k - 64; // within word1, parity starts at bit `offset`
                             // In word1, we can take at most (64 - offset) bits:
        let bits_from_word1 = if parity_bits < (64 - offset) {
            parity_bits
        } else {
            64 - offset
        };
        let first_part = (row[1] >> offset)
            & if bits_from_word1 == 64 {
                !0
            } else {
                (1u64 << bits_from_word1) - 1
            };
        let remaining_bits = parity_bits.saturating_sub(bits_from_word1);
        let second_part = if remaining_bits > 0 {
            row[2]
                & if remaining_bits == 64 {
                    !0
                } else {
                    (1u64 << remaining_bits) - 1
                }
        } else {
            0
        };
        [first_part, second_part]
    }

    /// Reduces N×k generator storage using coordinate row swaps and message-basis column XORs.
    ///
    /// Returns the pivot count and updates the generator row mapping and basis
    /// transformation; standard storage and its caches remain unchanged.
    pub fn reduce_generator(&mut self) -> usize {
        let mut rank = 0;

        // Iterate over first k rows (message bits)
        for row in 0..self.k {
            if let Some(pivot_row) = (row..N).find(|&search_row_idx| self.bit(search_row_idx, row))
            {
                // If pivot isn't in the current row, swap it in
                if pivot_row != row {
                    self.swap_rows(pivot_row, row);
                }

                // Zero out rest of message bits rows using this pivot
                self.eliminate_columns_vectorized(row);

                rank += 1;
            }
        }

        rank
    }

    pub(crate) fn eliminate_columns_vectorized(&mut self, rank: usize) {
        // Row 'rank' contains a 1 in column 'rank'
        let pivot_row = self.generator[rank];

        // Column XORs leave only the pivot bit set in this row while preserving the code.
        for col in 0..self.k {
            if col != rank {
                let col_chunk = col / 64;
                let col_bit = col % 64;
                if pivot_row[col_chunk] & (1u64 << col_bit) != 0 {
                    for row in 0..N {
                        let new_bit_value = self.bit(row, col) ^ self.bit(row, rank);
                        if new_bit_value {
                            self.set_bit(row, col);
                        } else {
                            self.clear_bit(row, col);
                        }
                    }

                    // Track this operation in our 'E' matrix
                    self.update_transformation_for_elimination(rank, col);
                }
            }
        }
    }

    #[inline(always)]
    /// Updates the transformation matrix to record that the target column has been
    /// replaced by the XOR (i.e. addition over GF(2)) of itself with the pivot column.
    ///
    /// # Arguments
    /// * `pivot_col` - The index of the pivot column (the column that remains unchanged).
    /// * `target_col` - The index of the column that is eliminated via XOR with the pivot.
    pub(crate) fn update_transformation_for_elimination(
        &mut self,
        pivot_col: usize,
        target_col: usize,
    ) {
        // Since transformation is stored in column-major order, each transformation[i]
        // is a [u64; 2] that represents a k-bit vector.
        self.transformation[target_col][0] ^= self.transformation[pivot_col][0];
        self.transformation[target_col][1] ^= self.transformation[pivot_col][1];
    }

    /// Updates the transformation matrix to record that columns `col1` and `col2`
    /// have been swapped.
    pub(crate) fn update_transformation_for_swap(&mut self, col1: usize, col2: usize) {
        self.transformation.swap(col1, col2);
    }

    /// Reorders generator rows using a destination-to-source permutation of `0..174`.
    ///
    /// Replaces the stored row mapping with `perm`; standard storage and caches are unchanged.
    pub fn apply_row_permutation(&mut self, perm: &[usize; 174]) {
        let mut temp = [[0u64; 2]; N];

        for i in 0..N {
            temp[i] = self.generator[perm[i]];
        }

        self.generator = temp;
        self.pi1 = *perm;
    }

    pub(crate) fn apply_standard_column_permutation(&mut self, perm: &[usize; 174]) {
        let mut temp = [[0u64; 3]; 91];

        for row in 0..self.k {
            for old_col in 0..N {
                if self.standard_bit(row, old_col) {
                    let new_col = perm[old_col];
                    let chunk = new_col / 64;
                    let bit = new_col % 64;
                    temp[row][chunk] |= 1u64 << bit;
                }
            }
        }

        self.standard_matrix = temp;
        self.standard_pi1 = *perm;
    }

    pub(crate) fn compute_standard_h(&mut self) {
        let n_minus_k = N - self.k;
        self.standard_h = [[0; 3]; 97];

        // For each parity equation (corresponding to each parity bit)
        for i in 0..n_minus_k {
            // Find which message bits are used in this parity equation
            // by looking at column k+i in G
            let parity_col = self.k + i;

            // Copy the message bit connections from G
            for msg_bit in 0..self.k {
                if self.standard_bit(msg_bit, parity_col) {
                    let chunk = msg_bit / 64;
                    let bit = msg_bit % 64;
                    self.standard_h[i][chunk] |= 1u64 << bit;
                }
            }

            // Add identity part for the parity bit
            let h_col = self.k + i;
            let chunk = h_col / 64;
            let bit = h_col % 64;
            self.standard_h[i][chunk] |= 1u64 << bit;

            // Verify this equation works by checking against G
            let mut sum = false;

            // Check contribution from each column of G
            for col in 0..N {
                let mut column_contribution = false;

                // Get bit from H (our parity equation)
                let h_bit = if col < self.k {
                    // Message part
                    let chunk = col / 64;
                    let bit = col % 64;
                    (self.standard_h[i][chunk] & (1u64 << bit)) != 0
                } else if col == parity_col {
                    // Identity part
                    true
                } else {
                    false
                };

                // If H has this bit set, add contribution from this column of G
                if h_bit {
                    // Add up all 1s in this column of G where H has 1s
                    for row in 0..self.k {
                        if self.standard_bit(row, col) {
                            column_contribution = !column_contribution;
                        }
                    }
                    if column_contribution {
                        sum = !sum;
                    }
                }
            }
            assert!(!sum, "H * G^T should be zero for row {i}");
        }
    }

    /// Refreshes cached `H = [Pᵀ | I]` from systematic standard storage `[I | P]`.
    pub fn compute_standard_parity_checks(&mut self) {
        let n_minus_k = N - self.k;
        self.standard_h = [[0u64; 3]; 97];

        // For G = [I | P], the parity-check matrix is H = [P^T | I].
        for h_row in 0..n_minus_k {
            for msg_col in 0..self.k {
                if self.standard_bit(msg_col, self.k + h_row) {
                    let chunk = msg_col / 64;
                    let bit = msg_col % 64;
                    self.standard_h[h_row][chunk] |= 1u64 << bit;
                }
            }
            // Set the identity part in the parity region.
            let pos = self.k + h_row;
            let chunk = pos / 64;
            let bit = pos % 64;
            self.standard_h[h_row][chunk] |= 1u64 << bit;
        }
    }

    /// Checks a codeword in current standard coordinates against the cached parity-check matrix.
    ///
    /// Requires [`Self::compute_standard_parity_checks`] or [`Self::replace_standard_rows`]
    /// after the last standard-matrix change.
    pub fn is_standard_codeword(&self, codeword: &[bool; N]) -> bool {
        for h_row in &self.standard_h {
            let mut parity = false;
            for col in 0..N {
                let chunk = col / 64;
                let bit = col % 64;
                let h_bit = (h_row[chunk] & (1 << bit)) != 0;
                parity ^= h_bit & codeword[col];
            }
            if parity {
                return false;
            }
        }
        true
    }

    pub(crate) fn generate_codeword(&self, message: &[bool]) -> [bool; N] {
        assert_eq!(message.len(), self.k, "Message length must match k");
        let mut codeword = [false; N];

        // First k bits are message (systematic form)
        codeword[..self.k].copy_from_slice(message);

        // For each parity bit (positions k to N-1)
        for i in self.k..N {
            let mut parity = false;
            // XOR with message bits where column i has 1s
            for (j, &msg_bit) in message.iter().enumerate() {
                if self.bit(i, j) && msg_bit {
                    parity = !parity;
                }
            }
            codeword[i] = parity;
        }

        codeword
    }

    pub(crate) fn generate_standard_codeword(&self, message: &[bool]) -> [bool; N] {
        assert_eq!(message.len(), self.k, "Message length must match k");
        let mut codeword = [false; N];

        // First k bits are message (systematic form)
        codeword[..self.k].copy_from_slice(message);

        // For each parity bit (positions k to N-1)
        for i in self.k..N {
            let mut parity = false;
            // XOR with message bits where column i has 1s
            for (j, &msg_bit) in message.iter().enumerate() {
                if self.standard_bit(j, i) && msg_bit {
                    parity = !parity;
                }
            }
            codeword[i] = parity;
        }

        codeword
    }

    /// Encodes a message into current systematic standard coordinates using the cached transpose.
    ///
    /// Requires standard storage `[I | P]` and a transpose refreshed by
    /// [`Self::compute_transposed_standard`] or [`Self::replace_standard_rows`].
    ///
    /// # Panics
    /// Panics unless `message.len() == self.k()`.
    #[inline(always)]
    pub fn encode_standard(&self, message: &[bool]) -> [bool; N] {
        assert_eq!(message.len(), self.k, "Message length must match k");
        let mut codeword = [false; N];

        // Copy the message into the first k positions.
        codeword[..self.k].copy_from_slice(message);

        // Pack the message bits into two u64 values.
        let mut packed_msg_low = 0u64;
        let mut packed_msg_high = 0u64;
        for (i, &bit) in message.iter().enumerate() {
            if bit {
                if i < 64 {
                    packed_msg_low |= 1u64 << i;
                } else {
                    packed_msg_high |= 1u64 << (i - 64);
                }
            }
        }

        // Compute each parity bit for columns k to N-1.
        for i in self.k..N {
            let col_mask = self.standard_matrix_t[i];
            let count_low = (packed_msg_low & col_mask[0]).count_ones();
            let count_high = (packed_msg_high & col_mask[1]).count_ones();
            let parity = ((count_low + count_high) & 1) != 0;
            codeword[i] = parity;
        }

        codeword
    }

    /// Reads k×N standard storage, with `row < k` and `col < 174`.
    #[inline(always)]
    pub fn standard_bit(&self, row: usize, col: usize) -> bool {
        debug_assert!(row < self.k && col < N);
        let chunk = col / 64;
        let bit = col % 64;
        (self.standard_matrix[row][chunk] & (1u64 << bit)) != 0
    }

    #[inline(always)]
    pub(crate) fn set_standard_bit(&mut self, row: usize, col: usize) {
        debug_assert!(row < self.k && col < N);
        let chunk = col / 64;
        let bit = col % 64;
        self.standard_matrix[row][chunk] |= 1u64 << bit;
    }

    #[inline(always)]
    pub(crate) fn clear_standard_bit(&mut self, row: usize, col: usize) {
        debug_assert!(row < self.k && col < N);
        let chunk = col / 64;
        let bit = col % 64;
        self.standard_matrix[row][chunk] &= !(1u64 << bit);
    }

    #[inline(always)]
    pub(crate) fn flip_standard_bit(&mut self, row: usize, col: usize) {
        debug_assert!(row < self.k && col < N);
        let chunk = col / 64;
        let bit = col % 64;
        self.standard_matrix[row][chunk] ^= 1u64 << bit;
    }

    #[inline(always)]
    pub(crate) fn swap_standard_rows(&mut self, row1: usize, row2: usize) {
        debug_assert!(row1 < self.k && row2 < self.k);
        self.standard_matrix.swap(row1, row2);
    }

    /// XORs row1 into row2
    #[inline(always)]
    pub(crate) fn add_standard_rows(&mut self, row1: usize, row2: usize) {
        debug_assert!(row1 < self.k && row2 < self.k);
        self.standard_matrix[row2][0] ^= self.standard_matrix[row1][0];
        self.standard_matrix[row2][1] ^= self.standard_matrix[row1][1];
        self.standard_matrix[row2][2] ^= self.standard_matrix[row1][2];
    }

    /// Transposes generator storage into k×N standard storage without refreshing caches.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_standard_form(&mut self) {
        // Clear standard matrix
        self.standard_matrix = [[0u64; 3]; 91];

        // For each row i of output (was column i of input)
        for i in 0..self.k {
            // For each column j of output (was row j of input)
            for j in 0..N {
                if self.bit(j, i) {
                    self.set_standard_bit(i, j);
                }
            }
        }
    }

    /// Converts from standard k×N form back to our N×k form
    #[allow(clippy::wrong_self_convention)]
    fn from_standard_form(&mut self) {
        // Clear existing generator
        self.generator = [[0u64; 2]; N];

        // For each row i of input (becomes column i of output)
        for i in 0..self.k {
            // For each column j of input (becomes row j of output)
            for j in 0..N {
                if self.standard_bit(i, j) {
                    self.set_bit(j, i);
                }
            }
        }
    }

    /// Transforms a k×N generator matrix in standard form into systematic form [I|P] using Gaussian elimination.
    /// Returns the rank and a permutation vector π₂ that tracks the column swaps needed during GE.
    ///
    /// # Operations in k×N standard form:
    /// - Row swaps: ✓ (preserves rowspace directly)
    /// - Row additions: ✓ (preserves rowspace directly)
    /// - Column swaps: ✓ (changes code coordinates but tracked by π₂)
    /// - Column additions: ✘ (would change the code itself)
    ///
    /// # Algorithm
    /// The approach avoids physical column swaps during GE by:
    /// 1. Scanning columns left-to-right (most to least reliable)
    /// 2. For each desired pivot position i:
    ///    - If column i has a 1 in correct row, use it as pivot
    ///    - If it has a 1 in wrong row, do row swap to fix
    ///    - If no 1s found, mark column as "skipped" and try next
    ///    - Once pivot found, use row operations to zero other 1s
    /// 3. Track in indices2/π₂:
    ///    - positions 1..k: which columns were used for pivots
    ///    - positions k+1..N: skipped columns in reliability order
    /// 4. At end, apply single permutation to place columns in systematic order
    ///
    /// Since columns are processed in reliability order, skipped columns automatically
    /// maintain their relative reliability ordering in the final permutation.
    ///
    /// # Returns
    /// - rank: Number of linearly independent columns found (should be k for valid code)
    /// - π₂: Length-N permutation vector mapping destination positions to source columns
    ///
    /// # Note
    /// Assumes matrix is already in k×N standard form. Must call to_standard_form() first
    /// and from_standard_form() after if working with N×k generator matrix.
    pub(crate) fn transform_to_systematic_form_standard(&mut self) -> (usize, [usize; N]) {
        let mut rank = 0;
        let mut col_idx = 0; // Current column being processed
        let mut pi2 = [0; N]; // Will track column permutations
        let mut nskipped = 0;

        // For each row where we want a pivot
        'pivot_search: for id in 0..self.k {
            loop {
                // First check if we already have a pivot
                if self.standard_bit(id, col_idx) {
                    // Found pivot in the right position
                    pi2[id] = col_idx;

                    // Zero out rest of column using row operations
                    self.eliminate_other_ones_in_column(id, col_idx);

                    rank += 1;
                    col_idx += 1;
                    continue 'pivot_search;
                }

                // No pivot in current position, look for one below
                if let Some(pivot_row) =
                    ((id + 1)..self.k).find(|&row| self.standard_bit(row, col_idx))
                {
                    // Found pivot, but in wrong row. Swap rows
                    self.swap_standard_rows(id, pivot_row);

                    pi2[id] = col_idx;

                    // Zero out rest of column using row operations
                    self.eliminate_other_ones_in_column(id, col_idx);

                    rank += 1;
                    col_idx += 1;
                    continue 'pivot_search;
                }

                // No pivot found in this column at all
                pi2[self.k + nskipped] = col_idx; // Record skipped column
                nskipped += 1;

                col_idx += 1;
                if col_idx >= N {
                    // Ran out of columns - matrix not full rank
                    break 'pivot_search;
                }
            }
        }

        // Fill in remaining indices for any unused columns
        for i in (self.k + nskipped)..N {
            pi2[i] = i;
        }

        // Apply the column permutation we've built up
        self.apply_column_permutation(&pi2);

        (rank, pi2)
    }
    // Rebuild the transpose after constructing or changing the standard matrix.
    pub(crate) fn compute_transposed_standard_simple(&mut self) {
        let mut transposed = [[0u64; 2]; N];
        for row in 0..self.k {
            // Get the row as 3 u64 values (only N bits are used).
            let row_bits = self.standard_matrix[row];
            for i in 0..N {
                let chunk = i / 64;
                let bit = i % 64;
                // If the (row, i) bit is set, then set bit 'row' in column i.
                if (row_bits[chunk] & (1u64 << bit)) != 0 {
                    if row < 64 {
                        transposed[i][0] |= 1u64 << row;
                    } else {
                        transposed[i][1] |= 1u64 << (row - 64);
                    }
                }
            }
        }
        self.standard_matrix_t = transposed;
    }

    /// Refreshes the standard transpose cache used by [`Self::encode_standard`].
    #[inline(always)]
    pub fn compute_transposed_standard(&mut self) {
        let mut transposed = [[0u64; 2]; N];

        for row in 0..self.k {
            // Get the row as an array of 3 u64 values. Only the lowest N bits are valid.
            let row_bits = self.standard_matrix[row];
            for (chunk_idx, &word) in row_bits.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let tz = bits.trailing_zeros() as usize;
                    let col = chunk_idx * 64 + tz;
                    if col < N {
                        if row < 64 {
                            transposed[col][0] |= 1u64 << row;
                        } else {
                            transposed[col][1] |= 1u64 << (row - 64);
                        }
                    }
                    bits &= bits - 1;
                }
            }
        }
        self.standard_matrix_t = transposed;
    }

    /// Reduces standard storage with bitwise row XORs after applying reliability permutation `pi1`.
    ///
    /// Requires a full-rank k×N matrix and `pi1` mapping destinations to source
    /// columns. Returns the rank and cumulative destination-to-source mapping,
    /// with pivot columns first; derived caches are not refreshed.
    pub fn reduce_standard_bitwise(&mut self, pi1: &[usize; N]) -> (usize, [usize; N]) {
        self.reduce_standard_with_packed_xor(pi1, false)
    }

    /// Uses packed row XORs with the same preconditions, mapping, and cache behavior as [`Self::reduce_standard_bitwise`].
    pub fn reduce_standard_packed(&mut self, pi1: &[usize; N]) -> (usize, [usize; N]) {
        self.reduce_standard_with_packed_xor(pi1, true)
    }

    fn reduce_standard_with_packed_xor(
        &mut self,
        pi1: &[usize; N],
        packed_xor: bool,
    ) -> (usize, [usize; N]) {
        // Apply the reliability permutation
        self.apply_column_permutation(pi1);

        // Record pivot columns first and skipped columns afterward.
        let mut pi2: [usize; N] = [0; N];
        let mut col_ptr = 0; // points to the current column being examined
        let mut nskipped = 0; // counts columns that did not serve as pivots
        let mut rank = 0;

        // Process each row of the k×N matrix.
        for row in 0..self.k {
            let mut pivot_found = false;
            // Search for a pivot in the current row, starting at col_ptr.
            while col_ptr < N && !pivot_found {
                if self.standard_bit(row, col_ptr) {
                    // Found a pivot in (row, col_ptr)
                    pivot_found = true;
                } else {
                    // Try swapping with a lower row that has a 1 in this column.
                    for r in (row + 1)..self.k {
                        if self.standard_bit(r, col_ptr) {
                            self.swap_standard_rows(row, r);
                            pivot_found = true;
                            break;
                        }
                    }
                }
                if !pivot_found {
                    // This column cannot serve as pivot for the current row.
                    // Record it in the "skipped" region (positions k..)
                    pi2[self.k + nskipped] = col_ptr;
                    col_ptr += 1;
                    nskipped += 1;
                }
            }
            if col_ptr >= N {
                // Ran out of columns before finding a pivot.
                break;
            }
            // Record the pivot column for this row.
            pi2[row] = col_ptr;
            rank += 1;

            // Eliminate the pivot column in all other rows.
            for r in 0..self.k {
                if r != row && self.standard_bit(r, col_ptr) {
                    if packed_xor {
                        self.add_standard_rows(row, r);
                    } else {
                        for c in 0..N {
                            if self.standard_bit(row, c) {
                                self.flip_standard_bit(r, c);
                            }
                        }
                    }
                }
            }
            col_ptr += 1;
        }
        // Append columns not visited by the pivot scan.
        for i in (self.k + nskipped)..N {
            pi2[i] = i;
        }

        // Compose destination-to-source mappings to recover natural coordinates.
        let mut cum_perm: [usize; N] = [0; N];
        for new_idx in 0..N {
            cum_perm[new_idx] = pi1[pi2[new_idx]];
        }

        // Move the pivot columns into the leading block.
        self.apply_column_permutation(&pi2);

        (rank, cum_perm)
    }

    /// Eliminates all ones in the given column (except for the pivot)
    /// Uses row operations (XOR) which are valid in standard form
    #[inline]
    fn eliminate_other_ones_in_column(&mut self, pivot_row: usize, col: usize) {
        for row in 0..self.k {
            if row != pivot_row && self.standard_bit(row, col) {
                self.add_standard_rows(pivot_row, row);
            }
        }
    }

    /// Applies a column permutation to the standard form matrix
    fn apply_column_permutation(&mut self, perm: &[usize]) {
        let mut new_matrix = [[0u64; 3]; 91];

        // For each column in the output
        for (new_col, &old_col) in perm.iter().enumerate() {
            // Copy this column to its new position
            for row in 0..self.k {
                if self.standard_bit(row, old_col) {
                    let chunk = new_col / 64;
                    let bit = new_col % 64;
                    new_matrix[row][chunk] |= 1u64 << bit;
                }
            }
        }

        self.standard_matrix = new_matrix;
    }

    /// Copies an N×k matrix into generator storage, leaving standard storage and caches zeroed.
    ///
    /// # Panics
    /// Panics unless `k` is in `77..=91` and the input has at least 174 rows of at least `k` bits.
    #[allow(clippy::ptr_arg)]
    pub fn from_row_major(matrix: &Vec<Vec<bool>>, k: usize) -> Self {
        let mut bit_matrix = Self::new(k);

        // Input matrix is N×k in row-major format
        for row in 0..N {
            for col in 0..k {
                if matrix[row][col] {
                    bit_matrix.set_bit(row, col);
                }
            }
        }

        bit_matrix
    }

    pub(crate) fn to_row_major(&self) -> Vec<Vec<bool>> {
        (0..self.k)
            .map(|col| {
                let mut row_vec = Vec::with_capacity(N);
                for row in 0..N {
                    row_vec.push(self.bit(row, col));
                }
                row_vec
            })
            .collect()
    }

    /// Checks parity directly from systematic standard storage in its current coordinates.
    ///
    /// Ignores cached parity checks and prints the first mismatched parity equation.
    #[inline(never)]
    pub fn is_valid_standard_codeword(&self, codeword: &[bool; N]) -> bool {
        // For each parity equation
        for i in 0..(N - self.k) {
            let mut sum = false;

            // Sum the message bits that participate in this parity equation
            for j in 0..self.k {
                let msg_bit = codeword[j];
                if self.standard_bit(j, self.k + i) && msg_bit {
                    // Check column k+i
                    sum = !sum;
                }
            }

            // Compare sum with the corresponding parity bit
            if sum != codeword[self.k + i] {
                println!("Mismatched parity for equation {i}");
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::BASE_GENERATOR_MATRIX;
    use rand::Rng;
    use std::cmp::Ordering;

    // fn generate_random_permutation() -> [usize; N] {
    //     let mut perm: Vec<usize> = (0..N).collect();
    //     let mut rng = rand::rng();
    //     perm.shuffle(&mut rng);
    //     let mut arr = [0usize; N];
    //     arr.copy_from_slice(&perm);
    //     arr
    // }

    /// Helper: generates a random message (Vec<bool>) of length k.
    fn generate_random_message(k: usize) -> Vec<bool> {
        let mut rng = rand::rng();
        (0..k).map(|_| rng.random_bool(0.5)).collect()
    }

    #[test]
    fn test_empty_matrix() {
        // Create a k×n matrix with all zeros (k rows, 174 columns)
        let k = 77; // Minimum valid k
        let input = vec![vec![false; k]; N]; // k rows, n=174 columns
        let mut matrix = BitMatrix::from_row_major(&input, k);

        // An all-zero matrix should have rank 0
        assert_eq!(matrix.reduce_generator(), 0);
    }

    #[test]
    fn test_single_element() {
        // Create a k×n matrix with a single 1 in first row, first column
        let k = 77; // Minimum valid k
        let mut input = vec![vec![false; k]; N]; // k rows, n=174 columns
        input[0][0] = true; // Set a single 1 at (row 0, col 0)

        let mut matrix = BitMatrix::from_row_major(&input, k);

        // A matrix with a single 1 should have rank 1
        assert_eq!(matrix.reduce_generator(), 1);

        // Verify the systematic form: first column should be [1, 0, 0, ..., 0]
        // Remember: bit(row, col) in our column-major storage
        assert!(matrix.bit(0, 0), "Expected 1 at (0,0)");
        for row in 1..k {
            assert!(!matrix.bit(row, 0), "Expected 0 at ({},0)", row);
        }
    }

    #[test]
    fn test_k_bounds() {
        // Test minimum k
        let k_min = 77;
        let input = vec![vec![false; k_min]; N];
        let _matrix = BitMatrix::from_bool_matrix(&input, k_min);

        // Test maximum k
        let k_max = 91;
        let input = vec![vec![false; k_max]; N];
        let _matrix = BitMatrix::from_bool_matrix(&input, k_max);

        // Test out of bounds k should panic
        assert!(
            std::panic::catch_unwind(|| {
                let k_invalid = 92;
                let input = vec![vec![false; k_invalid]; N];
                BitMatrix::from_bool_matrix(&input, k_invalid);
            })
            .is_err(),
            "Expected panic for k > 91"
        );

        assert!(
            std::panic::catch_unwind(|| {
                let k_invalid = 76;
                let input = vec![vec![false; k_invalid]; N];
                BitMatrix::from_bool_matrix(&input, k_invalid);
            })
            .is_err(),
            "Expected panic for k < 77"
        );
    }

    #[test]
    fn test_zero_matrix() {
        // Test a k×n zero matrix at minimum valid k
        let k = 77;
        let input = vec![vec![false; k]; N]; // 174 rows, k columns
        let mut matrix = BitMatrix::from_row_major(&input, k);
        // A zero matrix has rank 0 regardless of storage format
        assert_eq!(matrix.reduce_generator(), 0);
    }

    #[test]
    fn test_already_systematic() {
        let k = 77; // Minimum valid k
        let mut input = vec![vec![false; k]; N];

        // Already in systematic form
        for i in 0..k {
            input[i][i] = true;
        }

        let mut matrix = BitMatrix::from_row_major(&input, k);
        let rank = matrix.reduce_generator();

        assert_eq!(rank, k);
        // Verify still systematic
        for i in 0..k {
            for j in 0..k {
                assert_eq!(matrix.bit(i, j), i == j);
            }
        }
    }

    #[test]
    fn test_single_swap_needed() {
        let k = 77;
        let mut input = vec![vec![false; k]; N];

        // Like identity but rows 0 and 1 are swapped
        input[1][0] = true;
        input[0][1] = true;
        for i in 2..k {
            input[i][i] = true;
        }

        let mut matrix = BitMatrix::from_row_major(&input, k);
        let rank = matrix.reduce_generator();

        assert_eq!(rank, k);
        for i in 0..k {
            for j in 0..k {
                assert_eq!(matrix.bit(i, j), i == j);
            }
        }
    }

    #[test]
    fn test_elimination_needed() {
        let k = 77;
        let mut input = vec![vec![false; k]; N];

        // Set up matrix needing elimination:
        // 1 1 0 0 ...
        // 0 1 0 0 ...
        // 0 0 1 0 ...
        // etc.
        for i in 0..k {
            input[i][i] = true;
        }
        input[0][1] = true; // Extra 1 that needs eliminating

        let mut matrix = BitMatrix::from_row_major(&input, k);
        let rank = matrix.reduce_generator();

        assert_eq!(rank, k);
        for i in 0..k {
            for j in 0..k {
                assert_eq!(matrix.bit(i, j), i == j);
            }
        }
    }

    #[test]
    fn test_full_rank_from_lower_triangular() {
        let k = 77;
        let mut input = vec![vec![false; k]; N];

        // Lower triangular - requires upward elimination
        for i in 0..k {
            for j in 0..=i {
                input[i][j] = true;
            }
        }

        let mut matrix = BitMatrix::from_row_major(&input, k);
        let rank = matrix.reduce_generator();

        assert_eq!(rank, k);
        for i in 0..k {
            for j in 0..k {
                assert_eq!(matrix.bit(i, j), i == j);
            }
        }
    }

    #[test]
    fn test_randomly_permuted_generator() {
        let k = 91;
        let generator = CascadedGeneratorMatrix::new(0).unwrap();
        let mut matrix = BitMatrix::from_cascaded_generator(&generator);

        // Generate a random codeword in the original basis
        let message: [bool; 91] = std::array::from_fn(|i| i % 2 == 0);
        let codeword_cascaded_original = matrix.generate_codeword(&message);
        assert_eq!(
            codeword_cascaded_original.len(),
            N,
            "Codeword length must be 174"
        );

        // Convert generator to standard k x N form, and store in `standard_matrix` field
        matrix.to_standard_form();

        // Create a random permutation of the N = 174 bits
        let rand_permutation: [usize; N] = generate_random_permutation();

        // Permute the `standard_matrix` columns according to permutation
        matrix.apply_standard_column_permutation(&rand_permutation);

        // Use Gaussian Elimination to transform the permuted matrix back into systematic form
        let (rank, pi2) = matrix.transform_to_systematic_form_standard();
        matrix.standard_pi2 = pi2;

        // Check standard generator contains kxk identity
        assert_eq!(rank, k);
        for i in 0..k {
            for j in 0..k {
                assert_eq!(matrix.standard_bit(i, j), i == j);
            }
        }

        // Copy the transposed standard form matrix back into `generator`
        matrix.from_standard_form();

        // Check generator contains kxk identity
        assert_eq!(rank, k);
        for i in 0..k {
            for j in 0..k {
                assert_eq!(matrix.bit(i, j), i == j);
            }
        }

        // Generate a codeword in the *new* basis from the same message as before
        let codeword_cascaded_new = matrix.generate_codeword(&message);
        assert_eq!(
            codeword_cascaded_new.len(),
            N,
            "Codeword length must be 174"
        );

        // Permute the codeword with pi1 (the reliability permutation)
        let mut permuted_codeword_pi1 = [false; N];

        for i in 0..N {
            permuted_codeword_pi1[i] = codeword_cascaded_original[rand_permutation[i]];
        }

        // Permute it again with pi2, which tracks any row swaps needed to achieve systematic
        // form during Gaussian Elimination
        let mut permuted_codeword_pi2: [bool; N] = [false; N];
        for i in 0..N {
            permuted_codeword_pi2[i] = permuted_codeword_pi1[matrix.standard_pi2[i]];
        }

        assert!(
            matrix.is_standard_codeword(&permuted_codeword_pi2),
            "Permuted codeword pi2 must be valid codeword"
        );

        // Extract the message bits from the permuted codeword, so we can re-encode it
        let mut permuted_message_bits = [false; 91];
        permuted_message_bits.copy_from_slice(&permuted_codeword_pi2[..91]);

        // Re-encode it with the transformed generator (which should be in systematic form)
        let recomputed_permuted_codeword: [bool; 174] =
            matrix.generate_codeword(&permuted_message_bits);

        assert!(
            matrix.is_standard_codeword(&recomputed_permuted_codeword),
            "Recomputed permuted codeword must be valid codeword"
        );

        println!("Permuted generated codeword: {:?}", permuted_codeword_pi2);
        println!("Recomputed codeword: {:?}", recomputed_permuted_codeword);
    }

    #[test]
    fn test_sparse_random_matrix() {
        let k = 85;
        let mut input = vec![vec![false; k]; N];

        // Place 1s randomly but sparsely to ensure full rank
        // We'll put exactly one 1 in each column, plus some extras
        for col in 0..k {
            input[col][col] = true; // Ensures full rank
                                    // Add a few random 1s in this column
            for _ in 0..3 {
                let random_row = (col + 7) % N; // Deterministic "random" for test
                input[random_row][col] = true;
            }
        }

        let mut matrix = BitMatrix::from_row_major(&input, k);
        let rank = matrix.reduce_generator();

        assert_eq!(rank, k);
        for i in 0..k {
            for j in 0..k {
                assert_eq!(matrix.bit(i, j), i == j);
            }
        }
    }

    #[test]
    fn test_identity_portion() {
        // Test with minimum valid k
        let k = 77;
        let mut input = vec![vec![false; k]; N];
        // Set identity portion in first k columns
        for i in 0..k {
            input[i][i] = true;
        }
        let mut matrix = BitMatrix::from_row_major(&input, k);
        // Matrix with identity in first k columns should have rank k
        assert_eq!(matrix.reduce_generator(), k);

        // Verify the identity portion is preserved
        for i in 0..k {
            for j in 0..k {
                assert_eq!(
                    matrix.bit(i, j),
                    i == j,
                    "Identity check failed at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_basic_row_operations() {
        // Test with minimum valid k=77
        // Create a k×n matrix that will need column swaps
        // We'll set up an interesting pattern in the first 3×3 corner
        let k = 77;
        let mut input = vec![vec![false; k]; N];

        // Set up a 3×3 pattern that needs a row swap
        let pattern = [
            [false, true, false], // [0 1 0]
            [true, false, false], // [1 0 0]
            [false, false, true], // [0 0 1]
        ];

        // Copy pattern into top-left corner
        for i in 0..3 {
            for j in 0..3 {
                input[i][j] = pattern[j][i];
            }
        }

        let mut matrix = BitMatrix::from_row_major(&input, k);

        // Matrix should achieve rank at least 3 due to our pattern
        let rank = matrix.reduce_generator();
        assert!(rank >= 3, "Expected rank >= 3, got {}", rank);

        // Verify the systematic form in the 3×3 corner
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    matrix.bit(i, j),
                    i == j,
                    "Systematic form check failed at ({},{})",
                    i,
                    j
                );
            }
        }

        // The first 3 elements of column_permutation should show the swap
        assert_eq!(
            &matrix.pi1[0..3],
            &[1, 0, 2],
            "Expected column swaps [1,0,2], got {:?}",
            &matrix.pi1[0..3]
        );
    }

    #[test]
    fn test_linearly_dependent_rows() {
        // Create a k×n matrix with a linearly dependent pattern in first 3 rows
        let k = 77; // Minimum valid k
        let mut input = vec![vec![false; k]; N];

        // Set up pattern in first 3 rows, first 3 columns
        // Third row is XOR of first two rows
        let pattern = [
            [true, false, true], // Row 1
            [false, true, true], // Row 2
            [true, true, false], // Row 3 = Row 1 XOR Row 2
        ];

        // Copy pattern into top-left corner
        for i in 0..3 {
            for j in 0..3 {
                input[i][j] = pattern[j][i];
            }
        }

        let mut matrix = BitMatrix::from_row_major(&input, k);

        // Linear dependence should reduce rank
        let rank = matrix.reduce_generator();
        assert!(rank < k, "Expected rank < k due to linear dependency");

        // Verify systematic form in the independent part (first two rows)
        assert!(matrix.bit(0, 0), "Expected 1 at (0,0)");
        assert!(!matrix.bit(0, 1), "Expected 0 at (0,1)");
        assert!(!matrix.bit(1, 0), "Expected 0 at (1,0)");
        assert!(matrix.bit(1, 1), "Expected 1 at (1,1)");
    }

    #[test]
    fn test_full_rank_matrix() {
        // Create k×n matrix with upper triangular pattern in top-left corner
        let k = 77; // Minimum valid k
        let mut input = vec![vec![false; k]; N];

        // Set up upper triangular pattern in first 3×3 corner
        let pattern = [
            [true, true, false],  // [1 1 0]
            [false, true, true],  // [0 1 1]
            [false, false, true], // [0 0 1]
        ];

        // Copy pattern into top-left corner
        for i in 0..3 {
            for j in 0..3 {
                input[i][j] = pattern[j][i];
            }
        }

        let mut matrix = BitMatrix::from_row_major(&input, k);

        let rank = matrix.reduce_generator();
        assert!(rank >= 3, "Expected rank >= 3 from triangular pattern");

        // Verify systematic form in the 3×3 corner
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    matrix.bit(i, j),
                    i == j,
                    "Systematic form check failed at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_maximum_dimensions() {
        // Test with maximum valid k
        let k = 91;
        let mut input = vec![vec![false; k]; N];

        // Create a full rank matrix by setting identity pattern
        for i in 0..k {
            input[i][i] = true; // Identity pattern in first k columns
        }

        let mut matrix = BitMatrix::from_row_major(&input, k);
        let rank = matrix.reduce_generator();

        assert_eq!(rank, k, "Should achieve full rank k={}", k);

        // Verify systematic form in first k columns
        for i in 0..k {
            for j in 0..k {
                assert_eq!(
                    matrix.bit(i, j),
                    i == j,
                    "Failed systematic form check at ({},{})",
                    i,
                    j
                );
            }
        }

        // Column permutation should be length n=174, not k
        assert_eq!(
            matrix.pi1.len(),
            N,
            "Column permutation should track all columns"
        );
    }
    #[test]
    fn test_edge_cases() {
        // Test with minimum k=77
        let k = 77;
        let mut input = vec![vec![false; k]; N];

        // Make first k columns independent
        for i in 0..k {
            input[i][i] = true;
        }
        let mut matrix = BitMatrix::from_row_major(&input, k);
        let rank = matrix.reduce_generator();
        assert_eq!(rank, k, "Matrix with k=77 should achieve rank k");

        // Verify systematic form in first k columns
        for i in 0..k {
            for j in 0..k {
                assert_eq!(
                    matrix.bit(i, j),
                    i == j,
                    "k=77 matrix failed systematic check at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Test with maximum k=91
        let k = 91;
        let mut input = vec![vec![false; k]; N];
        for i in 0..k {
            input[i][i] = true;
        }
        let mut matrix = BitMatrix::from_row_major(&input, k);
        let rank = matrix.reduce_generator();
        assert_eq!(rank, k, "Matrix with k=91 should achieve rank k");

        // Verify systematic form in all k columns
        for i in 0..k {
            for j in 0..k {
                assert_eq!(
                    matrix.bit(i, j),
                    i == j,
                    "k=91 matrix failed systematic check at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_permutation_tracking() {
        let k = 77; // Minimum valid k for FT8

        // Create a N x k matrix with a known pattern that requires column swaps
        let mut input = vec![vec![false; k]; N];

        // Set up a 3×3 pattern in the top-left corner that requires column swaps
        let pattern = [
            [false, true, false], // [0 1 0]
            [true, false, false], // [1 0 0]
            [false, false, true], // [0 0 1]
        ];

        // Copy the pattern into the top-left corner of the matrix
        for i in 0..3 {
            for j in 0..3 {
                input[i][j] = pattern[j][i];
            }
        }

        // Set the identity portion in the first k columns
        for i in 3..k {
            input[i][i] = true;
        }

        // Create the BitMatrix from the input matrix
        let mut bit_matrix = BitMatrix::from_row_major(&input, k);

        // Get the initial column permutation (should be identity)
        let initial_permutation = bit_matrix.pi1;
        for i in 0..N {
            assert_eq!(
                initial_permutation[i], i,
                "Initial permutation should be identity"
            );
        }

        // Perform Gaussian elimination to convert the matrix to systematic form
        let rank = bit_matrix.reduce_generator();
        assert_eq!(rank, k, "Matrix should achieve full rank");

        // Verify that the column permutation reflects the swaps
        let final_permutation = bit_matrix.pi1;

        // Check the first 3 columns to ensure the expected swaps occurred
        assert_eq!(
            &final_permutation[0..3],
            &[1, 0, 2],
            "First 3 columns should show [1, 0, 2] swap pattern"
        );

        // Verify that the permutation is correctly applied to the matrix
        for i in 0..k {
            for j in 0..k {
                assert_eq!(
                    bit_matrix.bit(i, j),
                    i == j,
                    "Matrix not in systematic form at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Verify that the inverse permutation correctly undoes the permutation
        let mut inverse_permutation = [0; N];
        for i in 0..N {
            inverse_permutation[final_permutation[i]] = i;
        }

        // Apply the inverse permutation to the final permutation
        for i in 0..N {
            assert_eq!(
                inverse_permutation[final_permutation[i]], i,
                "Inverse permutation mismatch at position {}",
                i
            );
        }

        // Verify that applying the permutation and its inverse restores the original order
        let mut restored_permutation = [0; N];
        for i in 0..N {
            restored_permutation[i] = final_permutation[inverse_permutation[i]];
        }
        for i in 0..N {
            assert_eq!(
                restored_permutation[i], i,
                "Permutation and inverse did not restore original order at position {}",
                i
            );
        }
    }

    #[test]
    fn test_no_swaps_needed() {
        // Create k×n matrix already in systematic form
        let k = 77;
        let mut input = vec![vec![false; k]; N];

        // Set identity pattern in first k columns
        for i in 0..k {
            input[i][i] = true;
        }

        let mut matrix = BitMatrix::from_row_major(&input, k);
        let initial_perm = matrix.pi1;

        let rank = matrix.reduce_generator();
        assert_eq!(rank, k);

        // Permutation should be unchanged
        assert_eq!(
            matrix.pi1, initial_perm,
            "Column permutation should not change for matrix already in systematic form"
        );

        // Verify systematic form maintained
        for i in 0..k {
            for j in 0..k {
                assert_eq!(
                    matrix.bit(i, j),
                    i == j,
                    "Matrix not in systematic form at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_multiple_swaps() {
        // Create k×n matrix requiring multiple column swaps
        let k = 77;
        let mut input = vec![vec![false; k]; N];

        // Set up pattern in first 3×3 corner that needs multiple swaps
        let pattern = [
            [false, false, true], // [0 0 1]  Col 2 needs to move to 0
            [false, true, false], // [0 1 0]  Col 1 stays in 1
            [true, false, false], // [1 0 0]  Col 0 needs to move to 2
        ];

        // Copy pattern into top-left corner
        for i in 0..3 {
            for j in 0..3 {
                input[i][j] = pattern[i][j];
            }
        }

        let mut matrix = BitMatrix::from_row_major(&input, k);

        let rank = matrix.reduce_generator();
        assert!(rank >= 3, "Should achieve at least rank 3");

        // Verify expected swaps in first 3 columns
        assert_eq!(
            &matrix.pi1[0..3],
            &[2, 1, 0],
            "First 3 columns should show [2,1,0] swap pattern"
        );

        // Verify systematic form in 3×3 corner
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    matrix.bit(i, j),
                    i == j,
                    "Matrix not in systematic form at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_permutation_with_max_dimensions() {
        let k = 91; // Maximum valid k

        let mut input = vec![vec![false; k]; N];

        // Place 1s in pattern requiring swaps but within look-ahead distance
        for i in 0..k {
            // Place each 1 at most 10 columns away from target
            let col = (i + 5) % k;
            input[i][col] = true;
        }

        let mut matrix = BitMatrix::from_row_major(&input, k);
        let rank = matrix.reduce_generator();

        assert_eq!(rank, k, "Should achieve full rank k={}", k);

        // Verify systematic form
        for i in 0..k {
            for j in 0..k {
                assert_eq!(
                    matrix.bit(i, j),
                    i == j,
                    "Failed systematic form check at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_generator_matrix_properties() {
        // Create 174x91 generator matrix
        let k = 91;
        let cascaded_gen = CascadedGeneratorMatrix::new(0).unwrap();
        let matrix = BitMatrix::from_cascaded_generator(&cascaded_gen);

        // Check systematic form [I | P]
        // First k×k should be identity
        for i in 0..k {
            for j in 0..k {
                assert_eq!(
                    matrix.bit(j, i),
                    i == j,
                    "Identity portion incorrect at ({},{})",
                    i,
                    j
                );
            }
        }

        // Count 1s in each row of parity portion
        let mut row_weights = vec![0; N - k];
        for col in 0..k {
            for row in k..174 {
                if matrix.bit(row, col) {
                    row_weights[row - k] += 1;
                }
            }
        }

        // Print statistics
        println!("Parity portion statistics:");
        println!("Min row weight: {}", row_weights.iter().min().unwrap());
        println!("Max row weight: {}", row_weights.iter().max().unwrap());
        println!(
            "Avg row weight: {:.2}",
            row_weights.iter().sum::<usize>() as f64 / k as f64
        );
    }

    #[test]
    fn test_codeword_generation_and_validation() {
        use rand::Rng;
        let num_cascaded_bits = 0;
        let k = 91 - num_cascaded_bits;
        let cascaded_gen = CascadedGeneratorMatrix::new(num_cascaded_bits).unwrap();
        let matrix = BitMatrix::from_cascaded_generator(&cascaded_gen);

        let mut rng = rand::rng();
        for _ in 0..10 {
            // Run the test multiple times with different random messages
            let message: Vec<bool> = (0..k).map(|_| rng.random()).collect();

            // Generate codeword using BitMatrix
            let codeword = matrix.generate_codeword(&message);

            // Generate codeword using BaseGeneratorMatrix
            let mut message_array = [false; 91];
            message_array[..k].copy_from_slice(&message);
            let codeword_base = BASE_GENERATOR_MATRIX.encode(&message_array);

            // Compare codewords
            assert_eq!(
                codeword, codeword_base,
                "Codewords generated by BitMatrix and BaseGeneratorMatrix do not match"
            );

            // Convert to fixed-length array for validation
            let mut cw_array = [false; N];
            cw_array.copy_from_slice(&codeword);

            assert!(
                BASE_GENERATOR_MATRIX.is_valid_codeword(&cw_array),
                "BaseGeneratorMatrix codeword failed validation"
            );

            // Verify systematic property
            assert_eq!(
                &codeword[..k],
                &message[..],
                "Message bits not preserved in codeword"
            );
        }
    }

    #[test]
    fn test_systematic_form_codeword_validation() {
        let k = 91;
        let generator = CascadedGeneratorMatrix::new(0).unwrap();
        let mut matrix = BitMatrix::from_cascaded_generator(&generator);
        let rand_permutation: [usize; N] = generate_random_permutation();

        // Convert to standard form and transform to systematic
        matrix.to_standard_form();
        matrix.apply_standard_column_permutation(&rand_permutation);
        let (rank, _) = matrix.transform_to_systematic_form_standard();
        assert_eq!(rank, k, "Matrix should have full rank");

        // Compute parity check matrix
        matrix.compute_standard_parity_checks();

        // Generate test messages
        let test_cases = vec![
            // All zeros
            vec![false; k],
            // All ones
            vec![true; k],
            // Alternating bits
            (0..k).map(|i| i % 2 == 0).collect(),
            // Single one at different positions
            {
                let mut msg = vec![false; k];
                msg[0] = true;
                msg
            },
            {
                let mut msg = vec![false; k];
                msg[k / 2] = true;
                msg
            },
            {
                let mut msg = vec![false; k];
                msg[k - 1] = true;
                msg
            },
        ];

        for msg in test_cases {
            let codeword = matrix.generate_standard_codeword(&msg);

            // Test 1: Verify codeword length
            assert_eq!(codeword.len(), N, "Codeword should have length N=174");

            // Test 2: Verify systematic property (first k bits match message)
            for i in 0..k {
                assert_eq!(codeword[i], msg[i], "First k bits should match message");
            }

            // Test 3: Verify codeword is valid using parity check matrix
            assert!(
                matrix.is_standard_codeword(&codeword),
                "Generated codeword should satisfy parity check equations"
            );

            // Test 4: Verify codeword is valid using direct parity computation
            assert!(
                matrix.is_valid_standard_codeword(&codeword),
                "Generated codeword should satisfy direct parity checks"
            );
        }
    }

    #[test]
    fn test_codeword_linearity() {
        let k = 91;
        let generator = CascadedGeneratorMatrix::new(0).unwrap();
        let mut matrix = BitMatrix::from_cascaded_generator(&generator);
        let rand_permutation: [usize; N] = generate_random_permutation();

        // Convert to standard form and transform
        matrix.to_standard_form();
        matrix.apply_standard_column_permutation(&rand_permutation);
        matrix.transform_to_systematic_form_standard();
        matrix.compute_standard_h();

        // Generate two random messages
        let msg1: Vec<bool> = (0..k).map(|i| i % 3 == 0).collect();
        let msg2: Vec<bool> = (0..k).map(|i| i % 2 == 0).collect();

        // Generate their codewords
        let codeword1 = matrix.generate_standard_codeword(&msg1);
        let codeword2 = matrix.generate_standard_codeword(&msg2);

        // Create sum of messages (XOR)
        let msg_sum: Vec<bool> = msg1.iter().zip(msg2.iter()).map(|(&a, &b)| a ^ b).collect();

        // Generate codeword from sum
        let codeword_sum = matrix.generate_standard_codeword(&msg_sum);

        // Create sum of codewords (XOR)
        let mut direct_sum = [false; N];
        for i in 0..N {
            direct_sum[i] = codeword1[i] ^ codeword2[i];
        }

        // Test linearity: c(m1 + m2) = c(m1) + c(m2)
        assert_eq!(
            codeword_sum, direct_sum,
            "Codeword generation should be linear"
        );

        // Verify all codewords are valid
        assert!(
            matrix.is_standard_codeword(&codeword1),
            "First codeword should be valid"
        );
        assert!(
            matrix.is_standard_codeword(&codeword2),
            "Second codeword should be valid"
        );
        assert!(
            matrix.is_standard_codeword(&codeword_sum),
            "Sum codeword should be valid"
        );
        assert!(
            matrix.is_standard_codeword(&direct_sum),
            "Direct sum should be valid"
        );
    }

    #[test]
    fn test_codeword_modification() {
        let k = 91;
        let generator = CascadedGeneratorMatrix::new(0).unwrap();
        let mut matrix = BitMatrix::from_cascaded_generator(&generator);
        let rand_permutation: [usize; N] = generate_random_permutation();

        // Convert to standard form and transform
        matrix.to_standard_form();
        matrix.apply_standard_column_permutation(&rand_permutation);
        matrix.transform_to_systematic_form_standard();
        matrix.compute_standard_h();

        // Generate a valid codeword
        let msg: Vec<bool> = (0..k).map(|i| i % 2 == 0).collect();
        let mut codeword = matrix.generate_standard_codeword(&msg);

        // Verify it's valid
        assert!(
            matrix.is_standard_codeword(&codeword),
            "Original codeword should be valid"
        );

        // Test 1: Flip any single bit - should make codeword invalid
        for pos in 0..N {
            let original = codeword[pos];
            codeword[pos] = !original;
            assert!(
                !matrix.is_standard_codeword(&codeword),
                "Codeword with single bit flip at position {} should be invalid",
                pos
            );
            codeword[pos] = original;
        }

        // Test 2: Flip pairs of bits - most pairs should make codeword invalid
        // (except for certain pairs that form valid error patterns)
        let mut invalid_pair_count = 0;
        for i in 0..N {
            for j in (i + 1)..N {
                codeword[i] = !codeword[i];
                codeword[j] = !codeword[j];
                if !matrix.is_standard_codeword(&codeword) {
                    invalid_pair_count += 1;
                }
                codeword[i] = !codeword[i];
                codeword[j] = !codeword[j];
            }
        }

        // Most pairs should create invalid codewords
        let total_pairs = (N * (N - 1)) / 2;
        assert!(
            invalid_pair_count > total_pairs * 9 / 10,
            "Most double-bit errors should create invalid codewords"
        );
    }

    #[test]
    fn packed_basis_reduction_matches_scalar_for_all_cascade_levels() {
        for cascade_bits in 0..=14 {
            let generator = CascadedGeneratorMatrix::new(cascade_bits).unwrap();
            let k = 91 - cascade_bits;
            let pi1 = std::array::from_fn(|index| (index * 37) % N);

            let mut scalar = BitMatrix::from_cascaded_generator(&generator);
            scalar.to_standard_form();
            let scalar_result = scalar.reduce_standard_bitwise(&pi1);

            let mut packed = BitMatrix::from_cascaded_generator(&generator);
            packed.to_standard_form();
            let packed_result = packed.reduce_standard_packed(&pi1);

            assert_eq!(packed_result, scalar_result, "cascade_bits={cascade_bits}");
            assert_eq!(
                packed.standard_matrix, scalar.standard_matrix,
                "cascade_bits={cascade_bits}"
            );

            let message: Vec<bool> = (0..k).map(|index| index % 3 == 0).collect();
            assert_eq!(
                packed.encode_standard(&message),
                scalar.encode_standard(&message),
                "cascade_bits={cascade_bits}"
            );
        }
    }

    #[test]
    fn replacing_standard_rows_refreshes_all_derived_tables() {
        let permutation = std::array::from_fn(|index| N - 1 - index);

        for cascade_bits in 0..=14 {
            let generator = CascadedGeneratorMatrix::new(cascade_bits).unwrap();
            let mut reference = BitMatrix::from_cascaded_generator(&generator);
            reference.to_standard_form();
            let (rank, _) = reference.reduce_standard_packed(&permutation);
            assert_eq!(rank, reference.k());

            let rows = *reference.standard_matrix();
            reference.compute_transposed_standard();
            reference.compute_parity_matrix();
            reference.compute_standard_parity_checks();

            let mut replaced = BitMatrix::new(reference.k());
            replaced.replace_standard_rows(rows);

            assert_eq!(replaced.standard_matrix, reference.standard_matrix);
            assert_eq!(replaced.standard_matrix_t, reference.standard_matrix_t);
            assert_eq!(replaced.parity_matrix, reference.parity_matrix);
            assert_eq!(replaced.standard_h, reference.standard_h);
        }
    }

    #[test]
    fn test_generate_codeword_components() {
        let num_cascaded_bits = 5;
        let k = 91 - num_cascaded_bits;
        let generator = CascadedGeneratorMatrix::new(num_cascaded_bits).unwrap();
        let mut matrix = BitMatrix::from_cascaded_generator(&generator);
        let rand_permutation: [usize; N] = generate_random_permutation();

        // Convert to standard form and transform
        matrix.to_standard_form();
        let (rank, _cum_perm) = matrix.reduce_standard_bitwise(&rand_permutation);
        assert_eq!(rank, k, "Rank must reach k.");
        matrix.compute_standard_parity_checks();

        // Test message with known result
        let mut test_msg = vec![false; k];
        test_msg[0] = true;
        test_msg[1] = true;

        let codeword = matrix.generate_standard_codeword(&test_msg);

        assert!(matrix.is_standard_codeword(&codeword));

        // Verify systematic part
        for i in 0..k {
            assert_eq!(
                codeword[i], test_msg[i],
                "Systematic part mismatch at position {}",
                i
            );
        }

        // Manually compute each parity bit and compare
        for i in k..N {
            let mut computed_parity = false;
            for j in 0..k {
                if matrix.standard_bit(j, i) {
                    computed_parity ^= test_msg[j];
                }
            }
            assert_eq!(
                codeword[i], computed_parity,
                "Parity bit mismatch at position {}",
                i
            );
        }
    }

    /// Randomly permutes the columns of the matrix in standard form
    /// Returns the permutation used for testing/verification
    pub(crate) fn generate_random_permutation() -> [usize; N] {
        let mut rng = rand::rng();

        // Generate random weights for all positions
        let rand_vec: Vec<f32> = (0..N).map(|_| rng.random()).collect();

        // Create permutation by sorting indices by weights
        let mut indices: Vec<usize> = (0..N).collect();
        indices.sort_by(|&i, &j| {
            rand_vec[j]
                .partial_cmp(&rand_vec[i])
                .unwrap_or(Ordering::Equal)
        });

        let permutation: [usize; N] = indices.try_into().unwrap();

        permutation
    }

    #[test]
    fn test_h_g_multiplication() {
        let num_cascaded_bits_array = [0, 3, 7, 14];

        for &num_cascaded_bits in &num_cascaded_bits_array {
            let k = 91 - num_cascaded_bits;
            println!(
                "Testing with k = {} / num_cascaded_bits = {}",
                k, num_cascaded_bits
            );
            let generator = CascadedGeneratorMatrix::new(num_cascaded_bits).unwrap();
            let mut matrix = BitMatrix::from_cascaded_generator(&generator);
            let rand_permutation: [usize; N] = generate_random_permutation();

            // Convert to standard form and transform to systematic form.
            matrix.to_standard_form();
            let (rank, _) = matrix.reduce_standard_bitwise(&rand_permutation);
            assert_eq!(rank, k, "Matrix should have full rank");

            // Compute the parity-check matrix H.
            matrix.compute_standard_parity_checks();
            matrix.compute_transposed_standard();

            // Generate and verify a test codeword.
            let mut test_msg = vec![false; k];
            test_msg[0] = true;
            test_msg[1] = true;
            let codeword = matrix.encode_standard(&test_msg);
            assert!(
                matrix.is_valid_standard_codeword(&codeword),
                "Generated codeword should be valid"
            );

            // Verification of H * G^T = 0 using the structure of H and G in systematic form.
            // For each parity-check equation i and each generator row r,
            // we know that in systematic form G = [I | P] and H = [P^T | I].
            // Thus, the only nonzero contributions in the dot product come from:
            // - j = r (in the message part), where H[i, r] == G[r, k+i]
            // - j = k+i (the identity part of H), where the contribution is G[r, k+i].
            // Hence, the dot product is:
            //    dot = G[r, k+i] ⊕ G[r, k+i] = 0.
            for i in 0..(N - k) {
                for r in 0..k {
                    // For the message part: only consider j == r.
                    let msg_contrib = if r < k {
                        // H[i, r] should equal G[r, k+i] (since H[i, r] was set from that bit).
                        let chunk = r / 64;
                        let bit = r % 64;
                        ((matrix.standard_h[i][chunk] >> bit) & 1) != 0
                    } else {
                        false
                    };
                    // For the identity part: contribution is G[r, k+i].
                    let id_contrib = matrix.standard_bit(r, k + i);
                    // The invariant is: msg_contrib ⊕ id_contrib == 0.
                    let dot = msg_contrib ^ id_contrib;
                    assert!(
                        !dot,
                        "H * G^T nonzero at parity equation {} and generator row {}: msg_contrib {} ⊕ id_contrib {} is 1",
                        i,
                        r,
                        msg_contrib as u8,
                        id_contrib as u8
                    );
                }
            }
        }
    }

    #[test]
    fn test_encode_standard_and_parity_checks() {
        // Test over different cascade lengths.
        let cascaded_bits_arr = [0, 3, 7, 14];
        for &num_cascaded_bits in cascaded_bits_arr.iter() {
            let k = 91 - num_cascaded_bits;
            println!(
                "Testing with num_cascaded_bits = {} (k = {})",
                num_cascaded_bits, k
            );

            // Create the cascaded generator matrix and then the BitMatrix.
            let generator = CascadedGeneratorMatrix::new(num_cascaded_bits)
                .expect("Failed to create CascadedGeneratorMatrix");
            let mut matrix = BitMatrix::from_cascaded_generator(&generator);

            // Generate a random permutation.
            let rand_permutation: [usize; N] = generate_random_permutation();

            // Convert to standard form and transform to systematic form.
            matrix.to_standard_form();
            let (rank, _) = matrix.reduce_standard_bitwise(&rand_permutation);
            assert_eq!(rank, k, "Matrix should have full rank (k = {})", k);

            // Compute the parity-check matrix H and precompute the transposed standard matrix.
            matrix.compute_standard_parity_checks();
            matrix.compute_transposed_standard();

            // For a few randomly generated messages, generate codewords and verify validity.
            for _ in 0..10 {
                let test_msg = generate_random_message(k);
                let codeword = matrix.encode_standard(&test_msg);
                assert!(
                    matrix.is_valid_standard_codeword(&codeword),
                    "Generated codeword should be valid for num_cascaded_bits {} with message {:?}",
                    num_cascaded_bits,
                    test_msg
                );
            }
        }
    }

    #[test]
    fn test_standard_codeword_generation() {
        // Create a base generator matrix and convert to systematic form
        let k = 91;
        let generator = CascadedGeneratorMatrix::new(0).unwrap();
        let mut matrix = BitMatrix::from_cascaded_generator(&generator);
        let rand_permutation: [usize; N] = generate_random_permutation();

        // Convert to standard form and transform to systematic
        matrix.to_standard_form();
        matrix.apply_standard_column_permutation(&rand_permutation);
        let (rank, _) = matrix.transform_to_systematic_form_standard();
        assert_eq!(rank, k, "Matrix should have full rank");

        // Compute parity check matrix
        matrix.compute_standard_h();

        // Create test message with m0=1, m1=1
        let mut message = vec![false; k];
        message[0] = true;
        message[1] = true;

        // Generate codeword
        let codeword = matrix.generate_standard_codeword(&message);

        // First k bits should match message
        for i in 0..k {
            assert_eq!(codeword[i], message[i]);
        }

        assert!(
            matrix.is_valid_standard_codeword(&codeword),
            "Generated codeword should be valid"
        );
        assert!(
            matrix.is_standard_codeword(&codeword),
            "Generated codeword should be valid"
        );
    }

    #[test]
    fn test_standard_reduction_permutation_validity() {
        // Create generator matrix
        let k = 91;
        let generator = CascadedGeneratorMatrix::new(0).unwrap();
        let mut matrix = BitMatrix::from_cascaded_generator(&generator);

        // Convert to standard form (k x N)
        matrix.to_standard_form();

        // Generate test message and get original codeword
        let message: [bool; 91] = std::array::from_fn(|i| i % 2 == 0);
        let original_codeword = matrix.generate_standard_codeword(&message);
        matrix.compute_standard_parity_checks();
        assert!(
            matrix.is_standard_codeword(&original_codeword),
            "Generated codeword should be valid"
        );

        // Generate random permutation (π1), to simulate the reliability permutation
        let pi1: [usize; N] = generate_random_permutation();

        // Transform to systematic form using GE, giving π1 and getting the cumulative permutation
        // (π1 and π2) in return, along with the rank.
        let (rank, cum_perm) = matrix.reduce_standard_bitwise(&pi1);
        assert_eq!(rank, k, "Matrix should have full rank");

        // Verify systematic form - check identity portion
        for i in 0..k {
            for j in 0..k {
                assert_eq!(
                    matrix.standard_bit(i, j),
                    i == j,
                    "Position ({}, {}) should be {} in identity portion",
                    i,
                    j,
                    if i == j { 1 } else { 0 }
                );
            }
        }

        // Compute H matrix and verify codeword validity
        matrix.compute_standard_parity_checks();

        // Encode in the most reliable basis before restoring natural coordinate order.
        let c_mrb = matrix.generate_standard_codeword(&message);

        let mut inverse_cum_perm = [0; N];
        for i in 0..N {
            inverse_cum_perm[cum_perm[i]] = i;
        }

        let mut c_orig = [false; 174];
        for i in 0..N {
            c_orig[i] = c_mrb[inverse_cum_perm[i]];
        }

        // Check against the parity equations in natural coordinate order.
        let mut matrix = BitMatrix::from_cascaded_generator(&generator);
        matrix.to_standard_form();
        matrix.compute_standard_parity_checks();

        assert!(
            matrix.is_standard_codeword(&c_orig),
            "Generated codeword transformed into c_orig should be valid"
        );
    }
}
