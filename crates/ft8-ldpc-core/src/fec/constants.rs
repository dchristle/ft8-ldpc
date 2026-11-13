//! Sparse parity-check adjacency and generator tables for the (174,91) LDPC
//! code shared by FT8 and FT4.

use crate::protocol::{K, M};

/// Three one-based parity-check indices for each codeword bit.
pub const CHECKS_FOR_BIT: [[usize; 3]; 174] = [
    [16, 45, 73],
    [25, 51, 62],
    [33, 58, 78],
    [1, 44, 45],
    [2, 7, 61],
    [3, 6, 54],
    [4, 35, 48],
    [5, 13, 21],
    [8, 56, 79],
    [9, 64, 69],
    [10, 19, 66],
    [11, 36, 60],
    [12, 37, 58],
    [14, 32, 43],
    [15, 63, 80],
    [17, 28, 77],
    [18, 74, 83],
    [22, 53, 81],
    [23, 30, 34],
    [24, 31, 40],
    [26, 41, 76],
    [27, 57, 70],
    [29, 49, 65],
    [3, 38, 78],
    [5, 39, 82],
    [46, 50, 73],
    [51, 52, 74],
    [55, 71, 72],
    [44, 67, 72],
    [43, 68, 78],
    [1, 32, 59],
    [2, 6, 71],
    [4, 16, 54],
    [7, 65, 67],
    [8, 30, 42],
    [9, 22, 31],
    [10, 18, 76],
    [11, 23, 82],
    [12, 28, 61],
    [13, 52, 79],
    [14, 50, 51],
    [15, 81, 83],
    [17, 29, 60],
    [19, 33, 64],
    [20, 26, 73],
    [21, 34, 40],
    [24, 27, 77],
    [25, 55, 58],
    [35, 53, 66],
    [36, 48, 68],
    [37, 46, 75],
    [38, 45, 47],
    [39, 57, 69],
    [41, 56, 62],
    [20, 49, 53],
    [46, 52, 63],
    [45, 70, 75],
    [27, 35, 80],
    [1, 15, 30],
    [2, 68, 80],
    [3, 36, 51],
    [4, 28, 51],
    [5, 31, 56],
    [6, 20, 37],
    [7, 40, 82],
    [8, 60, 69],
    [9, 10, 49],
    [11, 44, 57],
    [12, 39, 59],
    [13, 24, 55],
    [14, 21, 65],
    [16, 71, 78],
    [17, 30, 76],
    [18, 25, 80],
    [19, 61, 83],
    [22, 38, 77],
    [23, 41, 50],
    [7, 26, 58],
    [29, 32, 81],
    [33, 40, 73],
    [18, 34, 48],
    [13, 42, 64],
    [5, 26, 43],
    [47, 69, 72],
    [54, 55, 70],
    [45, 62, 68],
    [10, 63, 67],
    [14, 66, 72],
    [22, 60, 74],
    [35, 39, 79],
    [1, 46, 64],
    [1, 24, 66],
    [2, 5, 70],
    [3, 31, 65],
    [4, 49, 58],
    [1, 4, 5],
    [6, 60, 67],
    [7, 32, 75],
    [8, 48, 82],
    [9, 35, 41],
    [10, 39, 62],
    [11, 14, 61],
    [12, 71, 74],
    [13, 23, 78],
    [11, 35, 55],
    [15, 16, 79],
    [7, 9, 16],
    [17, 54, 63],
    [18, 50, 57],
    [19, 30, 47],
    [20, 64, 80],
    [21, 28, 69],
    [22, 25, 43],
    [13, 22, 37],
    [2, 47, 51],
    [23, 54, 74],
    [26, 34, 72],
    [27, 36, 37],
    [21, 36, 63],
    [29, 40, 44],
    [19, 26, 57],
    [3, 46, 82],
    [14, 15, 58],
    [33, 52, 53],
    [30, 43, 52],
    [6, 9, 52],
    [27, 33, 65],
    [25, 69, 73],
    [38, 55, 83],
    [20, 39, 77],
    [18, 29, 56],
    [32, 48, 71],
    [42, 51, 59],
    [28, 44, 79],
    [34, 60, 62],
    [31, 45, 61],
    [46, 68, 77],
    [6, 24, 76],
    [8, 10, 78],
    [40, 41, 70],
    [17, 50, 53],
    [42, 66, 68],
    [4, 22, 72],
    [36, 64, 81],
    [13, 29, 47],
    [2, 8, 81],
    [56, 67, 73],
    [5, 38, 50],
    [12, 38, 64],
    [59, 72, 80],
    [3, 26, 79],
    [45, 76, 81],
    [1, 65, 74],
    [7, 18, 77],
    [11, 56, 59],
    [14, 39, 54],
    [16, 37, 66],
    [10, 28, 55],
    [15, 60, 70],
    [17, 25, 82],
    [20, 30, 31],
    [12, 67, 68],
    [23, 75, 80],
    [27, 32, 62],
    [24, 69, 75],
    [19, 21, 71],
    [34, 53, 61],
    [35, 46, 47],
    [33, 59, 76],
    [40, 43, 83],
    [41, 42, 63],
    [49, 75, 83],
    [20, 44, 48],
    [42, 49, 57],
];

/// Flattened CHECKS_FOR_BIT matrix for cache-friendly sequential access.
/// Layout: [var0_check0, var0_check1, var0_check2, var1_check0, ...]
/// Values are 0-based check indices (already subtracted 1 from CHECKS_FOR_BIT's 1-based values).
/// 174 variables × 3 checks each = 522 bytes total.
/// All FT8 variables have exactly 3 check connections, so no sentinel needed.
pub const CHECKS_FOR_BIT_FLAT: [u8; 522] = {
    let mut result = [0u8; 522];
    let mut i = 0;
    while i < 174 {
        let mut j = 0;
        while j < 3 {
            // CHECKS_FOR_BIT values are 1-based, subtract 1 for 0-based.
            result[i * 3 + j] = (CHECKS_FOR_BIT[i][j] - 1) as u8;
            j += 1;
        }
        i += 1;
    }
    result
};

/// One-based codeword bit indices for each parity check, with zero padding.
pub const BITS_FOR_CHECK: [[usize; 7]; 83] = [
    [4, 31, 59, 91, 92, 96, 153],
    [5, 32, 60, 93, 115, 146, 0],
    [6, 24, 61, 94, 122, 151, 0],
    [7, 33, 62, 95, 96, 143, 0],
    [8, 25, 63, 83, 93, 96, 148],
    [6, 32, 64, 97, 126, 138, 0],
    [5, 34, 65, 78, 98, 107, 154],
    [9, 35, 66, 99, 139, 146, 0],
    [10, 36, 67, 100, 107, 126, 0],
    [11, 37, 67, 87, 101, 139, 158],
    [12, 38, 68, 102, 105, 155, 0],
    [13, 39, 69, 103, 149, 162, 0],
    [8, 40, 70, 82, 104, 114, 145],
    [14, 41, 71, 88, 102, 123, 156],
    [15, 42, 59, 106, 123, 159, 0],
    [1, 33, 72, 106, 107, 157, 0],
    [16, 43, 73, 108, 141, 160, 0],
    [17, 37, 74, 81, 109, 131, 154],
    [11, 44, 75, 110, 121, 166, 0],
    [45, 55, 64, 111, 130, 161, 173],
    [8, 46, 71, 112, 119, 166, 0],
    [18, 36, 76, 89, 113, 114, 143],
    [19, 38, 77, 104, 116, 163, 0],
    [20, 47, 70, 92, 138, 165, 0],
    [2, 48, 74, 113, 128, 160, 0],
    [21, 45, 78, 83, 117, 121, 151],
    [22, 47, 58, 118, 127, 164, 0],
    [16, 39, 62, 112, 134, 158, 0],
    [23, 43, 79, 120, 131, 145, 0],
    [19, 35, 59, 73, 110, 125, 161],
    [20, 36, 63, 94, 136, 161, 0],
    [14, 31, 79, 98, 132, 164, 0],
    [3, 44, 80, 124, 127, 169, 0],
    [19, 46, 81, 117, 135, 167, 0],
    [7, 49, 58, 90, 100, 105, 168],
    [12, 50, 61, 118, 119, 144, 0],
    [13, 51, 64, 114, 118, 157, 0],
    [24, 52, 76, 129, 148, 149, 0],
    [25, 53, 69, 90, 101, 130, 156],
    [20, 46, 65, 80, 120, 140, 170],
    [21, 54, 77, 100, 140, 171, 0],
    [35, 82, 133, 142, 171, 174, 0],
    [14, 30, 83, 113, 125, 170, 0],
    [4, 29, 68, 120, 134, 173, 0],
    [1, 4, 52, 57, 86, 136, 152],
    [26, 51, 56, 91, 122, 137, 168],
    [52, 84, 110, 115, 145, 168, 0],
    [7, 50, 81, 99, 132, 173, 0],
    [23, 55, 67, 95, 172, 174, 0],
    [26, 41, 77, 109, 141, 148, 0],
    [2, 27, 41, 61, 62, 115, 133],
    [27, 40, 56, 124, 125, 126, 0],
    [18, 49, 55, 124, 141, 167, 0],
    [6, 33, 85, 108, 116, 156, 0],
    [28, 48, 70, 85, 105, 129, 158],
    [9, 54, 63, 131, 147, 155, 0],
    [22, 53, 68, 109, 121, 174, 0],
    [3, 13, 48, 78, 95, 123, 0],
    [31, 69, 133, 150, 155, 169, 0],
    [12, 43, 66, 89, 97, 135, 159],
    [5, 39, 75, 102, 136, 167, 0],
    [2, 54, 86, 101, 135, 164, 0],
    [15, 56, 87, 108, 119, 171, 0],
    [10, 44, 82, 91, 111, 144, 149],
    [23, 34, 71, 94, 127, 153, 0],
    [11, 49, 88, 92, 142, 157, 0],
    [29, 34, 87, 97, 147, 162, 0],
    [30, 50, 60, 86, 137, 142, 162],
    [10, 53, 66, 84, 112, 128, 165],
    [22, 57, 85, 93, 140, 159, 0],
    [28, 32, 72, 103, 132, 166, 0],
    [28, 29, 84, 88, 117, 143, 150],
    [1, 26, 45, 80, 128, 147, 0],
    [17, 27, 89, 103, 116, 153, 0],
    [51, 57, 98, 163, 165, 172, 0],
    [21, 37, 73, 138, 152, 169, 0],
    [16, 47, 76, 130, 137, 154, 0],
    [3, 24, 30, 72, 104, 139, 0],
    [9, 40, 90, 106, 134, 151, 0],
    [15, 58, 60, 74, 111, 150, 163],
    [18, 42, 79, 144, 146, 152, 0],
    [25, 38, 65, 99, 122, 160, 0],
    [17, 42, 75, 129, 170, 172, 0],
];

/// Eight-byte check records containing a count followed by zero-based bit indices and zero padding.
pub const BITS_FOR_CHECK_FLAT_WITH_LEN: [u8; 83 * 8] = {
    let mut flat = [0u8; 83 * 8];
    let mut i = 0;
    while i < 83 {
        // First byte is count of non-zero entries
        let mut count = 0u8;
        let mut j = 0;
        while j < 7 {
            if BITS_FOR_CHECK[i][j] != 0 {
                count += 1;
            }
            j += 1;
        }
        flat[i * 8] = count;

        // Next bytes are the actual indices
        let mut k = 0;
        let mut written = 0;
        while k < 7 && written < count {
            if BITS_FOR_CHECK[i][k] != 0 {
                // The table uses one-based indices; packed entries use zero-based indices.
                flat[i * 8 + 1 + written as usize] = (BITS_FOR_CHECK[i][k] - 1) as u8;
                written += 1;
            }
            k += 1;
        }
        i += 1;
    }
    flat
};

/// Eight zero-based bit-index slots per check, with valid lengths in [`BIT_COUNT_FOR_CHECK`].
pub const BITS_FOR_CHECK_PADDED: [[u8; 8]; 83] = {
    let mut padded = [[0u8; 8]; 83];
    let mut i = 0;
    while i < 83 {
        let mut k = 0;
        let mut written = 0;
        while k < 7 {
            if BITS_FOR_CHECK[i][k] != 0 {
                // Convert to u8 and subtract 1 for 0-based indexing
                padded[i][written] = (BITS_FOR_CHECK[i][k] - 1) as u8;
                written += 1;
            }
            k += 1;
        }
        i += 1;
    }
    padded
};

/// Seven zero-based bit-index slots per check in a flat array, with lengths in [`BIT_COUNT_FOR_CHECK`].
pub const BITS_FOR_CHECK_FLAT: [u8; 83 * 7] = {
    let mut flat = [0u8; 83 * 7];
    let mut i = 0;
    while i < 83 {
        // Next bytes are the actual indices
        let mut k = 0;
        let mut written = 0;
        while k < 7 {
            if BITS_FOR_CHECK[i][k] != 0 {
                // The table uses one-based indices; packed entries use zero-based indices.
                flat[i * 7 + written as usize] = (BITS_FOR_CHECK[i][k] - 1) as u8;
                written += 1;
            }
            k += 1;
        }
        i += 1;
    }
    flat
};

/// Eight zero-based bit-index slots per check in a flat array, with lengths in [`BIT_COUNT_FOR_CHECK`].
pub const BITS_FOR_CHECK_FLAT_PADDED: [u8; 83 * 8] = {
    let mut flat = [0u8; 83 * 8];
    let mut i = 0;
    while i < 83 {
        // Next bytes are the actual indices
        let mut k = 0;
        let mut written = 0;
        while k < 7 {
            if BITS_FOR_CHECK[i][k] != 0 {
                // The table uses one-based indices; packed entries use zero-based indices.
                flat[i * 8 + written as usize] = (BITS_FOR_CHECK[i][k] - 1) as u8;
                written += 1;
            }
            k += 1;
        }
        i += 1;
    }
    flat
};

// Helper const fn to check the values are in range at compile time
const fn validate_nm_values() {
    let mut i = 0;
    while i < 83 {
        let mut j = 0;
        while j < 7 {
            // Verify all values fit in u8
            assert!(BITS_FOR_CHECK[i][j] <= 174);
            j += 1;
        }
        i += 1;
    }
}

// Force validation at compile time
const _: () = validate_nm_values();

/// Number of codeword bits participating in each parity check.
pub const BIT_COUNT_FOR_CHECK: [usize; 83] = [
    7, 6, 6, 6, 7, 6, 7, 6, 6, 7, 6, 6, 7, 7, 6, 6, 6, 7, 6, 7, 6, 7, 6, 6, 6, 7, 6, 6, 6, 7, 6, 6,
    6, 6, 7, 6, 6, 6, 7, 7, 6, 6, 6, 6, 7, 7, 6, 6, 6, 6, 7, 6, 6, 6, 7, 6, 6, 6, 6, 7, 6, 6, 6, 7,
    6, 6, 6, 7, 7, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6,
];
/// Number of parity checks connected to each codeword bit.
pub const CHECK_COUNT_PER_BIT: usize = 3;

/// The parity portion of the systematic generator matrix: 83 rows of 91 bits.
const GENERATOR_DAT: &str = include_str!("../../data/generator.dat");

/// Parity equations of the code: entry `[i][j]` is true when message bit `j` is XORed into
/// parity bit `i`.
pub const GENERATOR_PARITY_EQUATIONS: [[bool; K]; M] =
    parse_generator_dat(GENERATOR_DAT.as_bytes());

const fn is_binary(text: &[u8], start: usize, end: usize) -> bool {
    let mut i = start;
    while i < end {
        if text[i] != b'0' && text[i] != b'1' {
            return false;
        }
        i += 1;
    }
    true
}

/// Reads the rows of 91 binary digits and ignores the header lines.
const fn parse_generator_dat(text: &[u8]) -> [[bool; K]; M] {
    let mut rows = [[false; K]; M];
    let mut found = 0;
    let mut line_start = 0;
    let mut i = 0;
    while i <= text.len() {
        if i == text.len() || text[i] == b'\n' {
            let mut line_end = i;
            if line_end > line_start && text[line_end - 1] == b'\r' {
                line_end -= 1;
            }
            if line_end - line_start == K && is_binary(text, line_start, line_end) {
                assert!(found < M, "generator.dat has more than 83 rows");
                let mut j = 0;
                while j < K {
                    rows[found][j] = text[line_start + j] == b'1';
                    j += 1;
                }
                found += 1;
            }
            line_start = i + 1;
        }
        i += 1;
    }
    assert!(found == M, "generator.dat must hold 83 rows of 91 bits");
    rows
}

// ============================================================================
// O(1) Position Lookup Tables
// ============================================================================
// These tables eliminate linear searches when mapping between (check, variable)
// pairs and their positions in the sparse LDPC matrix representation.

/// Given a (bit, check) pair, returns the position (0-2) of that check
/// in the bit's connection list (`CHECKS_FOR_BIT[bit]`).
///
/// `BIT_CHECK_POSITION[bit][check]` is the position in `CHECKS_FOR_BIT[bit]`
/// where `CHECKS_FOR_BIT[bit][pos] == check + 1`.
///
/// Returns 255 if the variable and check are not connected (sentinel value).
///
/// # Usage
/// ```
/// use ft8_ldpc_core::fec::constants::{BIT_CHECK_POSITION, CHECKS_FOR_BIT};
///
/// let bit = 0;
/// let check = CHECKS_FOR_BIT[bit][0] - 1;
/// let pos = BIT_CHECK_POSITION[bit][check];
/// assert_eq!(pos, 0);
/// assert_eq!(BIT_CHECK_POSITION[bit][0], 255);
/// ```
pub const BIT_CHECK_POSITION: [[u8; 83]; 174] = {
    let mut table = [[255u8; 83]; 174];
    let mut var = 0;
    while var < 174 {
        let mut pos = 0;
        while pos < 3 {
            let check_1based = CHECKS_FOR_BIT[var][pos];
            if check_1based > 0 {
                let check_0based = check_1based - 1;
                table[var][check_0based] = pos as u8;
            }
            pos += 1;
        }
        var += 1;
    }
    table
};

/// Given a (check, variable) pair, returns the position (0-7) of that variable
/// in the check's connection list (`BITS_FOR_CHECK_PADDED[check]`).
///
/// `CHECK_BIT_POSITION[check][var]` = position in `BITS_FOR_CHECK_PADDED[check]` where
/// `BITS_FOR_CHECK_PADDED[check][pos] == var`
///
/// Returns 255 if the check and variable are not connected (sentinel value).
///
/// # Usage
/// ```
/// use ft8_ldpc_core::fec::constants::{BITS_FOR_CHECK_PADDED, CHECK_BIT_POSITION};
///
/// let check = 0;
/// let var = BITS_FOR_CHECK_PADDED[check][0] as usize;
/// let pos = CHECK_BIT_POSITION[check][var];
/// assert_eq!(pos, 0);
/// assert_eq!(CHECK_BIT_POSITION[check][0], 255);
/// ```
pub const CHECK_BIT_POSITION: [[u8; 174]; 83] = {
    let mut table = [[255u8; 174]; 83];
    let mut check = 0;
    while check < 83 {
        let mut pos = 0;
        while pos < 8 {
            let var_0based = BITS_FOR_CHECK_PADDED[check][pos] as usize;
            // BITS_FOR_CHECK_PADDED stores 0-based indices, with 0 as padding for unused slots.
            // Since bit 0 is valid, check against the actual row weight.
            if pos < BIT_COUNT_FOR_CHECK[check] && var_0based < 174 {
                table[check][var_0based] = pos as u8;
            }
            pos += 1;
        }
        check += 1;
    }
    table
};

// Compile-time validation of lookup tables
const _: () = {
    // Validate BIT_CHECK_POSITION: for each (bit, check) pair that should be connected,
    // verify the lookup returns the correct position
    let mut var = 0;
    while var < 174 {
        let mut pos = 0;
        while pos < 3 {
            let check_1based = CHECKS_FOR_BIT[var][pos];
            if check_1based > 0 {
                let check_0based = check_1based - 1;
                let lookup_pos = BIT_CHECK_POSITION[var][check_0based];
                assert!(lookup_pos == pos as u8);
            }
            pos += 1;
        }
        var += 1;
    }

    // Validate CHECK_BIT_POSITION: for each (check, bit) pair that should be connected,
    // verify the lookup returns the correct position
    let mut check = 0;
    while check < 83 {
        let mut pos = 0;
        while pos < BIT_COUNT_FOR_CHECK[check] {
            let var_0based = BITS_FOR_CHECK_PADDED[check][pos] as usize;
            if var_0based < 174 {
                let lookup_pos = CHECK_BIT_POSITION[check][var_0based];
                assert!(lookup_pos == pos as u8);
            }
            pos += 1;
        }
        check += 1;
    }
};

/// `VAR_CHECK_PRODUCT_IDX[bit][pos]` = (check_idx, product_pos) for linear product access.
///
/// For each bit (0..174) and each of its 3 connected checks (0..3),
/// gives the check index and position within that check's product array.
/// This enables linear iteration through bits when updating check-to-bit messages,
/// avoiding scattered lookups through `CHECKS_FOR_BIT_FLAT` and `CHECK_BIT_POSITION`.
///
/// # Layout
/// - Outer dimension: 174 bits
/// - Inner dimension: 3 checks per bit
/// - Value: (check_index_0based, position_in_check_products)
///
/// # Usage
/// ```
/// use ft8_ldpc_core::fec::constants::{BITS_FOR_CHECK_PADDED, CHECKS_FOR_BIT, VAR_CHECK_PRODUCT_IDX};
///
/// let bit = 0;
/// let pos = 0;
/// let (check_idx, product_pos) = VAR_CHECK_PRODUCT_IDX[bit][pos];
/// assert_eq!(check_idx as usize + 1, CHECKS_FOR_BIT[bit][pos]);
/// assert_eq!(BITS_FOR_CHECK_PADDED[check_idx as usize][product_pos as usize] as usize, bit);
/// ```
pub const VAR_CHECK_PRODUCT_IDX: [[(u8, u8); 3]; 174] = {
    let mut table = [[(0u8, 0u8); 3]; 174];
    let mut var = 0;
    while var < 174 {
        let mut pos = 0;
        while pos < 3 {
            // CHECKS_FOR_BIT[bit][pos] is a 1-based check index.
            let check_1based = CHECKS_FOR_BIT[var][pos];
            let check_0based = check_1based - 1;
            // CHECK_BIT_POSITION[check][bit] gives the position in the check's product array.
            let product_pos = CHECK_BIT_POSITION[check_0based][var];
            table[var][pos] = (check_0based as u8, product_pos);
            pos += 1;
        }
        var += 1;
    }
    table
};

// Validate VAR_CHECK_PRODUCT_IDX at compile time
const _VAR_CHECK_PRODUCT_IDX_VALIDATION: () = {
    let mut var = 0;
    while var < 174 {
        let mut pos = 0;
        while pos < 3 {
            let (check_idx, product_pos) = VAR_CHECK_PRODUCT_IDX[var][pos];
            // Verify check_idx matches CHECKS_FOR_BIT_FLAT.
            let expected_check = CHECKS_FOR_BIT_FLAT[var * 3 + pos];
            assert!(check_idx == expected_check);
            // Verify product_pos is valid (< 8)
            assert!(product_pos < 8);
            // Verify CHECK_BIT_POSITION gives the same result.
            let expected_pos = CHECK_BIT_POSITION[check_idx as usize][var];
            assert!(product_pos == expected_pos);
            pos += 1;
        }
        var += 1;
    }
};

/// `INCOMING_MSG_INDICES[check][pos]` = (var, msg_pos) for check-to-variable message lookup.
///
/// For each check (0..83) and each position within that check (0..8),
/// gives (bit_index, msg_position_in_check_to_bit).
/// This eliminates the `BIT_CHECK_POSITION` lookup inside the bit-to-check message loop.
///
/// # Layout
/// - `bit` = `BITS_FOR_CHECK_PADDED[check][pos]` (which bit connects to this check at this position)
/// - `msg_pos` = `BIT_CHECK_POSITION[bit][check]` (which slot in `check_to_bit[bit]` holds the message from this check)
///
/// # Usage
/// ```
/// use ft8_ldpc_core::fec::constants::{CHECKS_FOR_BIT, INCOMING_MSG_INDICES};
///
/// let check = 0;
/// let pos = 0;
/// let (bit, msg_pos) = INCOMING_MSG_INDICES[check][pos];
/// assert_eq!(CHECKS_FOR_BIT[bit as usize][msg_pos as usize], check + 1);
/// ```
pub const INCOMING_MSG_INDICES: [[(u8, u8); 8]; 83] = {
    let mut table = [[(0u8, 0u8); 8]; 83];
    let mut check = 0;
    while check < 83 {
        let mut pos = 0;
        while pos < 8 {
            let var = BITS_FOR_CHECK_PADDED[check][pos];
            let msg_pos = BIT_CHECK_POSITION[var as usize][check];
            table[check][pos] = (var, msg_pos);
            pos += 1;
        }
        check += 1;
    }
    table
};

// Validate INCOMING_MSG_INDICES at compile time
const _INCOMING_MSG_INDICES_VALIDATION: () = {
    let mut check = 0;
    while check < 83 {
        let mut pos = 0;
        while pos < BIT_COUNT_FOR_CHECK[check] {
            let (var, msg_pos) = INCOMING_MSG_INDICES[check][pos];
            // Verify bit matches BITS_FOR_CHECK_PADDED.
            assert!(var == BITS_FOR_CHECK_PADDED[check][pos]);
            // Verify msg_pos matches BIT_CHECK_POSITION.
            let expected_msg_pos = BIT_CHECK_POSITION[var as usize][check];
            assert!(msg_pos == expected_msg_pos);
            // Verify msg_pos is valid (< 3 for FT8's column weight)
            assert!(msg_pos < 3);
            pos += 1;
        }
        check += 1;
    }
};

/// Packed variable and message positions for each check edge.
///
/// Each entry stores the variable in the high byte and the message position
/// in the low byte: `(var << 8) | msg_pos`.
///
/// # Usage
/// ```
/// use ft8_ldpc_core::fec::constants::{INCOMING_MSG_INDICES, INCOMING_MSG_INDICES_PACKED};
///
/// let check = 0;
/// let pos = 0;
/// let packed = INCOMING_MSG_INDICES_PACKED[check][pos];
/// let var = (packed >> 8) as usize;
/// let msg_pos = (packed & 0xFF) as usize;
/// assert_eq!((var as u8, msg_pos as u8), INCOMING_MSG_INDICES[check][pos]);
/// ```
pub const INCOMING_MSG_INDICES_PACKED: [[u16; 8]; 83] = {
    let mut table = [[0u16; 8]; 83];
    let mut check = 0;
    while check < 83 {
        let mut pos = 0;
        while pos < 8 {
            let var = BITS_FOR_CHECK_PADDED[check][pos];
            let msg_pos = BIT_CHECK_POSITION[var as usize][check];
            table[check][pos] = ((var as u16) << 8) | (msg_pos as u16);
            pos += 1;
        }
        check += 1;
    }
    table
};

// Validate INCOMING_MSG_INDICES_PACKED at compile time
const _INCOMING_MSG_INDICES_PACKED_VALIDATION: () = {
    let mut check = 0;
    while check < 83 {
        let mut pos = 0;
        while pos < BIT_COUNT_FOR_CHECK[check] {
            let packed = INCOMING_MSG_INDICES_PACKED[check][pos];
            let var = (packed >> 8) as u8;
            let msg_pos = (packed & 0xFF) as u8;
            // Verify unpacked values match original constant
            let (expected_var, expected_msg_pos) = INCOMING_MSG_INDICES[check][pos];
            assert!(var == expected_var);
            assert!(msg_pos == expected_msg_pos);
            pos += 1;
        }
        check += 1;
    }
};

/// Packed check and product positions for each variable edge.
///
/// Each entry stores the check index in the high byte and the product position
/// in the low byte: `(check_idx << 8) | product_pos`.
///
/// # Usage
/// ```
/// use ft8_ldpc_core::fec::constants::{VAR_CHECK_PRODUCT_IDX, VAR_CHECK_PRODUCT_IDX_PACKED};
///
/// let var = 0;
/// let pos = 0;
/// let packed = VAR_CHECK_PRODUCT_IDX_PACKED[var][pos];
/// let check_idx = (packed >> 8) as usize;
/// let product_pos = (packed & 0xFF) as usize;
/// assert_eq!((check_idx as u8, product_pos as u8), VAR_CHECK_PRODUCT_IDX[var][pos]);
/// ```
pub const VAR_CHECK_PRODUCT_IDX_PACKED: [[u16; 3]; 174] = {
    let mut table = [[0u16; 3]; 174];
    let mut var = 0;
    while var < 174 {
        let mut pos = 0;
        while pos < 3 {
            let (check_idx, product_pos) = VAR_CHECK_PRODUCT_IDX[var][pos];
            table[var][pos] = ((check_idx as u16) << 8) | (product_pos as u16);
            pos += 1;
        }
        var += 1;
    }
    table
};

// Validate VAR_CHECK_PRODUCT_IDX_PACKED at compile time
const _VAR_CHECK_PRODUCT_IDX_PACKED_VALIDATION: () = {
    let mut var = 0;
    while var < 174 {
        let mut pos = 0;
        while pos < 3 {
            let packed = VAR_CHECK_PRODUCT_IDX_PACKED[var][pos];
            let check_idx = (packed >> 8) as u8;
            let product_pos = (packed & 0xFF) as u8;
            // Verify unpacked values match original constant
            let (expected_check, expected_pos) = VAR_CHECK_PRODUCT_IDX[var][pos];
            assert!(check_idx == expected_check);
            assert!(product_pos == expected_pos);
            // Verify check_idx < 83 and product_pos < 8 for safe unchecked access
            assert!(check_idx < 83);
            assert!(product_pos < 8);
            pos += 1;
        }
        var += 1;
    }
};
