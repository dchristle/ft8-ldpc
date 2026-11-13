use std::cmp::Ordering;

const N: usize = 174;

// Compact storage for a test error pattern (TEP): the flipped positions of the
// most reliable basis (MRB) and the pattern's log-probability.

#[derive(Debug, Clone, Copy)]
#[repr(align(4))]
pub struct TestErrorPattern {
    log_pe_value: f32,  // log(P(e)) value for this TEP
    positions: [u8; 8], // Fixed array for positions of 1s
    num_positions: u8,  // Number of valid positions (0 to 8)
}

impl TestErrorPattern {
    // Accessor methods for private fields

    /// Get the log probability value
    pub(crate) fn log_pe_value(&self) -> f32 {
        self.log_pe_value
    }

    /// Get the number of positions
    pub fn num_positions(&self) -> u8 {
        self.num_positions
    }

    /// Get a reference to the positions array
    pub(crate) fn positions(&self) -> &[u8; 8] {
        &self.positions
    }

    /// Set the log probability value
    pub(crate) fn set_log_pe_value(&mut self, value: f32) {
        self.log_pe_value = value;
    }

    /// Set the number of positions
    pub(crate) fn set_num_positions(&mut self, num: u8) {
        self.num_positions = num;
    }

    /// Get a mutable reference to the positions array for internal modifications
    pub(crate) fn positions_mut(&mut self) -> &mut [u8; 8] {
        &mut self.positions
    }

    pub(crate) const fn new() -> Self {
        Self {
            log_pe_value: 1e-24,
            positions: [0; 8],
            num_positions: 0,
        }
    }

    pub fn with_positions(positions: &[u8]) -> Self {
        let mut tep = Self::new();
        assert!(
            positions.len() <= 8,
            "TEP cannot have more than 8 positions"
        );

        for (i, &pos) in positions.iter().enumerate() {
            tep.positions[i] = pos;
        }
        tep.num_positions = positions.len() as u8;
        tep
    }

    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_bit_array(&self) -> [u8; N] {
        let mut result = [0u8; N];
        for i in 0..self.num_positions as usize {
            result[self.positions[i] as usize] = 1;
        }
        result
    }

    // Get extended child (append k to positions)
    #[inline]
    pub(crate) fn get_extended_child<F>(&self, k: u8, calc_log_pe: F) -> Option<Self>
    where
        F: FnOnce(&TestErrorPattern) -> f32,
    {
        if self.num_positions >= 8 || // Array full
            self.num_positions > 0 && self.positions[self.num_positions as usize - 1] >= k
        {
            // Can't extend if the last position is >= k
            return None;
        }

        let mut child = *self;
        child.positions[child.num_positions as usize] = k;
        child.num_positions += 1;
        child.log_pe_value = calc_log_pe(&child);
        Some(child)
    }

    // Get adjacent child (decrement last position)
    #[inline]
    pub(crate) fn get_adjacent_child<F>(&self, calc_log_pe: F) -> Option<Self>
    where
        F: FnOnce(&TestErrorPattern) -> f32,
    {
        if self.num_positions == 0 {
            return None;
        }

        let last_idx = self.num_positions as usize - 1;
        if self.num_positions > 1 && self.positions[last_idx - 1] + 1 == self.positions[last_idx] {
            // if z_{l-1} + 1 = z_l, no adjacent TEP exists
            return None;
        }

        if self.positions[last_idx] == 0 {
            // adjacent TEP index would be out-of-range
            return None;
        }

        let mut child = *self;
        child.positions[last_idx] -= 1;
        child.log_pe_value = calc_log_pe(&child);
        Some(child)
    }
}

impl PartialOrd for TestErrorPattern {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for TestErrorPattern {
    fn eq(&self, other: &Self) -> bool {
        self.log_pe_value == other.log_pe_value
    }
}

impl Eq for TestErrorPattern {}

impl Ord for TestErrorPattern {
    fn cmp(&self, other: &Self) -> Ordering {
        // Use partial_cmp for f32 values, defaulting to Equal in case of NaN
        self.log_pe_value
            .partial_cmp(&other.log_pe_value)
            .unwrap_or(Ordering::Equal)
    }
}
