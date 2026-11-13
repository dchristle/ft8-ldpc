//! Depth-first test-error-pattern traversal without a global probability sort.

use super::probability::ProbabilityCalculator;
use super::tep::TestErrorPattern;

/// Configuration for TEP generation
#[derive(Debug, Clone, Copy)]
pub struct TepConfig {
    /// Maximum number of positions per TEP
    pub max_order: usize,
    /// Message length (k)
    pub k: u8,
}

/// Generates Test Error Patterns by depth-first traversal of extending and adjacent
/// children without a global probability sort.
pub struct TepIterator {
    config: TepConfig,
    current_depth: usize,
    stack: Vec<TepNode>,
    finished: bool,
    probability_calc: Option<ProbabilityCalculator>,
}

/// Internal node representation for TEP tree traversal
#[derive(Debug, Clone)]
struct TepNode {
    tep: TestErrorPattern,
    next_position: u8,
}

impl TepIterator {
    /// Create a new TEP iterator with the given configuration
    pub fn new(config: TepConfig) -> Self {
        let mut iterator = Self {
            config,
            current_depth: 0,
            stack: Vec::new(),
            finished: false,
            probability_calc: None,
        };

        // Start with the empty TEP (no errors)
        iterator.stack.push(TepNode {
            tep: TestErrorPattern::new(),
            next_position: 0,
        });

        iterator
    }

    /// Set the probability calculator for accurate log P(e) calculations
    pub fn set_probability_calculator(&mut self, calc: ProbabilityCalculator) {
        self.probability_calc = Some(calc);
    }

    /// Generate all TEPs for a given order (depth)
    pub fn teps_for_order(&mut self, order: usize) -> Vec<TestErrorPattern> {
        let mut result = Vec::new();

        // Skip to the desired order
        while self.current_depth < order {
            if let Some(tep) = self.next() {
                // Skip TEPs until we reach the desired order
                if tep.num_positions() == order as u8 {
                    result.push(tep);
                    break;
                }
            } else {
                return result; // No more TEPs
            }
        }

        // Collect all TEPs at this order
        for tep in self.by_ref() {
            if tep.num_positions() == order as u8 {
                result.push(tep);
            } else if tep.num_positions() > order as u8 {
                // We've moved to the next order, stop
                break;
            }
        }

        result
    }

    /// Get the current depth of the iterator
    pub fn current_depth(&self) -> usize {
        self.current_depth
    }

    /// Check if the iterator has finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Reset the iterator to start from the beginning
    pub fn reset(&mut self) {
        self.current_depth = 0;
        self.stack.clear();
        self.finished = false;

        // Restart with empty TEP
        self.stack.push(TepNode {
            tep: TestErrorPattern::new(),
            next_position: 0,
        });
    }

    /// Generate children for a given TEP node
    fn generate_children(&self, node: &TepNode) -> Vec<TepNode> {
        let mut children = Vec::new();
        let current_order = node.tep.num_positions() as usize;

        // Don't generate children if we've reached max order
        if current_order >= self.config.max_order {
            return children;
        }

        // Create probability calculation closure
        let calc_log_pe = |tep: &TestErrorPattern| -> f32 {
            if let Some(ref prob_calc) = self.probability_calc {
                prob_calc.calculate_log_pe(tep)
            } else {
                // Fallback: use a simple heuristic based on number of positions
                -(tep.num_positions() as f32).ln()
            }
        };

        // Extended children: add new positions after the current ones
        if current_order < self.config.max_order {
            let start_pos = if current_order == 0 {
                0
            } else {
                node.tep.positions()[current_order - 1] + 1
            };

            for pos in start_pos..self.config.k {
                if let Some(child_tep) = node.tep.add_position(pos, calc_log_pe) {
                    // Only add if the position is within bounds
                    if child_tep.num_positions() <= self.config.max_order as u8 {
                        children.push(TepNode {
                            tep: child_tep,
                            next_position: pos + 1,
                        });
                    }
                }
            }
        }

        // Adjacent children: decrement the last position
        if current_order > 0 {
            if let Some(child_tep) = node.tep.decrement_last_position(calc_log_pe) {
                let last_idx = current_order - 1;
                let last_pos = node.tep.positions()[last_idx];
                children.push(TepNode {
                    tep: child_tep,
                    next_position: last_pos,
                });
            }
        }

        children
    }
}

impl Iterator for TepIterator {
    type Item = TestErrorPattern;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.stack.is_empty() {
            return None;
        }

        // Pop the next node from the stack
        let node = self.stack.pop()?;
        let current_tep = node.tep;

        // Update current depth
        self.current_depth = current_tep.num_positions() as usize;

        // Reverse insertion preserves child order when popping the stack.
        let children = self.generate_children(&node);
        for child in children.into_iter().rev() {
            self.stack.push(child);
        }

        if self.stack.is_empty() {
            self.finished = true;
        }

        Some(current_tep)
    }
}

/// Trait extension for TestErrorPattern to support tree operations
trait TepTreeOps {
    /// Add a position to the TEP if possible (extended child)
    fn add_position<F>(self, position: u8, calc_log_pe: F) -> Option<TestErrorPattern>
    where
        F: FnOnce(&TestErrorPattern) -> f32;

    /// Decrement the last position if possible (adjacent child)
    fn decrement_last_position<F>(self, calc_log_pe: F) -> Option<TestErrorPattern>
    where
        F: FnOnce(&TestErrorPattern) -> f32;
}

impl TepTreeOps for TestErrorPattern {
    fn add_position<F>(self, position: u8, calc_log_pe: F) -> Option<TestErrorPattern>
    where
        F: FnOnce(&TestErrorPattern) -> f32,
    {
        self.get_extended_child(position, calc_log_pe)
    }

    fn decrement_last_position<F>(self, calc_log_pe: F) -> Option<TestErrorPattern>
    where
        F: FnOnce(&TestErrorPattern) -> f32,
    {
        self.get_adjacent_child(calc_log_pe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tep_iterator_creation() {
        let config = TepConfig {
            max_order: 2,
            k: 10,
        };
        let iterator = TepIterator::new(config);
        assert_eq!(iterator.current_depth(), 0);
        assert!(!iterator.is_finished());
    }

    #[test]
    fn test_tep_iterator_basic_iteration() {
        let config = TepConfig { max_order: 2, k: 3 };
        let mut iterator = TepIterator::new(config);

        // Should start with empty TEP
        let first_tep = iterator.next().unwrap();
        assert_eq!(first_tep.num_positions(), 0);

        // Should generate single-position TEPs next
        let second_tep = iterator.next().unwrap();
        assert_eq!(second_tep.num_positions(), 1);
    }

    #[test]
    fn test_tep_iterator_reset() {
        let config = TepConfig { max_order: 2, k: 5 };
        let mut iterator = TepIterator::new(config);

        // Advance the iterator
        iterator.next();
        iterator.next();

        // Reset and verify we're back at the beginning
        iterator.reset();
        assert_eq!(iterator.current_depth(), 0);
        assert!(!iterator.is_finished());

        let first_tep = iterator.next().unwrap();
        assert_eq!(first_tep.num_positions(), 0);
    }
}
