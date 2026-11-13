//! Optional acceptance check for CRC-valid candidate messages.
//!
//! During search, PB-OSD asks the gate whether a CRC-valid candidate that improves
//! the best valid distance may update the `best_valid_*` fields. With no gate every
//! such improvement is accepted. The gate does not control the primary result or
//! `success`, and an input that already satisfies the code and CRC bypasses it.

use std::sync::Arc;

/// Decides whether a CRC-valid 77-bit payload is acceptable.
pub trait MessageGate: Send + Sync {
    /// `weighted_distance` is the candidate's distance in the decoder's internal scale.
    fn accept(&self, message: &[bool; 77], weighted_distance: f32) -> bool;
}

/// Accepts every message.
#[derive(Debug, Clone, Copy, Default)]
pub struct AcceptAll;

impl MessageGate for AcceptAll {
    fn accept(&self, _message: &[bool; 77], _weighted_distance: f32) -> bool {
        true
    }
}

/// Shared acceptance check owned by decoder configurations.
pub type SharedMessageGate = Arc<dyn MessageGate>;

#[inline]
pub(super) fn gate_accepts(
    gate: &Option<SharedMessageGate>,
    message: &[bool],
    weighted_distance: f32,
) -> bool {
    match gate {
        None => true,
        Some(gate) => {
            let mut bits = [false; 77];
            bits.copy_from_slice(&message[..77]);
            gate.accept(&bits, weighted_distance)
        }
    }
}
