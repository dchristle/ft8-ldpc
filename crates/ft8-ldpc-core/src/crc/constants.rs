//! CRC dimensions for the FT8 protocol.

pub use crate::protocol::{CRC_LEN, MESSAGE_WITH_CRC_BITS, PAYLOAD_BITS};

const _: () = {
    assert!(PAYLOAD_BITS == 77, "PAYLOAD_BITS must be 77");
    assert!(CRC_LEN == 14, "CRC_LEN must be 14");
    assert!(
        MESSAGE_WITH_CRC_BITS == 91,
        "MESSAGE_WITH_CRC_BITS must be 91"
    );
};
