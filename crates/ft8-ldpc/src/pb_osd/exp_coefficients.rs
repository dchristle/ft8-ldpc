//! Degree-seven relative-minimax exponential coefficients on [-ln(2), ln(2)].
//!
//! Estrin order: c0, c4, c2, c6, c1, c5, c3, c7.

pub(super) const EXP_COEFFICIENTS_ESTRIN: [f32; 8] = [
    f32::from_bits(0x3f800000), // c0
    f32::from_bits(0x3d2aa390), // c4
    f32::from_bits(0x3f00000a), // c2
    f32::from_bits(0x3ab91bcb), // c6
    f32::from_bits(0x3f800001), // c1
    f32::from_bits(0x3c0897a6), // c5
    f32::from_bits(0x3e2aaa6a), // c3
    f32::from_bits(0x394d4a8b), // c7
];
