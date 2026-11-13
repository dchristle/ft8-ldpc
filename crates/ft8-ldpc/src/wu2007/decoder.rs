//! Cached generators for validated cascade settings.
use ft8_ldpc_core::fec::CascadedGeneratorMatrix;
use std::sync::OnceLock;

// Cache each cascaded generator once per process and copy it into each decoder.
static GENERATORS: [OnceLock<CascadedGeneratorMatrix>; 15] = [const { OnceLock::new() }; 15];
pub(crate) fn generator(cascade_bits: usize) -> CascadedGeneratorMatrix {
    GENERATORS[cascade_bits]
        .get_or_init(|| CascadedGeneratorMatrix::new(cascade_bits).expect("validated cascade"))
        .clone()
}
