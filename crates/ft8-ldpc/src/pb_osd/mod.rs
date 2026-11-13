//! Probability-based ordered-statistics decoding (PB-OSD) for the FT8 (174,91) code.
//!
//! The decoder follows C. Yue, M. Shirvanimoghaddam, G. Park, O.-S. Park, B. Vucetic and
//! Y. Li, "Probability-Based Ordered-Statistics Decoding for Short Block Codes", IEEE
//! Communications Letters, 2021 (DOI 10.1109/LCOMM.2021.3058978). The default
//! [`SearchStrategy::LayeredBranchPruned`] visits test error patterns in breadth layers
//! without a global probability sort; `SimpleByWeight` enumerates by weight.
//! `EagerProbabilitySort` and `LazyProbabilityBestFirst` visit patterns in descending
//! probability order. A success probability stops the search early and a promising
//! probability discards patterns that cannot improve the best codeword.
//! Ordered-statistics decoding is due to M. P. C. Fossorier and S. Lin, IEEE Transactions on
//! Information Theory, vol. 41, no. 5, 1995. The basis reduction can process several pivots per
//! panel with Gray-code combination tables (the Method of Four Russians: V. L. Arlazarov,
//! E. A. Dinic, M. A. Kronrod and I. A. Faradzev, 1970; M. Albrecht, G. Bard and C. Pernet,
//! arXiv:1111.6549, Appendix A, Algorithm 6). The Q-function approximation constants are from
//! M. López-Benítez and F. Casadevall, IEEE Transactions on Communications, vol. 59, no. 4, 2011.
//!
//! Inputs follow the [crate-level LLR convention](crate).
//!
//! The LDPC tables and the CRC definition are from S. Franke, B. Somerville and J. Taylor, "The
//! FT4 and FT8 Communication Protocols", QEX, July/August 2020, through the `ft8-ldpc-core` crate.
//!
//! The primary [`DecodeResult::codeword`] and [`DecodeResult::llr_distance`] describe the
//! lowest-distance candidate found, which may fail the CRC; [`DecodeResult::success`]
//! reports finding a candidate, not CRC validity. [`DecodeResult::best_valid`] holds the
//! lowest-distance CRC-valid candidate accepted by the gate during search, or `None`.
//! Read payload bits from the selected codeword's first 77 positions.
//!
//! A decoder may hold a [`MessageGate`] that vets CRC-valid search candidates only when
//! they improve the best accepted internal-scale distance. The default gate accepts every improvement.
//! The gate does not control the primary result or `success`. An input that already
//! satisfies the code and CRC is returned as the primary and valid candidate at zero distance
//! without calling the gate.

#![deny(missing_docs)]
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

mod distance;
mod encoder;
#[cfg(target_arch = "aarch64")]
mod exp_coefficients;
#[cfg(test)]
mod ge;
#[cfg(test)]
mod integration_tests;
mod message_gate;
mod normal;
mod probability;
#[cfg(test)]
#[path = "../../tools/pb_osd/q_validation.rs"]
mod q_validation;
#[cfg(test)]
mod scoring_tests;
mod simd;
#[cfg(test)]
mod surgical_tests;
mod tep;
mod tep_iterator;

use encoder::FrameDecoder;
pub use encoder::{CalibrationBin, CalibrationTrace, Diagnostics, SearchStats, StopReason};
#[cfg(test)]
use ge::BitMatrix;
pub use message_gate::{AcceptAll, MessageGate, SharedMessageGate};

/// Basis-reduction implementation used while constructing a PB-OSD decoder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum BasisReductionKernel {
    /// Select the production kernel for the active message length.
    #[default]
    Auto,
    /// Bit-at-a-time elimination.
    Bitwise,
    /// Eliminate a row with three packed `u64` XOR operations.
    Packed,
    /// Four pivots per panel, with Gray-code combination tables.
    M4ri4,
    /// Five pivots per panel, with Gray-code combination tables.
    M4ri5,
}

impl BasisReductionKernel {
    pub(crate) const fn resolved(self) -> Self {
        match self {
            Self::Auto => Self::M4ri4,
            kernel => kernel,
        }
    }
}

/// TEP scheduling policy used by PB-OSD research experiments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchStrategy {
    /// Existing breadth-layer traversal with descendant pruning and fused batches.
    #[default]
    LayeredBranchPruned,
    /// Enumerate every weight in order; a rejected TEP does not remove descendants.
    SimpleByWeight,
    /// Materialize all TEPs and sort them by probability before evaluation.
    EagerProbabilitySort,
    /// Yue Algorithm 1 selection-list traversal using a probability max-heap.
    LazyProbabilityBestFirst,
}

/// Threshold behavior for PB-OSD probability controls.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum Threshold {
    /// Preserve the threshold derived by the input reliabilities.
    #[default]
    Auto,
    /// Multiply the derived threshold by a non-negative finite factor.
    AutoScaled(f32),
    /// Disable the probability control.
    Disabled,
    /// Override the derived threshold with a probability in `[0, 1]`.
    Explicit(f32),
}

impl Threshold {
    fn validate_scale(scale: f32) -> f32 {
        assert!(
            scale.is_finite() && scale >= 0.0,
            "PB-OSD automatic threshold scale must be finite and non-negative"
        );
        scale
    }

    fn override_value(self, disabled_value: f32) -> Option<f32> {
        match self {
            Self::Auto => None,
            Self::AutoScaled(_) => None,
            Self::Disabled => Some(disabled_value),
            Self::Explicit(value) => {
                assert!(
                    value.is_finite() && (0.0..=1.0).contains(&value),
                    "explicit PB-OSD threshold must be a finite probability"
                );
                Some(value)
            }
        }
    }

    fn apply_p_pro(self, decoder: &mut FrameDecoder) {
        if let Self::AutoScaled(scale) = self {
            decoder.scale_p_pro_threshold(Self::validate_scale(scale));
            return;
        }
        if let Some(value) = self.override_value(0.0) {
            decoder.set_p_pro_threshold(value);
        }
    }

    fn apply_p_suc(self, decoder: &mut FrameDecoder) {
        if let Self::AutoScaled(scale) = self {
            decoder.scale_p_suc_threshold(Self::validate_scale(scale));
            return;
        }
        if let Some(value) = self.override_value(f32::INFINITY) {
            decoder.set_p_suc_threshold(value);
        }
    }
}

/// MRB error-count model used to derive automatic probability thresholds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum EpsilonModel {
    /// Approximate all MRB bits with their mean error probability.
    #[default]
    BinomialMean,
    /// Use the individual MRB bit-error probabilities exactly.
    ExactPoissonBinomial,
}

/// Approximation used for PB-OSD's promising probability.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum PromisingProbabilityModel {
    /// Use Yue et al.'s parity-count model with mean probability and reliability.
    #[default]
    PaperMeanBinomial,
    /// Approximate parity weighted distance from its individual-bit moments.
    WeightedNormal,
}

/// PB-OSD search and input-scaling settings.
#[derive(Clone)]
pub struct Config {
    /// CRC bits folded into the generator, from 0 through 14.
    pub cascade_bits: usize,
    /// Largest test-error-pattern weight; search patterns hold at most eight positions.
    pub max_order: usize,
    /// Scale used by probability calculations and conversion to LLR-distance units.
    pub gamma: f32,
    /// Exact natural-order reliabilities, or `None` to derive `abs(llr) / (4 * gamma)`.
    pub reliabilities: Option<[f32; 174]>,
    /// Positive finite multiplier applied only to probability-model LLR magnitudes.
    pub probability_temperature: f32,
    /// Gaussian-elimination kernel used to construct each frame's reliable basis.
    pub basis_reduction_kernel: BasisReductionKernel,
    /// Test-error-pattern traversal policy.
    pub search_strategy: SearchStrategy,
    /// Error-count model used to derive automatic thresholds.
    pub epsilon_model: EpsilonModel,
    /// Model used to estimate whether a pattern can improve the incumbent.
    pub promising_probability_model: PromisingProbabilityModel,
    /// Promising-probability pruning threshold.
    pub p_pro_threshold: Threshold,
    /// Success-probability stopping threshold.
    pub p_suc_threshold: Threshold,
    /// Whether only CRC-valid candidates tighten the pruning distance.
    pub crc_gated_ppro: bool,
    /// Whether to include detailed distance-improvement diagnostics.
    pub collect_diagnostics: bool,
    /// Whether to include search-scheduler counters.
    pub collect_search_stats: bool,
    /// Whether reference schedulers collect probability-calibration bins.
    pub collect_calibration_trace: bool,
    /// Acceptance check for CRC-valid improvements; defaults to [`AcceptAll`].
    pub message_gate: SharedMessageGate,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cascade_bits: 0,
            max_order: 2,
            gamma: 1.0,
            reliabilities: None,
            probability_temperature: 1.0,
            basis_reduction_kernel: BasisReductionKernel::default(),
            search_strategy: SearchStrategy::default(),
            epsilon_model: EpsilonModel::default(),
            promising_probability_model: PromisingProbabilityModel::default(),
            p_pro_threshold: Threshold::Auto,
            p_suc_threshold: Threshold::Auto,
            crc_gated_ppro: false,
            collect_diagnostics: false,
            collect_search_stats: false,
            collect_calibration_trace: false,
            message_gate: std::sync::Arc::new(AcceptAll),
        }
    }
}

/// A configuration that cannot initialize the probability model or generator.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The CRC cascade count exceeds fourteen.
    CascadeBitsOutOfRange {
        /// Requested cascade count.
        value: usize,
        /// Largest supported cascade count.
        max: usize,
    },
    /// The probability temperature is nonfinite or nonpositive.
    ProbabilityTemperature,
    /// An explicit reliability is nonfinite or negative.
    Reliability {
        /// Natural codeword position of the invalid value.
        index: usize,
    },
    /// The product `4 * gamma * reliability` is nonfinite or negative.
    ProbabilityMagnitude {
        /// Natural codeword position of the invalid product.
        index: usize,
    },
    /// Derived reliabilities require a positive finite value of `4 * gamma`.
    ReliabilityScale,
    /// The promising-probability threshold or scale is outside its allowed range.
    PromisingThreshold,
    /// The success-probability threshold or scale is outside its allowed range.
    SuccessThreshold,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CascadeBitsOutOfRange { value, max } => {
                write!(f, "cascade_bits must be in 0..={max}, got {value}")
            }
            Self::ProbabilityTemperature => {
                f.write_str("probability_temperature must be positive and finite")
            }
            Self::Reliability { index } => {
                write!(f, "reliability {index} must be finite and non-negative")
            }
            Self::ProbabilityMagnitude { index } => write!(
                f,
                "probability magnitude {index} must be finite and non-negative"
            ),
            Self::ReliabilityScale => {
                f.write_str("4 * gamma must be positive and finite when deriving reliabilities")
            }
            Self::PromisingThreshold => f.write_str("invalid promising-probability threshold"),
            Self::SuccessThreshold => f.write_str("invalid success-probability threshold"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    /// Checks generator, reliability, temperature, and probability-threshold constraints.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.cascade_bits > 14 {
            return Err(ConfigError::CascadeBitsOutOfRange {
                value: self.cascade_bits,
                max: 14,
            });
        }
        if !self.probability_temperature.is_finite() || self.probability_temperature <= 0.0 {
            return Err(ConfigError::ProbabilityTemperature);
        }
        if let Some(alpha) = &self.reliabilities {
            for (index, &value) in alpha.iter().enumerate() {
                if !value.is_finite() || value < 0.0 {
                    return Err(ConfigError::Reliability { index });
                }
                let magnitude = 4.0 * self.gamma * value;
                if !magnitude.is_finite() || magnitude < 0.0 {
                    return Err(ConfigError::ProbabilityMagnitude { index });
                }
            }
        } else {
            let scale = 4.0 * self.gamma;
            if !scale.is_finite() || scale <= 0.0 {
                return Err(ConfigError::ReliabilityScale);
            }
        }
        for (error, threshold) in [
            (ConfigError::PromisingThreshold, self.p_pro_threshold),
            (ConfigError::SuccessThreshold, self.p_suc_threshold),
        ] {
            let valid = match threshold {
                Threshold::Auto | Threshold::Disabled => true,
                Threshold::AutoScaled(value) => value.is_finite() && value >= 0.0,
                Threshold::Explicit(value) => value.is_finite() && (0.0..=1.0).contains(&value),
            };
            if !valid {
                return Err(error);
            }
        }
        Ok(())
    }
}

/// Search outcome corresponding to the selected primary candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum DecodeKind {
    /// The engine explicitly reported failure; also check [`DecodeResult::success`].
    Failed = 0,
    /// A candidate was returned before exhausting the search.
    EarlyStop = 1,
    /// The search completed, whether or not a finite-distance candidate was found.
    Exhausted = 2,
}

/// Primary and CRC-valid candidates returned by one PB-OSD search.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodeResult {
    /// Whether any candidate was found, independently of CRC or gate acceptance.
    pub success: bool,
    /// Whether the search stopped early or completed; candidate availability is given by `success`.
    pub decode_kind: DecodeKind,
    /// Lowest-distance candidate in natural order, or the input hard decisions when `success` is false.
    /// The first 77 bits contain the payload when a candidate is available.
    pub codeword: [bool; 174],
    /// Differences from the input hard decisions, meaningful only when `success` is true.
    /// Holds zero otherwise; never uses a negative sentinel.
    pub hard_error_count: i32,
    /// Primary candidate distance in LLR units, meaningful only when `success` is true.
    /// Otherwise holds infinity, or NaN when `gamma` is zero.
    /// Scaling can also produce a nonfinite distance for a successfully selected candidate.
    pub llr_distance: f32,
    /// Lowest-distance CRC-valid candidate accepted by the gate, if any.
    /// An already-valid input is included without calling the gate.
    pub best_valid: Option<ValidCandidate>,
    /// Number of test error patterns processed.
    pub num_teps_processed: u32,
    /// Final test-error-pattern depth reached.
    pub final_depth: u8,
    /// Whether an early-stopping condition ended the search.
    pub early_stop: bool,
    /// Optional diagnostics with distances in the internal reliability scale.
    pub diagnostics: Option<Diagnostics>,
    /// Optional scheduler counters.
    pub search_stats: Option<SearchStats>,
    /// Optional calibration data from a reference scheduler.
    pub calibration_trace: Option<Box<CalibrationTrace>>,
}

/// A selected CRC-valid codeword and its distance.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ValidCandidate {
    /// Codeword in natural order, with the 77-bit payload in `codeword[..77]`.
    pub codeword: [bool; 174],
    /// Distance in LLR units after scaling the finite internal distance by `4 * gamma`.
    /// May be infinite or NaN after scaling; presence of this record establishes acceptance.
    pub llr_distance: f32,
}

/// Reusable PB-OSD configuration and generator, with per-frame basis construction.
pub struct Decoder {
    config: Config,
    generator: ft8_ldpc_core::fec::CascadedGeneratorMatrix,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new(Config::default()).expect("valid default configuration")
    }
}

impl Decoder {
    /// Validates the configuration and constructs its CRC-cascaded generator.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        config.validate()?;
        let generator = ft8_ldpc_core::fec::CascadedGeneratorMatrix::new(config.cascade_bits)
            .expect("validated cascade count");
        Ok(Self { config, generator })
    }

    /// Returns the configuration used for subsequent frames.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Searches a frame using the [crate-level LLR convention](crate).
    ///
    /// Each call constructs an independent reliable basis. Explicit reliabilities are
    /// used unchanged; otherwise `abs(llr) / (4 * gamma)` is evaluated in `f32`.
    ///
    /// # Panics
    ///
    /// Panics if derived reliabilities or their reconstructed probability magnitudes
    /// are nonfinite. With explicit reliabilities, LLRs are used only for hard
    /// decisions, including the floating-point comparison behavior for nonfinite values.
    /// Search patterns hold at most eight positions; very high orders can exhaust
    /// memory or overflow pattern counts.
    pub fn decode(&self, llr: &[f32; 174]) -> DecodeResult {
        let config = &self.config;
        let alpha = config
            .reliabilities
            .unwrap_or_else(|| std::array::from_fn(|i| llr[i].abs() / (4.0 * config.gamma)));
        let mut frame = FrameDecoder::new_with_cascaded_generator_and_probability_temperature(
            alpha,
            config.gamma,
            &self.generator,
            *llr,
            config.max_order,
            config.basis_reduction_kernel,
            config.probability_temperature,
        );
        frame.set_epsilon_model(config.epsilon_model);
        frame.set_promising_probability_model(config.promising_probability_model);
        config.p_pro_threshold.apply_p_pro(&mut frame);
        config.p_suc_threshold.apply_p_suc(&mut frame);
        frame.set_search_strategy(config.search_strategy);
        frame.set_crc_gated_ppro(config.crc_gated_ppro);
        frame.set_diagnostics_enabled(config.collect_diagnostics);
        frame.set_search_stats_enabled(config.collect_search_stats);
        frame.set_calibration_trace_enabled(config.collect_calibration_trace);
        frame.set_message_gate(Some(config.message_gate.clone()));
        let result = frame.decode();
        DecodeResult {
            success: result.success,
            decode_kind: match result.decode_type {
                0 => DecodeKind::Failed,
                1 => DecodeKind::EarlyStop,
                2 => DecodeKind::Exhausted,
                _ => unreachable!("frame decode kind"),
            },
            codeword: result.codeword,
            hard_error_count: result.hard_errors,
            llr_distance: result.distance,
            best_valid: result.valid_candidate_found.then_some(ValidCandidate {
                codeword: result.best_valid_codeword,
                llr_distance: result.best_valid_distance,
            }),
            num_teps_processed: result.num_teps_processed,
            final_depth: result.final_depth,
            early_stop: result.early_stop,
            diagnostics: result.diagnostics,
            search_stats: result.search_stats,
            calibration_trace: result.calibration_trace,
        }
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    fn input() -> [f32; 174] {
        std::array::from_fn(|i| {
            let magnitude = 0.25 + ((i * 17) % 31) as f32 * 0.0625;
            if i % 3 == 0 {
                -magnitude
            } else {
                magnitude
            }
        })
    }

    fn reference(config: &Config, llr: &[f32; 174]) -> Box<FrameDecoder> {
        let generator =
            ft8_ldpc_core::fec::CascadedGeneratorMatrix::new(config.cascade_bits).unwrap();
        let alpha = config
            .reliabilities
            .unwrap_or_else(|| llr.map(|x| x.abs() / (4.0 * config.gamma)));
        let mut frame = FrameDecoder::new_with_cascaded_generator_and_probability_temperature(
            alpha,
            config.gamma,
            &generator,
            *llr,
            config.max_order,
            config.basis_reduction_kernel,
            config.probability_temperature,
        );
        frame.set_epsilon_model(config.epsilon_model);
        frame.set_promising_probability_model(config.promising_probability_model);
        config.p_pro_threshold.apply_p_pro(&mut frame);
        config.p_suc_threshold.apply_p_suc(&mut frame);
        frame.set_search_strategy(config.search_strategy);
        frame.set_crc_gated_ppro(config.crc_gated_ppro);
        frame.set_diagnostics_enabled(config.collect_diagnostics);
        frame.set_search_stats_enabled(config.collect_search_stats);
        frame.set_calibration_trace_enabled(config.collect_calibration_trace);
        frame.set_message_gate(Some(config.message_gate.clone()));
        frame
    }

    #[test]
    fn configuration_reports_invalid_probability_inputs() {
        let mut config = Config::default();
        config.probability_temperature = f32::NAN;
        assert_eq!(config.validate(), Err(ConfigError::ProbabilityTemperature));
        config.probability_temperature = 1.0;
        config.reliabilities = Some([0.0; 174]);
        config.gamma = -1.0;
        assert_eq!(config.validate(), Ok(()));
        config.reliabilities.as_mut().unwrap()[13] = 1.0;
        assert_eq!(
            config.validate(),
            Err(ConfigError::ProbabilityMagnitude { index: 13 })
        );
        config.gamma = 0.0;
        assert_eq!(config.validate(), Ok(()));
        config.reliabilities.as_mut().unwrap()[13] = f32::INFINITY;
        assert_eq!(
            config.validate(),
            Err(ConfigError::Reliability { index: 13 })
        );
        config.reliabilities = None;
        assert_eq!(config.validate(), Err(ConfigError::ReliabilityScale));
        config.gamma = 1.0;
        config.p_pro_threshold = Threshold::AutoScaled(-1.0);
        assert_eq!(config.validate(), Err(ConfigError::PromisingThreshold));
    }

    #[test]
    fn wrapper_preserves_supplied_reliabilities_and_reuses_configuration() {
        let llr = input();
        let changed = llr.map(|x| -x * 0.5);
        let config = Config {
            max_order: 1,
            reliabilities: Some(std::array::from_fn(|i| (i % 11) as f32 * 0.125)),
            gamma: 0.75,
            collect_diagnostics: true,
            collect_search_stats: true,
            ..Config::default()
        };
        let decoder = Decoder::new(config.clone()).unwrap();
        for data in [&llr, &llr, &changed, &llr] {
            let actual = decoder.decode(data);
            let expected = reference(&config, data).decode();
            assert_eq!(actual.codeword, expected.codeword);
            assert_eq!(actual.llr_distance.to_bits(), expected.distance.to_bits());
            assert_eq!(actual.best_valid.is_some(), expected.valid_candidate_found);
            if let Some(candidate) = actual.best_valid {
                assert_eq!(
                    candidate.llr_distance.to_bits(),
                    expected.best_valid_distance.to_bits()
                );
                assert_eq!(candidate.codeword, expected.best_valid_codeword);
            }
            assert_eq!(actual.hard_error_count, expected.hard_errors);
            assert_eq!(actual.decode_kind as u8, expected.decode_type);
            assert_eq!(actual.num_teps_processed, expected.num_teps_processed);
            assert_eq!(actual.search_stats, expected.search_stats);
            assert_eq!(
                format!("{:?}", actual.diagnostics),
                format!("{:?}", expected.diagnostics)
            );
        }
        let mut old = reference(&config, &llr);
        let first = old.decode();
        let repeated = old.decode();
        assert_eq!(format!("{first:?}"), format!("{repeated:?}"));
    }

    #[test]
    fn gate_filters_crc_candidates_and_clean_inputs_bypass_it() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        struct Reject(Arc<AtomicUsize>);
        impl MessageGate for Reject {
            fn accept(&self, _: &[bool; 77], _: f32) -> bool {
                self.0.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let config = Config {
            cascade_bits: 14,
            max_order: 1,
            p_pro_threshold: Threshold::Disabled,
            p_suc_threshold: Threshold::Disabled,
            ..Config::default()
        };
        let accepted = Decoder::new(config.clone()).unwrap().decode(&input());
        assert!(accepted.best_valid.is_some());
        let decoder = Decoder::new(Config {
            message_gate: Arc::new(Reject(calls.clone())),
            ..config
        })
        .unwrap();
        let rejected = decoder.decode(&input());
        assert!(calls.load(Ordering::Relaxed) > 0);
        assert_eq!(rejected.codeword, accepted.codeword);
        assert_eq!(
            rejected.llr_distance.to_bits(),
            accepted.llr_distance.to_bits()
        );
        assert!(rejected.best_valid.is_none());
        let count = calls.load(Ordering::Relaxed);
        let clean = decoder.decode(&[4.0; 174]);
        assert_eq!(
            clean.best_valid.unwrap().llr_distance.to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(calls.load(Ordering::Relaxed), count);
    }

    #[test]
    fn valid_candidate_presence_survives_nonfinite_distance_scaling() {
        struct Reject;
        impl MessageGate for Reject {
            fn accept(&self, _: &[bool; 77], _: f32) -> bool {
                false
            }
        }
        let mut llr = [1.0; 174];
        llr[173] = -1.0;
        for strategy in [
            SearchStrategy::LayeredBranchPruned,
            SearchStrategy::SimpleByWeight,
            SearchStrategy::EagerProbabilitySort,
            SearchStrategy::LazyProbabilityBestFirst,
        ] {
            for gamma in [0.0, 0.125] {
                let config = Config {
                    gamma,
                    reliabilities: Some([f32::MAX / 2.0; 174]),
                    max_order: 0,
                    search_strategy: strategy,
                    p_pro_threshold: Threshold::Disabled,
                    p_suc_threshold: Threshold::Disabled,
                    ..Config::default()
                };
                let accepted = Decoder::new(config.clone()).unwrap().decode(&llr);
                assert!(accepted.success);
                let candidate = accepted.best_valid.unwrap();
                assert_eq!(candidate.codeword, [false; 174]);
                if gamma == 0.0 {
                    assert!(candidate.llr_distance.is_nan());
                } else {
                    assert_eq!(candidate.llr_distance, f32::INFINITY);
                }
                let rejected = Decoder::new(Config {
                    message_gate: std::sync::Arc::new(Reject),
                    ..config
                })
                .unwrap()
                .decode(&llr);
                assert!(rejected.success);
                assert!(rejected.best_valid.is_none());
                assert_eq!(rejected.codeword, accepted.codeword);
                assert_eq!(
                    rejected.llr_distance.to_bits(),
                    accepted.llr_distance.to_bits()
                );
            }
        }
    }

    #[test]
    fn unsuccessful_search_preserves_primary_placeholders() {
        let mut llr = [1.0; 174];
        llr[172..].fill(-1.0);
        for strategy in [
            SearchStrategy::LayeredBranchPruned,
            SearchStrategy::SimpleByWeight,
            SearchStrategy::EagerProbabilitySort,
            SearchStrategy::LazyProbabilityBestFirst,
        ] {
            for gamma in [0.0, 0.125] {
                let decoder = Decoder::new(Config {
                    gamma,
                    reliabilities: Some([f32::MAX; 174]),
                    max_order: 0,
                    search_strategy: strategy,
                    p_pro_threshold: Threshold::Disabled,
                    p_suc_threshold: Threshold::Disabled,
                    ..Config::default()
                })
                .unwrap();
                for _ in 0..2 {
                    let result = decoder.decode(&llr);
                    assert!(!result.success);
                    assert!(result.best_valid.is_none());
                    assert_eq!(result.codeword, llr.map(|value| value <= 0.0));
                    assert_eq!(result.hard_error_count, 0);
                    if gamma == 0.0 {
                        assert!(result.llr_distance.is_nan());
                    } else {
                        assert_eq!(result.llr_distance, f32::INFINITY);
                    }
                }
            }
        }
    }

    #[test]
    fn auto_thresholds_preserve_decoder_defaults() {
        assert_eq!(Threshold::Auto.override_value(0.0), None);
        assert_eq!(Threshold::Auto.override_value(f32::INFINITY), None);
        assert_eq!(Threshold::AutoScaled(4.0).override_value(0.0), None);
    }

    #[test]
    fn automatic_threshold_scale_must_be_finite_and_non_negative() {
        assert_eq!(Threshold::validate_scale(0.0), 0.0);
        assert_eq!(Threshold::validate_scale(4.0), 4.0);
    }

    #[test]
    fn disabled_thresholds_use_control_specific_sentinels() {
        assert_eq!(Threshold::Disabled.override_value(0.0), Some(0.0));
        assert_eq!(
            Threshold::Disabled.override_value(f32::INFINITY),
            Some(f32::INFINITY)
        );
    }
}
