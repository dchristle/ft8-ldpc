//! Fixed-point layered offset-min-sum decoder for FT8 LDPC.
//!
//! The decoder uses:
//! - layered scheduling
//! - quantized `i16` messages and beliefs
//! - degree-specialized min1/min2 check updates
//! - saturating integer arithmetic in the hot loop
//!
//! Input LLR conventions are documented at the [crate root](crate).
//! The kernels use the opposite sign (a non-negative belief is bit 1), so the sign is
//! flipped on entry and in `refined_llrs`.

#![deny(missing_docs)]

use ft8_ldpc_core::fec::constants::INCOMING_MSG_INDICES_PACKED;
use ft8_ldpc_core::protocol::{M, N};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const DEFAULT_MAX_MESSAGE_VALUE: i16 = 63;
const DEFAULT_MAX_BELIEF_VALUE: i16 = 255;
/// A stalled decode is not abandoned before this many iterations have run.
const MIN_ITERATIONS_BEFORE_STALL_STOP: usize = 3;
const DEGREE_6_CHECKS: [usize; 59] = [
    1, 2, 3, 5, 7, 8, 10, 11, 14, 15, 16, 18, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 35, 36,
    37, 40, 41, 42, 43, 46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 60, 61, 62, 64, 65, 66, 69, 70,
    72, 73, 74, 75, 76, 77, 78, 80, 81, 82,
];
const DEGREE_7_CHECKS: [usize; 24] = [
    0, 4, 6, 9, 12, 13, 17, 19, 21, 25, 29, 34, 38, 39, 44, 45, 50, 54, 59, 63, 67, 68, 71, 79,
];

/// Iteration, quantization, and distance-normalization settings.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// With 0, no iteration runs and `unsatisfied_checks` is reported as 83 without being computed.
    pub max_iterations: usize,
    /// Stop after this many iterations in a row without a new lowest unsatisfied-check count.
    pub stall_limit: usize,
    /// The decode succeeds when at most this many parity checks are unsatisfied.
    pub max_unsatisfied_checks: usize,
    /// Multiplier applied before rounding input LLRs to integers.
    pub llr_scale: f32,
    /// Magnitude reduction for check messages; negative values are treated as zero.
    pub offset: i16,
    /// Nonnegative magnitude bound for quantized input LLRs.
    pub max_channel_value: i16,
    /// Upper bound on check-message magnitudes.
    pub max_message_value: i16,
    /// Magnitude bound for beliefs; must be nonnegative when iterations are enabled.
    pub max_belief_value: i16,
    /// Distance normalizer; `None` uses one quarter of the mean absolute input LLR.
    pub gamma_estimate: Option<f32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_iterations: 8,
            stall_limit: 3,
            max_unsatisfied_checks: 0,
            llr_scale: 8.0,
            offset: 2,
            max_channel_value: 127,
            max_message_value: 63,
            max_belief_value: 255,
            gamma_estimate: None,
        }
    }
}

impl Config {
    /// Checks bounds used by the saturating arithmetic without changing settings.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_channel_value < 0 {
            return Err(ConfigError::NegativeChannelBound(self.max_channel_value));
        }
        if self.max_iterations > 0 && self.max_belief_value < 0 {
            return Err(ConfigError::NegativeBeliefBound(self.max_belief_value));
        }
        Ok(())
    }
}

/// A magnitude bound that cannot be used by the decoder's clamp operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The input quantization bound is negative.
    NegativeChannelBound(i16),
    /// The belief bound is negative while iterations are enabled.
    NegativeBeliefBound(i16),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NegativeChannelBound(value) => write!(f, "channel magnitude bound must be nonnegative, got {value}"),
            Self::NegativeBeliefBound(value) => write!(f, "belief magnitude bound must be nonnegative when iterations are enabled, got {value}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Shared decoding settings and optional distance normalizers for individual lanes.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchConfig<const LANES: usize> {
    /// Settings applied to every lane.
    pub decoder: Config,
    /// Per-lane normalizers; `None` uses [`Config::gamma_estimate`] for each lane.
    pub gamma_estimates: Option<[f32; LANES]>,
}

impl<const LANES: usize> BatchConfig<LANES> {
    /// Validates the shared decoder settings when at least one lane uses them.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if LANES == 0 {
            Ok(())
        } else {
            self.decoder.validate()
        }
    }
}

/// Hard decisions, parity-check convergence, and distance measurements for one input.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodeResult {
    /// Hard decisions in natural codeword order.
    pub codeword: [bool; N],
    /// Whether the unsatisfied-check count is within the configured threshold; CRC is not checked.
    pub success: bool,
    /// Number of completed decoding iterations.
    pub iterations: usize,
    /// Sum of `|LLR| / (4 * gamma)` at changed hard decisions, or zero when `gamma < 1e-9`.
    pub normalized_distance: f32,
    /// Number of decisions that differ from the input hard decisions.
    pub hard_error_count: usize,
    /// Number of failed parity checks, or 83 when no iteration ran.
    pub unsatisfied_checks: usize,
    /// Negated final integer beliefs divided by `llr_scale`; zero or nonfinite scales can yield nonfinite values.
    pub refined_llrs: [f32; N],
}

/// Scalar layered offset-min-sum decoder with reusable configuration.
#[derive(Debug, Clone)]
pub struct Decoder {
    config: Config,
}

/// Batched decoder that reuses working storage across calls.
#[derive(Debug)]
pub struct BatchDecoder<const LANES: usize> {
    config: Config,
    gamma_estimates: Option<[f32; LANES]>,
    llr_q: Box<[[i16; LANES]; N]>,
    c2v: Box<[[[i16; LANES]; 8]; M]>,
    converged_llr: Box<[[i16; N]; LANES]>,
    converged: [bool; LANES],
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new(Config::default()).expect("valid default configuration")
    }
}

impl Decoder {
    /// Builds a decoder after validating its configuration.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Decodes input LLRs in natural codeword order.
    pub fn decode(&self, channel_llrs: &[f32; N]) -> DecodeResult {
        let gamma_estimate = self.config.gamma_estimate;
        let offset = self.config.offset.max(0);
        let mut channel_q = [0i16; N];
        let mut llr_q = [0i16; N];
        let mut c2v = [[0i16; 3]; N];

        for i in 0..N {
            let q = quantize(
                -channel_llrs[i],
                self.config.llr_scale,
                self.config.max_channel_value,
            );
            channel_q[i] = q;
            llr_q[i] = q;
        }

        let mut lowest_unsatisfied = M;
        let mut stalled_iterations = 0usize;
        let mut final_unsatisfied = M;
        let mut iterations_run = 0usize;

        for iteration in 0..self.config.max_iterations {
            iterations_run = iteration + 1;

            if self.config.max_belief_value == DEFAULT_MAX_BELIEF_VALUE
                && self.config.max_message_value == DEFAULT_MAX_MESSAGE_VALUE
            {
                let mut i = 0usize;
                while i < DEGREE_6_CHECKS.len() {
                    dispatch_update_check_default_6(
                        DEGREE_6_CHECKS[i],
                        offset,
                        &mut c2v,
                        &mut llr_q,
                    );
                    i += 1;
                }
                let mut i = 0usize;
                while i < DEGREE_7_CHECKS.len() {
                    dispatch_update_check_default_7(
                        DEGREE_7_CHECKS[i],
                        offset,
                        &mut c2v,
                        &mut llr_q,
                    );
                    i += 1;
                }
            } else {
                let mut i = 0usize;
                while i < DEGREE_6_CHECKS.len() {
                    dispatch_update_check_runtime_6(
                        DEGREE_6_CHECKS[i],
                        offset,
                        self.config.max_message_value,
                        self.config.max_belief_value,
                        &mut c2v,
                        &mut llr_q,
                    );
                    i += 1;
                }
                let mut i = 0usize;
                while i < DEGREE_7_CHECKS.len() {
                    dispatch_update_check_runtime_7(
                        DEGREE_7_CHECKS[i],
                        offset,
                        self.config.max_message_value,
                        self.config.max_belief_value,
                        &mut c2v,
                        &mut llr_q,
                    );
                    i += 1;
                }
            }

            final_unsatisfied = count_unsatisfied_checks(&llr_q);
            if final_unsatisfied < lowest_unsatisfied {
                lowest_unsatisfied = final_unsatisfied;
                stalled_iterations = 0;
            } else {
                stalled_iterations += 1;
            }

            if final_unsatisfied <= self.config.max_unsatisfied_checks {
                break;
            }

            if stalled_iterations >= self.config.stall_limit
                && iteration + 1 >= MIN_ITERATIONS_BEFORE_STALL_STOP
            {
                break;
            }
        }

        let mut codeword = [false; N];
        let mut refined_llrs = [0.0f32; N];
        for i in 0..N {
            codeword[i] = llr_q[i] >= 0;
            refined_llrs[i] = refined_llr(llr_q[i], self.config.llr_scale);
        }

        let initial_decisions: [bool; N] = std::array::from_fn(|i| channel_llrs[i] <= 0.0);
        let hard_error_count = codeword
            .iter()
            .zip(initial_decisions.iter())
            .filter(|(a, b)| a != b)
            .count();

        let gamma_est = gamma_estimate.unwrap_or_else(|| mean_magnitude_gamma(channel_llrs));
        let normalized_distance = if gamma_est < 1e-9 {
            0.0
        } else {
            let normalizer = 4.0 * gamma_est;
            codeword
                .iter()
                .zip(initial_decisions.iter())
                .zip(channel_llrs.iter())
                .map(|((final_bit, initial_bit), llr)| {
                    if final_bit != initial_bit {
                        llr.abs() / normalizer
                    } else {
                        0.0
                    }
                })
                .sum()
        };

        DecodeResult {
            codeword,
            success: final_unsatisfied <= self.config.max_unsatisfied_checks,
            iterations: iterations_run,
            normalized_distance,
            hard_error_count,
            unsatisfied_checks: final_unsatisfied,
            refined_llrs,
        }
    }
}

impl<const LANES: usize> Default for BatchDecoder<LANES> {
    fn default() -> Self {
        Self::new(BatchConfig::default()).expect("valid default configuration")
    }
}

impl<const LANES: usize> BatchDecoder<LANES> {
    /// Allocates batch working storage after validating the configuration.
    pub fn new(config: BatchConfig<LANES>) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self {
            config: config.decoder,
            gamma_estimates: config.gamma_estimates,
            llr_q: vec![[0i16; LANES]; N]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            c2v: vec![[[0i16; LANES]; 8]; M]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            converged_llr: vec![[0i16; N]; LANES]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            converged: [false; LANES],
        })
    }

    /// Decodes one input per lane, retaining working storage for subsequent calls.
    pub fn decode(&mut self, channel_llrs: &[[f32; N]; LANES]) -> [DecodeResult; LANES] {
        let gamma_estimates = self
            .gamma_estimates
            .or_else(|| self.config.gamma_estimate.map(|g| [g; LANES]));
        let offset = self.config.offset.max(0);
        let llr_q = &mut *self.llr_q;
        let c2v = &mut *self.c2v;
        let converged_llr = &mut *self.converged_llr;
        let converged = &mut self.converged;

        let mut var = 0usize;
        while var < N {
            let mut lane = 0usize;
            while lane < LANES {
                llr_q[var][lane] = quantize(
                    -channel_llrs[lane][var],
                    self.config.llr_scale,
                    self.config.max_channel_value,
                );
                lane += 1;
            }
            var += 1;
        }

        let mut lowest_unsatisfied = [M; LANES];
        let mut stalled_iterations = [0usize; LANES];
        let mut final_unsatisfied = [M; LANES];
        let mut converged_unsatisfied = [M; LANES];
        let mut lane_iterations = [0usize; LANES];
        let mut iterations_run = 0usize;
        let mut offset_lanes = [offset; LANES];
        *converged = [false; LANES];

        let default_bounds = self.config.max_belief_value == DEFAULT_MAX_BELIEF_VALUE
            && self.config.max_message_value == DEFAULT_MAX_MESSAGE_VALUE;

        if self.config.max_iterations > 0 {
            // Iteration 0: first-iter dispatch (no c2v reads, all lanes active)
            iterations_run = 1;

            if default_bounds {
                #[cfg(target_arch = "aarch64")]
                if LANES == 8 {
                    let offset_vec = unsafe { vdupq_n_s16(offset) };
                    let mut i = 0usize;
                    while i < DEGREE_6_CHECKS.len() {
                        unsafe {
                            batch_neon_update_check_default_6_8_first_iter(
                                DEGREE_6_CHECKS[i],
                                offset_vec,
                                &mut *(c2v as *mut _ as *mut [[[i16; 8]; 8]; M]),
                                &mut *(llr_q as *mut _ as *mut [[i16; 8]; N]),
                            );
                        }
                        i += 1;
                    }
                    let mut i = 0usize;
                    while i < DEGREE_7_CHECKS.len() {
                        unsafe {
                            batch_neon_update_check_default_7_8_first_iter(
                                DEGREE_7_CHECKS[i],
                                offset_vec,
                                &mut *(c2v as *mut _ as *mut [[[i16; 8]; 8]; M]),
                                &mut *(llr_q as *mut _ as *mut [[i16; 8]; N]),
                            );
                        }
                        i += 1;
                    }
                    let llr_q_8 = unsafe { &*(llr_q as *const _ as *const [[i16; 8]; N]) };
                    let counts = unsafe { batch_count_unsatisfied_checks_neon_8(llr_q_8) };
                    let mut lane = 0usize;
                    while lane < LANES {
                        final_unsatisfied[lane] = counts[lane] as usize;
                        lane += 1;
                    }
                }

                #[cfg(target_arch = "aarch64")]
                if LANES != 8 {
                    let mut i = 0usize;
                    while i < DEGREE_6_CHECKS.len() {
                        batch_scalar_update_check_default_first_iter::<LANES, 6>(
                            DEGREE_6_CHECKS[i],
                            offset,
                            c2v,
                            llr_q,
                            false,
                        );
                        i += 1;
                    }
                    let mut i = 0usize;
                    while i < DEGREE_7_CHECKS.len() {
                        batch_scalar_update_check_default_first_iter::<LANES, 7>(
                            DEGREE_7_CHECKS[i],
                            offset,
                            c2v,
                            llr_q,
                            true,
                        );
                        i += 1;
                    }
                    final_unsatisfied = batch_count_unsatisfied_checks_scalar::<LANES>(llr_q);
                }

                #[cfg(not(target_arch = "aarch64"))]
                {
                    let mut i = 0usize;
                    while i < DEGREE_6_CHECKS.len() {
                        batch_scalar_update_check_default_first_iter::<LANES, 6>(
                            DEGREE_6_CHECKS[i],
                            offset,
                            c2v,
                            llr_q,
                            false,
                        );
                        i += 1;
                    }
                    let mut i = 0usize;
                    while i < DEGREE_7_CHECKS.len() {
                        batch_scalar_update_check_default_first_iter::<LANES, 7>(
                            DEGREE_7_CHECKS[i],
                            offset,
                            c2v,
                            llr_q,
                            true,
                        );
                        i += 1;
                    }
                    final_unsatisfied = batch_count_unsatisfied_checks_scalar::<LANES>(llr_q);
                }
            } else {
                let mut i = 0usize;
                while i < DEGREE_6_CHECKS.len() {
                    batch_scalar_update_check_runtime_first_iter::<LANES, 6>(
                        DEGREE_6_CHECKS[i],
                        offset,
                        self.config.max_message_value,
                        self.config.max_belief_value,
                        c2v,
                        llr_q,
                        false,
                    );
                    i += 1;
                }
                let mut i = 0usize;
                while i < DEGREE_7_CHECKS.len() {
                    batch_scalar_update_check_runtime_first_iter::<LANES, 7>(
                        DEGREE_7_CHECKS[i],
                        offset,
                        self.config.max_message_value,
                        self.config.max_belief_value,
                        c2v,
                        llr_q,
                        true,
                    );
                    i += 1;
                }
                final_unsatisfied = batch_count_unsatisfied_checks_scalar::<LANES>(llr_q);
            }

            let mut all_done = true;
            let mut lane = 0usize;
            while lane < LANES {
                if final_unsatisfied[lane] < lowest_unsatisfied[lane] {
                    lowest_unsatisfied[lane] = final_unsatisfied[lane];
                    stalled_iterations[lane] = 0;
                } else {
                    stalled_iterations[lane] += 1;
                }

                if final_unsatisfied[lane] <= self.config.max_unsatisfied_checks {
                    let mut v = 0usize;
                    while v < N {
                        converged_llr[lane][v] = llr_q[v][lane];
                        v += 1;
                    }
                    converged[lane] = true;
                    converged_unsatisfied[lane] = final_unsatisfied[lane];
                    offset_lanes[lane] = i16::MAX;
                    lane_iterations[lane] = 1;
                } else {
                    all_done = false;
                }
                lane += 1;
            }

            if !all_done {
                // Iterations 1+: use offset quenching instead of masked kernels
                let mut iteration = 1usize;
                while iteration < self.config.max_iterations {
                    iterations_run = iteration + 1;

                    if default_bounds {
                        #[cfg(target_arch = "aarch64")]
                        if LANES == 8 {
                            let offset_vec = unsafe { vld1q_s16(offset_lanes.as_ptr()) };
                            let mut i = 0usize;
                            while i < DEGREE_6_CHECKS.len() {
                                unsafe {
                                    batch_neon_update_check_default_6_8_all_active(
                                        DEGREE_6_CHECKS[i],
                                        offset_vec,
                                        &mut *(c2v as *mut _ as *mut [[[i16; 8]; 8]; M]),
                                        &mut *(llr_q as *mut _ as *mut [[i16; 8]; N]),
                                    );
                                }
                                i += 1;
                            }
                            let mut i = 0usize;
                            while i < DEGREE_7_CHECKS.len() {
                                unsafe {
                                    batch_neon_update_check_default_7_8_all_active(
                                        DEGREE_7_CHECKS[i],
                                        offset_vec,
                                        &mut *(c2v as *mut _ as *mut [[[i16; 8]; 8]; M]),
                                        &mut *(llr_q as *mut _ as *mut [[i16; 8]; N]),
                                    );
                                }
                                i += 1;
                            }
                            let llr_q_8 = unsafe { &*(llr_q as *const _ as *const [[i16; 8]; N]) };
                            let counts = unsafe { batch_count_unsatisfied_checks_neon_8(llr_q_8) };
                            let mut lane = 0usize;
                            while lane < LANES {
                                final_unsatisfied[lane] = counts[lane] as usize;
                                lane += 1;
                            }
                        }

                        #[cfg(target_arch = "aarch64")]
                        if LANES != 8 {
                            batch_scalar_iteration_default::<LANES>(
                                offset,
                                c2v,
                                llr_q,
                                &offset_lanes,
                            );
                            final_unsatisfied =
                                batch_count_unsatisfied_checks_scalar::<LANES>(llr_q);
                        }

                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            batch_scalar_iteration_default::<LANES>(
                                offset,
                                c2v,
                                llr_q,
                                &offset_lanes,
                            );
                            final_unsatisfied =
                                batch_count_unsatisfied_checks_scalar::<LANES>(llr_q);
                        }
                    } else {
                        batch_scalar_iteration_runtime::<LANES>(
                            offset,
                            self.config.max_message_value,
                            self.config.max_belief_value,
                            c2v,
                            llr_q,
                            &offset_lanes,
                        );
                        final_unsatisfied = batch_count_unsatisfied_checks_scalar::<LANES>(llr_q);
                    }

                    let mut all_done = true;
                    let mut lane = 0usize;
                    while lane < LANES {
                        if converged[lane] {
                            lane += 1;
                            continue;
                        }

                        if final_unsatisfied[lane] < lowest_unsatisfied[lane] {
                            lowest_unsatisfied[lane] = final_unsatisfied[lane];
                            stalled_iterations[lane] = 0;
                        } else {
                            stalled_iterations[lane] += 1;
                        }

                        let lane_converged =
                            final_unsatisfied[lane] <= self.config.max_unsatisfied_checks;
                        let stall_stop = stalled_iterations[lane] >= self.config.stall_limit
                            && iteration + 1 >= MIN_ITERATIONS_BEFORE_STALL_STOP;
                        if lane_converged || stall_stop {
                            let mut v = 0usize;
                            while v < N {
                                converged_llr[lane][v] = llr_q[v][lane];
                                v += 1;
                            }
                            converged[lane] = true;
                            converged_unsatisfied[lane] = final_unsatisfied[lane];
                            offset_lanes[lane] = i16::MAX;
                            lane_iterations[lane] = iteration + 1;
                        } else {
                            all_done = false;
                        }
                        lane += 1;
                    }

                    if all_done {
                        break;
                    }
                    iteration += 1;
                }
            }
        }

        std::array::from_fn(|lane| {
            let lane_iters = if lane_iterations[lane] == 0 {
                iterations_run
            } else {
                lane_iterations[lane]
            };
            // Use snapshot for converged lanes, live llr_q for non-converged
            let (codeword, refined_llrs, lane_unsatisfied) = if converged[lane] {
                (
                    std::array::from_fn(|i| converged_llr[lane][i] >= 0),
                    std::array::from_fn(|i| {
                        refined_llr(converged_llr[lane][i], self.config.llr_scale)
                    }),
                    converged_unsatisfied[lane],
                )
            } else {
                (
                    std::array::from_fn(|i| llr_q[i][lane] >= 0),
                    std::array::from_fn(|i| refined_llr(llr_q[i][lane], self.config.llr_scale)),
                    final_unsatisfied[lane],
                )
            };
            let initial_decisions: [bool; N] =
                std::array::from_fn(|i| channel_llrs[lane][i] <= 0.0);
            let hard_error_count = codeword
                .iter()
                .zip(initial_decisions.iter())
                .filter(|(a, b)| a != b)
                .count();

            let gamma_est = gamma_estimates
                .map(|gammas| gammas[lane])
                .unwrap_or_else(|| mean_magnitude_gamma(&channel_llrs[lane]));
            let normalized_distance = if gamma_est < 1e-9 {
                0.0
            } else {
                let normalizer = 4.0 * gamma_est;
                codeword
                    .iter()
                    .zip(initial_decisions.iter())
                    .zip(channel_llrs[lane].iter())
                    .map(|((final_bit, initial_bit), llr)| {
                        if final_bit != initial_bit {
                            llr.abs() / normalizer
                        } else {
                            0.0
                        }
                    })
                    .sum()
            };

            DecodeResult {
                codeword,
                success: lane_unsatisfied <= self.config.max_unsatisfied_checks,
                iterations: lane_iters,
                normalized_distance,
                hard_error_count,
                unsatisfied_checks: lane_unsatisfied,
                refined_llrs,
            }
        })
    }
}

#[inline(always)]
fn dispatch_update_check_default_6(
    check_idx: usize,
    offset: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    arch_update_check_default_6(check_idx, offset, c2v, llr_q);
}

#[inline(always)]
fn dispatch_update_check_default_7(
    check_idx: usize,
    offset: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    arch_update_check_default_7(check_idx, offset, c2v, llr_q);
}

#[inline(always)]
fn dispatch_update_check_runtime_6(
    check_idx: usize,
    offset: i16,
    max_message_value: i16,
    max_belief_value: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    arch_update_check_runtime_6(
        check_idx,
        offset,
        max_message_value,
        max_belief_value,
        c2v,
        llr_q,
    );
}

#[inline(always)]
fn dispatch_update_check_runtime_7(
    check_idx: usize,
    offset: i16,
    max_message_value: i16,
    max_belief_value: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    arch_update_check_runtime_7(
        check_idx,
        offset,
        max_message_value,
        max_belief_value,
        c2v,
        llr_q,
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn arch_update_check_default_6(
    check_idx: usize,
    offset: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    scalar_update_check_default_6(check_idx, offset, c2v, llr_q);
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn arch_update_check_default_6(
    check_idx: usize,
    offset: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    scalar_update_check_default_6(check_idx, offset, c2v, llr_q);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn arch_update_check_default_7(
    check_idx: usize,
    offset: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    scalar_update_check_default_7(check_idx, offset, c2v, llr_q);
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn arch_update_check_default_7(
    check_idx: usize,
    offset: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    scalar_update_check_default_7(check_idx, offset, c2v, llr_q);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn arch_update_check_runtime_6(
    check_idx: usize,
    offset: i16,
    max_message_value: i16,
    max_belief_value: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    scalar_update_check_runtime_6(
        check_idx,
        offset,
        max_message_value,
        max_belief_value,
        c2v,
        llr_q,
    );
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn arch_update_check_runtime_6(
    check_idx: usize,
    offset: i16,
    max_message_value: i16,
    max_belief_value: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    scalar_update_check_runtime_6(
        check_idx,
        offset,
        max_message_value,
        max_belief_value,
        c2v,
        llr_q,
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn arch_update_check_runtime_7(
    check_idx: usize,
    offset: i16,
    max_message_value: i16,
    max_belief_value: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    scalar_update_check_runtime_7(
        check_idx,
        offset,
        max_message_value,
        max_belief_value,
        c2v,
        llr_q,
    );
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn arch_update_check_runtime_7(
    check_idx: usize,
    offset: i16,
    max_message_value: i16,
    max_belief_value: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    scalar_update_check_runtime_7(
        check_idx,
        offset,
        max_message_value,
        max_belief_value,
        c2v,
        llr_q,
    );
}

#[inline(always)]
fn scalar_update_check_default_6(
    check_idx: usize,
    offset: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    let edge_idx = unsafe { INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx) };
    let mut extrinsic = [0i16; 6];
    let mut min1 = i16::MAX;
    let mut min2 = i16::MAX;
    let mut min_pos = 0usize;
    let mut sign_parity = 0u16;

    let mut pos = 0usize;
    while pos < 6 {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;
        let msg_pos = (packed & 0xFF) as usize;
        let old_msg = unsafe { *c2v.get_unchecked(var).get_unchecked(msg_pos) };
        let ex = sat_sub_255(unsafe { *llr_q.get_unchecked(var) }, old_msg);
        extrinsic[pos] = ex;
        sign_parity ^= (ex as u16) & 0x8000;

        let mag = abs_small_i16(ex);
        if mag < min1 {
            min2 = min1;
            min1 = mag;
            min_pos = pos;
        } else if mag < min2 {
            min2 = mag;
        }
        pos += 1;
    }

    pos = 0;
    while pos < 6 {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;
        let msg_pos = (packed & 0xFF) as usize;
        let base_mag = if pos == min_pos { min2 } else { min1 };
        let mag = bounded_mag_63(base_mag, offset);
        let new_negative = (sign_parity ^ ((extrinsic[pos] as u16) & 0x8000)) != 0;
        let new_msg = apply_sign(mag, new_negative);
        unsafe {
            *c2v.get_unchecked_mut(var).get_unchecked_mut(msg_pos) = new_msg;
            *llr_q.get_unchecked_mut(var) = sat_add_255(extrinsic[pos], new_msg);
        }
        pos += 1;
    }
}

#[inline(always)]
fn scalar_update_check_default_7(
    check_idx: usize,
    offset: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    let edge_idx = unsafe { INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx) };
    let mut extrinsic = [0i16; 7];
    let mut min1 = i16::MAX;
    let mut min2 = i16::MAX;
    let mut min_pos = 0usize;
    let mut sign_parity = 0u16;

    let mut pos = 0usize;
    while pos < 7 {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;
        let msg_pos = (packed & 0xFF) as usize;
        let old_msg = unsafe { *c2v.get_unchecked(var).get_unchecked(msg_pos) };
        let ex = sat_sub_255(unsafe { *llr_q.get_unchecked(var) }, old_msg);

        extrinsic[pos] = ex;
        sign_parity ^= (ex as u16) & 0x8000;

        let mag = abs_small_i16(ex);
        if mag < min1 {
            min2 = min1;
            min1 = mag;
            min_pos = pos;
        } else if mag < min2 {
            min2 = mag;
        }

        pos += 1;
    }

    pos = 0;
    while pos < 7 {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;
        let msg_pos = (packed & 0xFF) as usize;

        let base_mag = if pos == min_pos { min2 } else { min1 };
        let mag = bounded_mag_63(base_mag, offset);
        let new_negative = (sign_parity ^ ((extrinsic[pos] as u16) & 0x8000) ^ 0x8000) != 0;
        let new_msg = apply_sign(mag, new_negative);

        unsafe {
            *c2v.get_unchecked_mut(var).get_unchecked_mut(msg_pos) = new_msg;
            *llr_q.get_unchecked_mut(var) = sat_add_255(extrinsic[pos], new_msg);
        }

        pos += 1;
    }
}

#[inline(always)]
fn scalar_update_check_runtime_6(
    check_idx: usize,
    offset: i16,
    max_message_value: i16,
    max_belief_value: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    let edge_idx = unsafe { INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx) };
    let mut extrinsic = [0i16; 6];
    let mut min1 = i16::MAX;
    let mut min2 = i16::MAX;
    let mut min_pos = 0usize;
    let mut sign_parity = 0u16;

    let mut pos = 0usize;
    while pos < 6 {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;
        let msg_pos = (packed & 0xFF) as usize;
        let old_msg = unsafe { *c2v.get_unchecked(var).get_unchecked(msg_pos) };
        let ex = sat_sub(
            unsafe { *llr_q.get_unchecked(var) },
            old_msg,
            max_belief_value,
        );
        extrinsic[pos] = ex;
        sign_parity ^= (ex as u16) & 0x8000;

        let mag = abs_small_i16(ex);
        if mag < min1 {
            min2 = min1;
            min1 = mag;
            min_pos = pos;
        } else if mag < min2 {
            min2 = mag;
        }
        pos += 1;
    }

    pos = 0;
    while pos < 6 {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;
        let msg_pos = (packed & 0xFF) as usize;
        let base_mag = if pos == min_pos { min2 } else { min1 };
        let mag = bounded_mag_runtime(base_mag, offset, max_message_value);
        let new_negative = (sign_parity ^ ((extrinsic[pos] as u16) & 0x8000)) != 0;
        let new_msg = apply_sign(mag, new_negative);
        unsafe {
            *c2v.get_unchecked_mut(var).get_unchecked_mut(msg_pos) = new_msg;
            *llr_q.get_unchecked_mut(var) = sat_add(extrinsic[pos], new_msg, max_belief_value);
        }
        pos += 1;
    }
}

#[inline(always)]
fn scalar_update_check_runtime_7(
    check_idx: usize,
    offset: i16,
    max_message_value: i16,
    max_belief_value: i16,
    c2v: &mut [[i16; 3]; N],
    llr_q: &mut [i16; N],
) {
    let edge_idx = unsafe { INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx) };
    let mut extrinsic = [0i16; 7];
    let mut sign_parity = 0u16;
    let mut min1 = i16::MAX;
    let mut min2 = i16::MAX;
    let mut min_pos = 0usize;

    let mut pos = 0usize;
    while pos < 7 {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;
        let msg_pos = (packed & 0xFF) as usize;
        let old_msg = unsafe { *c2v.get_unchecked(var).get_unchecked(msg_pos) };
        let ex = sat_sub(
            unsafe { *llr_q.get_unchecked(var) },
            old_msg,
            max_belief_value,
        );

        extrinsic[pos] = ex;
        sign_parity ^= (ex as u16) & 0x8000;

        let mag = abs_small_i16(ex);
        if mag < min1 {
            min2 = min1;
            min1 = mag;
            min_pos = pos;
        } else if mag < min2 {
            min2 = mag;
        }

        pos += 1;
    }

    pos = 0;
    while pos < 7 {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;
        let msg_pos = (packed & 0xFF) as usize;

        let base_mag = if pos == min_pos { min2 } else { min1 };
        let mag = bounded_mag_runtime(base_mag, offset, max_message_value);
        let new_negative = (sign_parity ^ ((extrinsic[pos] as u16) & 0x8000) ^ 0x8000) != 0;
        let new_msg = apply_sign(mag, new_negative);

        unsafe {
            *c2v.get_unchecked_mut(var).get_unchecked_mut(msg_pos) = new_msg;
            *llr_q.get_unchecked_mut(var) = sat_add(extrinsic[pos], new_msg, max_belief_value);
        }

        pos += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn batch_neon_update_check_default_6_8_all_active(
    check_idx: usize,
    offset_vec: int16x8_t,
    c2v: &mut [[[i16; 8]; 8]; M],
    llr_q: &mut [[i16; 8]; N],
) {
    let edge_idx = INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx);
    let sign_mask = vdupq_n_u16(0x8000);
    let max_belief = vdupq_n_s16(DEFAULT_MAX_BELIEF_VALUE);
    let min_belief = vdupq_n_s16(-DEFAULT_MAX_BELIEF_VALUE);
    let max_msg = vdupq_n_s16(DEFAULT_MAX_MESSAGE_VALUE);

    let mut extrinsic = [vdupq_n_s16(0); 6];
    let mut ex_signs = [vdupq_n_u16(0); 6];
    let mut vars = [0usize; 6];
    let mut sign_parity = vdupq_n_u16(0);
    let mut min1 = vdupq_n_s16(i16::MAX);
    let mut min2 = vdupq_n_s16(i16::MAX);
    let mut min_pos = vdupq_n_u16(0);

    let mut pos = 0usize;
    while pos < 6 {
        let packed = *edge_idx.get_unchecked(pos);
        let var = (packed >> 8) as usize;
        vars[pos] = var;

        let llr_vec = vld1q_s16(llr_q[var].as_ptr());
        let old_vec = vld1q_s16(c2v[check_idx][pos].as_ptr());
        let ex_vec = vqsubq_s16(llr_vec, old_vec);
        extrinsic[pos] = ex_vec;
        let ex_sign = vandq_u16(vreinterpretq_u16_s16(ex_vec), sign_mask);
        ex_signs[pos] = ex_sign;
        sign_parity = veorq_u16(sign_parity, ex_sign);

        let mag = vabsq_s16(ex_vec);
        let lt1 = vcltq_s16(mag, min1);
        let pos_vec = vdupq_n_u16(pos as u16);
        let prev_min1 = min1;
        min1 = vbslq_s16(lt1, mag, min1);
        min_pos = vbslq_u16(lt1, pos_vec, min_pos);
        min2 = vbslq_s16(lt1, prev_min1, min2);
        let lt2 = vandq_u16(vmvnq_u16(lt1), vcltq_s16(mag, min2));
        min2 = vbslq_s16(lt2, mag, min2);
        pos += 1;
    }

    pos = 0;
    while pos < 6 {
        let var = vars[pos];
        let ex_vec = extrinsic[pos];
        let ex_sign = ex_signs[pos];
        let use_min2 = vceqq_u16(min_pos, vdupq_n_u16(pos as u16));
        let base_mag = vbslq_s16(use_min2, min2, min1);
        let bounded = vminq_s16(
            vreinterpretq_s16_u16(vqsubq_u16(
                vreinterpretq_u16_s16(base_mag),
                vreinterpretq_u16_s16(offset_vec),
            )),
            max_msg,
        );
        let neg_sign = veorq_u16(sign_parity, ex_sign);
        let sign_fill = vshrq_n_s16(vreinterpretq_s16_u16(neg_sign), 15);
        let new_msg = vsubq_s16(veorq_s16(bounded, sign_fill), sign_fill);
        let new_llr = vmaxq_s16(
            vminq_s16(vaddq_s16(ex_vec, new_msg), max_belief),
            min_belief,
        );
        vst1q_s16(c2v[check_idx][pos].as_mut_ptr(), new_msg);
        vst1q_s16(llr_q[var].as_mut_ptr(), new_llr);
        pos += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn batch_neon_update_check_default_7_8_all_active(
    check_idx: usize,
    offset_vec: int16x8_t,
    c2v: &mut [[[i16; 8]; 8]; M],
    llr_q: &mut [[i16; 8]; N],
) {
    let edge_idx = INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx);
    let sign_mask = vdupq_n_u16(0x8000);
    let max_belief = vdupq_n_s16(DEFAULT_MAX_BELIEF_VALUE);
    let min_belief = vdupq_n_s16(-DEFAULT_MAX_BELIEF_VALUE);
    let max_msg = vdupq_n_s16(DEFAULT_MAX_MESSAGE_VALUE);
    let parity_flip = vdupq_n_u16(0x8000);
    let mut extrinsic = [vdupq_n_s16(0); 7];
    let mut ex_signs = [vdupq_n_u16(0); 7];
    let mut vars = [0usize; 7];
    let mut sign_parity = vdupq_n_u16(0);
    let mut min1 = vdupq_n_s16(i16::MAX);
    let mut min2 = vdupq_n_s16(i16::MAX);
    let mut min_pos = vdupq_n_u16(0);

    let mut pos = 0usize;
    while pos < 7 {
        let packed = *edge_idx.get_unchecked(pos);
        let var = (packed >> 8) as usize;
        vars[pos] = var;

        let llr_vec = vld1q_s16(llr_q[var].as_ptr());
        let old_vec = vld1q_s16(c2v[check_idx][pos].as_ptr());
        let ex_vec = vqsubq_s16(llr_vec, old_vec);
        extrinsic[pos] = ex_vec;
        let ex_sign = vandq_u16(vreinterpretq_u16_s16(ex_vec), sign_mask);
        ex_signs[pos] = ex_sign;
        sign_parity = veorq_u16(sign_parity, ex_sign);

        let mag = vabsq_s16(ex_vec);
        let lt1 = vcltq_s16(mag, min1);
        let pos_vec = vdupq_n_u16(pos as u16);
        let prev_min1 = min1;
        min1 = vbslq_s16(lt1, mag, min1);
        min_pos = vbslq_u16(lt1, pos_vec, min_pos);
        min2 = vbslq_s16(lt1, prev_min1, min2);
        let lt2 = vandq_u16(vmvnq_u16(lt1), vcltq_s16(mag, min2));
        min2 = vbslq_s16(lt2, mag, min2);
        pos += 1;
    }

    pos = 0;
    while pos < 7 {
        let var = vars[pos];
        let ex_vec = extrinsic[pos];
        let ex_sign = ex_signs[pos];
        let use_min2 = vceqq_u16(min_pos, vdupq_n_u16(pos as u16));
        let base_mag = vbslq_s16(use_min2, min2, min1);
        let bounded = vminq_s16(
            vreinterpretq_s16_u16(vqsubq_u16(
                vreinterpretq_u16_s16(base_mag),
                vreinterpretq_u16_s16(offset_vec),
            )),
            max_msg,
        );
        let neg_sign = veorq_u16(veorq_u16(sign_parity, ex_sign), parity_flip);
        let sign_fill = vshrq_n_s16(vreinterpretq_s16_u16(neg_sign), 15);
        let new_msg = vsubq_s16(veorq_s16(bounded, sign_fill), sign_fill);
        let new_llr = vmaxq_s16(
            vminq_s16(vaddq_s16(ex_vec, new_msg), max_belief),
            min_belief,
        );
        vst1q_s16(c2v[check_idx][pos].as_mut_ptr(), new_msg);
        vst1q_s16(llr_q[var].as_mut_ptr(), new_llr);
        pos += 1;
    }
}

/// First-iteration NEON degree-6 kernel: c2v is zero, so extrinsic = llr_q.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn batch_neon_update_check_default_6_8_first_iter(
    check_idx: usize,
    offset_vec: int16x8_t,
    c2v: &mut [[[i16; 8]; 8]; M],
    llr_q: &mut [[i16; 8]; N],
) {
    let edge_idx = INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx);
    let sign_mask = vdupq_n_u16(0x8000);
    let max_belief = vdupq_n_s16(DEFAULT_MAX_BELIEF_VALUE);
    let min_belief = vdupq_n_s16(-DEFAULT_MAX_BELIEF_VALUE);
    let max_msg = vdupq_n_s16(DEFAULT_MAX_MESSAGE_VALUE);
    let mut extrinsic = [vdupq_n_s16(0); 6];
    let mut ex_signs = [vdupq_n_u16(0); 6];
    let mut vars = [0usize; 6];
    let mut sign_parity = vdupq_n_u16(0);
    let mut min1 = vdupq_n_s16(i16::MAX);
    let mut min2 = vdupq_n_s16(i16::MAX);
    let mut min_pos = vdupq_n_u16(0);

    // Phase 1: extrinsic = llr_q (no c2v subtraction, no clamp needed — quantized to ±127)
    let mut pos = 0usize;
    while pos < 6 {
        let packed = *edge_idx.get_unchecked(pos);
        let var = (packed >> 8) as usize;
        vars[pos] = var;

        let llr_vec = vld1q_s16(llr_q[var].as_ptr());
        let ex_vec = llr_vec;
        extrinsic[pos] = ex_vec;
        let ex_sign = vandq_u16(vreinterpretq_u16_s16(ex_vec), sign_mask);
        ex_signs[pos] = ex_sign;
        sign_parity = veorq_u16(sign_parity, ex_sign);

        let mag = vabsq_s16(ex_vec);
        let lt1 = vcltq_s16(mag, min1);
        let pos_vec = vdupq_n_u16(pos as u16);
        let prev_min1 = min1;
        min1 = vbslq_s16(lt1, mag, min1);
        min_pos = vbslq_u16(lt1, pos_vec, min_pos);
        min2 = vbslq_s16(lt1, prev_min1, min2);
        let lt2 = vandq_u16(vmvnq_u16(lt1), vcltq_s16(mag, min2));
        min2 = vbslq_s16(lt2, mag, min2);
        pos += 1;
    }

    pos = 0;
    while pos < 6 {
        let var = vars[pos];
        let ex_vec = extrinsic[pos];
        let ex_sign = ex_signs[pos];
        let use_min2 = vceqq_u16(min_pos, vdupq_n_u16(pos as u16));
        let base_mag = vbslq_s16(use_min2, min2, min1);
        let bounded = vminq_s16(
            vreinterpretq_s16_u16(vqsubq_u16(
                vreinterpretq_u16_s16(base_mag),
                vreinterpretq_u16_s16(offset_vec),
            )),
            max_msg,
        );
        let neg_sign = veorq_u16(sign_parity, ex_sign);
        let sign_fill = vshrq_n_s16(vreinterpretq_s16_u16(neg_sign), 15);
        let new_msg = vsubq_s16(veorq_s16(bounded, sign_fill), sign_fill);
        let new_llr = vmaxq_s16(
            vminq_s16(vaddq_s16(ex_vec, new_msg), max_belief),
            min_belief,
        );
        vst1q_s16(c2v[check_idx][pos].as_mut_ptr(), new_msg);
        vst1q_s16(llr_q[var].as_mut_ptr(), new_llr);
        pos += 1;
    }
}

/// First-iteration NEON degree-7 kernel: c2v is zero, so extrinsic = llr_q.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn batch_neon_update_check_default_7_8_first_iter(
    check_idx: usize,
    offset_vec: int16x8_t,
    c2v: &mut [[[i16; 8]; 8]; M],
    llr_q: &mut [[i16; 8]; N],
) {
    let edge_idx = INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx);
    let sign_mask = vdupq_n_u16(0x8000);
    let max_belief = vdupq_n_s16(DEFAULT_MAX_BELIEF_VALUE);
    let min_belief = vdupq_n_s16(-DEFAULT_MAX_BELIEF_VALUE);
    let max_msg = vdupq_n_s16(DEFAULT_MAX_MESSAGE_VALUE);
    let parity_flip = vdupq_n_u16(0x8000);
    let mut extrinsic = [vdupq_n_s16(0); 7];
    let mut ex_signs = [vdupq_n_u16(0); 7];
    let mut vars = [0usize; 7];
    let mut sign_parity = vdupq_n_u16(0);
    let mut min1 = vdupq_n_s16(i16::MAX);
    let mut min2 = vdupq_n_s16(i16::MAX);
    let mut min_pos = vdupq_n_u16(0);

    // Phase 1: extrinsic = llr_q (no c2v subtraction, no clamp needed — quantized to ±127)
    let mut pos = 0usize;
    while pos < 7 {
        let packed = *edge_idx.get_unchecked(pos);
        let var = (packed >> 8) as usize;
        vars[pos] = var;

        let llr_vec = vld1q_s16(llr_q[var].as_ptr());
        let ex_vec = llr_vec;
        extrinsic[pos] = ex_vec;
        let ex_sign = vandq_u16(vreinterpretq_u16_s16(ex_vec), sign_mask);
        ex_signs[pos] = ex_sign;
        sign_parity = veorq_u16(sign_parity, ex_sign);

        let mag = vabsq_s16(ex_vec);
        let lt1 = vcltq_s16(mag, min1);
        let pos_vec = vdupq_n_u16(pos as u16);
        let prev_min1 = min1;
        min1 = vbslq_s16(lt1, mag, min1);
        min_pos = vbslq_u16(lt1, pos_vec, min_pos);
        min2 = vbslq_s16(lt1, prev_min1, min2);
        let lt2 = vandq_u16(vmvnq_u16(lt1), vcltq_s16(mag, min2));
        min2 = vbslq_s16(lt2, mag, min2);
        pos += 1;
    }

    pos = 0;
    while pos < 7 {
        let var = vars[pos];
        let ex_vec = extrinsic[pos];
        let ex_sign = ex_signs[pos];
        let use_min2 = vceqq_u16(min_pos, vdupq_n_u16(pos as u16));
        let base_mag = vbslq_s16(use_min2, min2, min1);
        let bounded = vminq_s16(
            vreinterpretq_s16_u16(vqsubq_u16(
                vreinterpretq_u16_s16(base_mag),
                vreinterpretq_u16_s16(offset_vec),
            )),
            max_msg,
        );
        let neg_sign = veorq_u16(veorq_u16(sign_parity, ex_sign), parity_flip);
        let sign_fill = vshrq_n_s16(vreinterpretq_s16_u16(neg_sign), 15);
        let new_msg = vsubq_s16(veorq_s16(bounded, sign_fill), sign_fill);
        let new_llr = vmaxq_s16(
            vminq_s16(vaddq_s16(ex_vec, new_msg), max_belief),
            min_belief,
        );
        vst1q_s16(c2v[check_idx][pos].as_mut_ptr(), new_msg);
        vst1q_s16(llr_q[var].as_mut_ptr(), new_llr);
        pos += 1;
    }
}

#[inline(always)]
fn batch_scalar_update_check_default<const LANES: usize, const DEG: usize>(
    check_idx: usize,
    offset: i16,
    c2v: &mut [[[i16; LANES]; 8]; M],
    llr_q: &mut [[i16; LANES]; N],
    active: &[bool; LANES],
    parity_flip: bool,
) {
    let edge_idx = unsafe { INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx) };
    let mut extrinsic = [[0i16; LANES]; DEG];
    let mut min1 = [i16::MAX; LANES];
    let mut min2 = [i16::MAX; LANES];
    let mut min_pos = [0usize; LANES];
    let mut sign_parity = [0u16; LANES];

    let mut pos = 0usize;
    while pos < DEG {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;

        let mut lane = 0usize;
        while lane < LANES {
            if !active[lane] {
                lane += 1;
                continue;
            }
            let old_msg = c2v[check_idx][pos][lane];
            let ex = sat_sub_255(llr_q[var][lane], old_msg);
            extrinsic[pos][lane] = ex;
            sign_parity[lane] ^= (ex as u16) & 0x8000;

            let mag = abs_small_i16(ex);
            if mag < min1[lane] {
                min2[lane] = min1[lane];
                min1[lane] = mag;
                min_pos[lane] = pos;
            } else if mag < min2[lane] {
                min2[lane] = mag;
            }

            lane += 1;
        }

        pos += 1;
    }

    pos = 0;
    while pos < DEG {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;

        let mut lane = 0usize;
        while lane < LANES {
            if !active[lane] {
                lane += 1;
                continue;
            }
            let base_mag = if pos == min_pos[lane] {
                min2[lane]
            } else {
                min1[lane]
            };
            let mag = bounded_mag_63(base_mag, offset);
            let mut new_negative =
                (sign_parity[lane] ^ ((extrinsic[pos][lane] as u16) & 0x8000)) != 0;
            if parity_flip {
                new_negative = !new_negative;
            }
            let new_msg = apply_sign(mag, new_negative);

            c2v[check_idx][pos][lane] = new_msg;
            llr_q[var][lane] = sat_add_255(extrinsic[pos][lane], new_msg);
            lane += 1;
        }

        pos += 1;
    }
}

#[inline(always)]
fn batch_scalar_update_check_runtime<const LANES: usize, const DEG: usize>(
    check_idx: usize,
    offset: i16,
    max_message_value: i16,
    max_belief_value: i16,
    c2v: &mut [[[i16; LANES]; 8]; M],
    llr_q: &mut [[i16; LANES]; N],
    active: &[bool; LANES],
    parity_flip: bool,
) {
    let edge_idx = unsafe { INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx) };
    let mut extrinsic = [[0i16; LANES]; DEG];
    let mut min1 = [i16::MAX; LANES];
    let mut min2 = [i16::MAX; LANES];
    let mut min_pos = [0usize; LANES];
    let mut sign_parity = [0u16; LANES];

    let mut pos = 0usize;
    while pos < DEG {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;

        let mut lane = 0usize;
        while lane < LANES {
            if !active[lane] {
                lane += 1;
                continue;
            }
            let old_msg = c2v[check_idx][pos][lane];
            let ex = sat_sub(llr_q[var][lane], old_msg, max_belief_value);
            extrinsic[pos][lane] = ex;
            sign_parity[lane] ^= (ex as u16) & 0x8000;

            let mag = abs_small_i16(ex);
            if mag < min1[lane] {
                min2[lane] = min1[lane];
                min1[lane] = mag;
                min_pos[lane] = pos;
            } else if mag < min2[lane] {
                min2[lane] = mag;
            }

            lane += 1;
        }

        pos += 1;
    }

    pos = 0;
    while pos < DEG {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;

        let mut lane = 0usize;
        while lane < LANES {
            if !active[lane] {
                lane += 1;
                continue;
            }
            let base_mag = if pos == min_pos[lane] {
                min2[lane]
            } else {
                min1[lane]
            };
            let mag = bounded_mag_runtime(base_mag, offset, max_message_value);
            let mut new_negative =
                (sign_parity[lane] ^ ((extrinsic[pos][lane] as u16) & 0x8000)) != 0;
            if parity_flip {
                new_negative = !new_negative;
            }
            let new_msg = apply_sign(mag, new_negative);

            c2v[check_idx][pos][lane] = new_msg;
            llr_q[var][lane] = sat_add(extrinsic[pos][lane], new_msg, max_belief_value);
            lane += 1;
        }

        pos += 1;
    }
}

/// First-iteration scalar batch kernel: c2v is zero, so extrinsic = clamp(llr_q).
#[inline(always)]
fn batch_scalar_update_check_default_first_iter<const LANES: usize, const DEG: usize>(
    check_idx: usize,
    offset: i16,
    c2v: &mut [[[i16; LANES]; 8]; M],
    llr_q: &mut [[i16; LANES]; N],
    parity_flip: bool,
) {
    let edge_idx = unsafe { INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx) };
    let mut extrinsic = [[0i16; LANES]; DEG];
    let mut min1 = [i16::MAX; LANES];
    let mut min2 = [i16::MAX; LANES];
    let mut min_pos = [0usize; LANES];
    let mut sign_parity = [0u16; LANES];

    let mut pos = 0usize;
    while pos < DEG {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;

        let mut lane = 0usize;
        while lane < LANES {
            // No c2v subtraction: extrinsic = clamp(llr_q)
            let ex = sat_sub_255(llr_q[var][lane], 0);
            extrinsic[pos][lane] = ex;
            sign_parity[lane] ^= (ex as u16) & 0x8000;

            let mag = abs_small_i16(ex);
            if mag < min1[lane] {
                min2[lane] = min1[lane];
                min1[lane] = mag;
                min_pos[lane] = pos;
            } else if mag < min2[lane] {
                min2[lane] = mag;
            }

            lane += 1;
        }

        pos += 1;
    }

    pos = 0;
    while pos < DEG {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;

        let mut lane = 0usize;
        while lane < LANES {
            let base_mag = if pos == min_pos[lane] {
                min2[lane]
            } else {
                min1[lane]
            };
            let mag = bounded_mag_63(base_mag, offset);
            let mut new_negative =
                (sign_parity[lane] ^ ((extrinsic[pos][lane] as u16) & 0x8000)) != 0;
            if parity_flip {
                new_negative = !new_negative;
            }
            let new_msg = apply_sign(mag, new_negative);

            c2v[check_idx][pos][lane] = new_msg;
            llr_q[var][lane] = sat_add_255(extrinsic[pos][lane], new_msg);
            lane += 1;
        }

        pos += 1;
    }
}

/// First-iteration scalar batch kernel with runtime bounds: c2v is zero.
#[inline(always)]
fn batch_scalar_update_check_runtime_first_iter<const LANES: usize, const DEG: usize>(
    check_idx: usize,
    offset: i16,
    max_message_value: i16,
    max_belief_value: i16,
    c2v: &mut [[[i16; LANES]; 8]; M],
    llr_q: &mut [[i16; LANES]; N],
    parity_flip: bool,
) {
    let edge_idx = unsafe { INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx) };
    let mut extrinsic = [[0i16; LANES]; DEG];
    let mut min1 = [i16::MAX; LANES];
    let mut min2 = [i16::MAX; LANES];
    let mut min_pos = [0usize; LANES];
    let mut sign_parity = [0u16; LANES];

    let mut pos = 0usize;
    while pos < DEG {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;

        let mut lane = 0usize;
        while lane < LANES {
            // No c2v subtraction: extrinsic = clamp(llr_q)
            let ex = sat_sub(llr_q[var][lane], 0, max_belief_value);
            extrinsic[pos][lane] = ex;
            sign_parity[lane] ^= (ex as u16) & 0x8000;

            let mag = abs_small_i16(ex);
            if mag < min1[lane] {
                min2[lane] = min1[lane];
                min1[lane] = mag;
                min_pos[lane] = pos;
            } else if mag < min2[lane] {
                min2[lane] = mag;
            }

            lane += 1;
        }

        pos += 1;
    }

    pos = 0;
    while pos < DEG {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;

        let mut lane = 0usize;
        while lane < LANES {
            let base_mag = if pos == min_pos[lane] {
                min2[lane]
            } else {
                min1[lane]
            };
            let mag = bounded_mag_runtime(base_mag, offset, max_message_value);
            let mut new_negative =
                (sign_parity[lane] ^ ((extrinsic[pos][lane] as u16) & 0x8000)) != 0;
            if parity_flip {
                new_negative = !new_negative;
            }
            let new_msg = apply_sign(mag, new_negative);

            c2v[check_idx][pos][lane] = new_msg;
            llr_q[var][lane] = sat_add(extrinsic[pos][lane], new_msg, max_belief_value);
            lane += 1;
        }

        pos += 1;
    }
}

#[inline(always)]
fn batch_scalar_iteration_default<const LANES: usize>(
    offset: i16,
    c2v: &mut [[[i16; LANES]; 8]; M],
    llr_q: &mut [[i16; LANES]; N],
    offset_lanes: &[i16; LANES],
) {
    let active: [bool; LANES] = std::array::from_fn(|lane| offset_lanes[lane] != i16::MAX);
    let mut i = 0usize;
    while i < DEGREE_6_CHECKS.len() {
        batch_scalar_update_check_default::<LANES, 6>(
            DEGREE_6_CHECKS[i],
            offset,
            c2v,
            llr_q,
            &active,
            false,
        );
        i += 1;
    }
    let mut i = 0usize;
    while i < DEGREE_7_CHECKS.len() {
        batch_scalar_update_check_default::<LANES, 7>(
            DEGREE_7_CHECKS[i],
            offset,
            c2v,
            llr_q,
            &active,
            true,
        );
        i += 1;
    }
}

#[inline(always)]
fn batch_scalar_iteration_runtime<const LANES: usize>(
    offset: i16,
    max_message_value: i16,
    max_belief_value: i16,
    c2v: &mut [[[i16; LANES]; 8]; M],
    llr_q: &mut [[i16; LANES]; N],
    offset_lanes: &[i16; LANES],
) {
    let active: [bool; LANES] = std::array::from_fn(|lane| offset_lanes[lane] != i16::MAX);
    let mut i = 0usize;
    while i < DEGREE_6_CHECKS.len() {
        batch_scalar_update_check_runtime::<LANES, 6>(
            DEGREE_6_CHECKS[i],
            offset,
            max_message_value,
            max_belief_value,
            c2v,
            llr_q,
            &active,
            false,
        );
        i += 1;
    }
    let mut i = 0usize;
    while i < DEGREE_7_CHECKS.len() {
        batch_scalar_update_check_runtime::<LANES, 7>(
            DEGREE_7_CHECKS[i],
            offset,
            max_message_value,
            max_belief_value,
            c2v,
            llr_q,
            &active,
            true,
        );
        i += 1;
    }
}

#[inline]
fn batch_count_unsatisfied_checks_scalar<const LANES: usize>(
    llr_q: &[[i16; LANES]; N],
) -> [usize; LANES] {
    let mut count = [0usize; LANES];

    let mut i = 0usize;
    while i < DEGREE_6_CHECKS.len() {
        batch_count_unsatisfied_check::<LANES, 6>(DEGREE_6_CHECKS[i], llr_q, &mut count);
        i += 1;
    }

    let mut i = 0usize;
    while i < DEGREE_7_CHECKS.len() {
        batch_count_unsatisfied_check::<LANES, 7>(DEGREE_7_CHECKS[i], llr_q, &mut count);
        i += 1;
    }

    count
}

#[inline(always)]
/// Converts a belief in the kernels' sign (non-negative is bit 1) to an LLR in the public sign.
fn refined_llr(belief: i16, scale: f32) -> f32 {
    -i32::from(belief) as f32 / scale
}

fn quantize(value: f32, scale: f32, limit: i16) -> i16 {
    let scaled = (value * scale).round();
    scaled.clamp(-(limit as f32), limit as f32) as i16
}

#[inline(always)]
fn sat_add(a: i16, b: i16, limit: i16) -> i16 {
    (a as i32 + b as i32).clamp(-(limit as i32), limit as i32) as i16
}

#[inline(always)]
fn sat_sub(a: i16, b: i16, limit: i16) -> i16 {
    (a as i32 - b as i32).clamp(-(limit as i32), limit as i32) as i16
}

#[inline(always)]
fn sat_add_255(a: i16, b: i16) -> i16 {
    let sum = a as i32 + b as i32;
    if sum > DEFAULT_MAX_BELIEF_VALUE as i32 {
        DEFAULT_MAX_BELIEF_VALUE
    } else if sum < -(DEFAULT_MAX_BELIEF_VALUE as i32) {
        -DEFAULT_MAX_BELIEF_VALUE
    } else {
        sum as i16
    }
}

#[inline(always)]
fn sat_sub_255(a: i16, b: i16) -> i16 {
    let diff = a as i32 - b as i32;
    if diff > DEFAULT_MAX_BELIEF_VALUE as i32 {
        DEFAULT_MAX_BELIEF_VALUE
    } else if diff < -(DEFAULT_MAX_BELIEF_VALUE as i32) {
        -DEFAULT_MAX_BELIEF_VALUE
    } else {
        diff as i16
    }
}

#[inline(always)]
fn bounded_mag_63(x: i16, offset: i16) -> i16 {
    if x <= offset {
        0
    } else if x - offset > DEFAULT_MAX_MESSAGE_VALUE {
        DEFAULT_MAX_MESSAGE_VALUE
    } else {
        x - offset
    }
}

#[inline(always)]
fn bounded_mag_runtime(x: i16, offset: i16, max_message_value: i16) -> i16 {
    if x <= offset {
        0
    } else {
        let mag = x - offset;
        if mag > max_message_value {
            max_message_value
        } else {
            mag
        }
    }
}

#[inline(always)]
fn apply_sign(mag: i16, negative: bool) -> i16 {
    if negative {
        -mag
    } else {
        mag
    }
}

#[inline]
fn count_unsatisfied_checks(llr_q: &[i16; N]) -> usize {
    let mut count = 0usize;
    let mut i = 0usize;
    while i < DEGREE_6_CHECKS.len() {
        count += count_unsatisfied_check_6(DEGREE_6_CHECKS[i], llr_q);
        i += 1;
    }
    let mut i = 0usize;
    while i < DEGREE_7_CHECKS.len() {
        count += count_unsatisfied_check_7(DEGREE_7_CHECKS[i], llr_q);
        i += 1;
    }
    count
}

#[inline(always)]
fn count_unsatisfied_check_6(check_idx: usize, llr_q: &[i16; N]) -> usize {
    let edge_idx = unsafe { INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx) };
    let mut parity = false;
    let mut pos = 0usize;
    while pos < 6 {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;
        parity ^= unsafe { *llr_q.get_unchecked(var) >= 0 };
        pos += 1;
    }
    parity as usize
}

#[inline(always)]
fn count_unsatisfied_check_7(check_idx: usize, llr_q: &[i16; N]) -> usize {
    let edge_idx = unsafe { INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx) };
    let mut parity = false;
    let mut pos = 0usize;
    while pos < 7 {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;
        parity ^= unsafe { *llr_q.get_unchecked(var) >= 0 };
        pos += 1;
    }
    parity as usize
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn batch_count_unsatisfied_checks_neon_8(llr_q: &[[i16; 8]; N]) -> [u16; 8] {
    let one = vdupq_n_u16(1);
    let mut count = vdupq_n_u16(0);

    let mut i = 0usize;
    while i < DEGREE_6_CHECKS.len() {
        let edge_idx = INCOMING_MSG_INDICES_PACKED.get_unchecked(DEGREE_6_CHECKS[i]);
        let var0 = (*edge_idx.get_unchecked(0) >> 8) as usize;
        let var1 = (*edge_idx.get_unchecked(1) >> 8) as usize;
        let var2 = (*edge_idx.get_unchecked(2) >> 8) as usize;
        let var3 = (*edge_idx.get_unchecked(3) >> 8) as usize;
        let var4 = (*edge_idx.get_unchecked(4) >> 8) as usize;
        let var5 = (*edge_idx.get_unchecked(5) >> 8) as usize;
        let parity = veorq_u16(
            veorq_u16(
                vreinterpretq_u16_s16(vld1q_s16(llr_q[var0].as_ptr())),
                vreinterpretq_u16_s16(vld1q_s16(llr_q[var1].as_ptr())),
            ),
            veorq_u16(
                veorq_u16(
                    vreinterpretq_u16_s16(vld1q_s16(llr_q[var2].as_ptr())),
                    vreinterpretq_u16_s16(vld1q_s16(llr_q[var3].as_ptr())),
                ),
                veorq_u16(
                    vreinterpretq_u16_s16(vld1q_s16(llr_q[var4].as_ptr())),
                    vreinterpretq_u16_s16(vld1q_s16(llr_q[var5].as_ptr())),
                ),
            ),
        );
        let parity_bits = vandq_u16(vshrq_n_u16(parity, 15), one);
        count = vaddq_u16(count, parity_bits);
        i += 1;
    }

    let mut i = 0usize;
    while i < DEGREE_7_CHECKS.len() {
        let edge_idx = INCOMING_MSG_INDICES_PACKED.get_unchecked(DEGREE_7_CHECKS[i]);
        let var0 = (*edge_idx.get_unchecked(0) >> 8) as usize;
        let var1 = (*edge_idx.get_unchecked(1) >> 8) as usize;
        let var2 = (*edge_idx.get_unchecked(2) >> 8) as usize;
        let var3 = (*edge_idx.get_unchecked(3) >> 8) as usize;
        let var4 = (*edge_idx.get_unchecked(4) >> 8) as usize;
        let var5 = (*edge_idx.get_unchecked(5) >> 8) as usize;
        let var6 = (*edge_idx.get_unchecked(6) >> 8) as usize;
        let parity = veorq_u16(
            veorq_u16(
                veorq_u16(
                    vreinterpretq_u16_s16(vld1q_s16(llr_q[var0].as_ptr())),
                    vreinterpretq_u16_s16(vld1q_s16(llr_q[var1].as_ptr())),
                ),
                veorq_u16(
                    vreinterpretq_u16_s16(vld1q_s16(llr_q[var2].as_ptr())),
                    vreinterpretq_u16_s16(vld1q_s16(llr_q[var3].as_ptr())),
                ),
            ),
            veorq_u16(
                veorq_u16(
                    vreinterpretq_u16_s16(vld1q_s16(llr_q[var4].as_ptr())),
                    vreinterpretq_u16_s16(vld1q_s16(llr_q[var5].as_ptr())),
                ),
                vreinterpretq_u16_s16(vld1q_s16(llr_q[var6].as_ptr())),
            ),
        );
        let parity_bits = vandq_u16(vmvnq_u16(vshrq_n_u16(parity, 15)), one);
        count = vaddq_u16(count, parity_bits);
        i += 1;
    }

    let mut out = [0u16; 8];
    vst1q_u16(out.as_mut_ptr(), count);
    out
}

#[inline(always)]
fn batch_count_unsatisfied_check<const LANES: usize, const DEG: usize>(
    check_idx: usize,
    llr_q: &[[i16; LANES]; N],
    count: &mut [usize; LANES],
) {
    let edge_idx = unsafe { INCOMING_MSG_INDICES_PACKED.get_unchecked(check_idx) };
    let mut parity = [false; LANES];
    let mut pos = 0usize;
    while pos < DEG {
        let packed = unsafe { *edge_idx.get_unchecked(pos) };
        let var = (packed >> 8) as usize;
        let mut lane = 0usize;
        while lane < LANES {
            parity[lane] ^= llr_q[var][lane] >= 0;
            lane += 1;
        }
        pos += 1;
    }

    let mut lane = 0usize;
    while lane < LANES {
        count[lane] += parity[lane] as usize;
        lane += 1;
    }
}

#[inline(always)]
fn abs_small_i16(x: i16) -> i16 {
    let sign = x >> 15;
    (x ^ sign) - sign
}

/// Default gamma estimate: the mean absolute LLR divided by 4.
fn mean_magnitude_gamma(llrs: &[f32; N]) -> f32 {
    llrs.iter().map(|llr| llr.abs()).sum::<f32>() / (N as f32 * 4.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ft8_ldpc_core::crc::verify_codeword_crc14;
    use ft8_ldpc_core::fec::encode_174_91;

    #[test]
    fn configuration_validates_only_bounds_used_by_clamps() {
        let mut config = Config::default();
        config.max_channel_value = -1;
        assert_eq!(
            config.validate(),
            Err(ConfigError::NegativeChannelBound(-1))
        );
        assert!(Decoder::new(config).is_err());
        assert!(BatchDecoder::<8>::new(BatchConfig {
            decoder: config,
            ..BatchConfig::default()
        })
        .is_err());

        config.max_channel_value = 0;
        config.max_belief_value = -1;
        assert_eq!(config.validate(), Err(ConfigError::NegativeBeliefBound(-1)));
        config.max_iterations = 0;
        assert_eq!(config.validate(), Ok(()));
        assert_eq!(
            Decoder::new(config).unwrap().decode(&[0.0; N]).iterations,
            0
        );

        config.llr_scale = 0.0;
        config.offset = -1;
        config.max_message_value = -1;
        config.gamma_estimate = Some(-1.0);
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn zero_lane_batches_do_not_validate_unused_bounds() {
        for decoder in [
            Config {
                max_channel_value: -1,
                ..Config::default()
            },
            Config {
                max_belief_value: -1,
                ..Config::default()
            },
        ] {
            assert!(decoder.validate().is_err());
            assert!(Decoder::new(decoder).is_err());
            assert!(BatchConfig::<1> {
                decoder,
                ..BatchConfig::default()
            }
            .validate()
            .is_err());

            let config = BatchConfig::<0> {
                decoder,
                ..BatchConfig::default()
            };
            assert_eq!(config.validate(), Ok(()));
            let mut batch = BatchDecoder::new(config).unwrap();
            assert!(batch.decode(&[]).is_empty());
            assert!(batch.decode(&[]).is_empty());
        }
    }

    #[test]
    fn batch_gamma_overrides_shared_normalizer() {
        let inputs: [[f32; N]; 4] = std::array::from_fn(|lane| {
            std::array::from_fn(|i| ((i * 17 + lane * 11) % 31) as f32 - 15.0)
        });
        let gamma_estimates = [0.0, 0.25, 1.0, 4.0];
        let config = Config {
            gamma_estimate: Some(7.0),
            ..Config::default()
        };
        let mut batch = BatchDecoder::new(BatchConfig {
            decoder: config,
            gamma_estimates: Some(gamma_estimates),
        })
        .unwrap();
        let results = batch.decode(&inputs);
        for lane in 0..4 {
            let scalar = Decoder::new(Config {
                gamma_estimate: Some(gamma_estimates[lane]),
                ..config
            })
            .unwrap()
            .decode(&inputs[lane]);
            assert_eq!(
                results[lane].normalized_distance.to_bits(),
                scalar.normalized_distance.to_bits()
            );
        }
        let mut batch = BatchDecoder::new(BatchConfig {
            decoder: config,
            gamma_estimates: None,
        })
        .unwrap();
        let results = batch.decode(&inputs);
        for lane in 0..4 {
            assert_eq!(
                results[lane].normalized_distance.to_bits(),
                Decoder::new(config)
                    .unwrap()
                    .decode(&inputs[lane])
                    .normalized_distance
                    .to_bits()
            );
        }
    }

    #[test]
    fn clean_codeword_decodes_with_zero_syndrome() {
        let decoder = Decoder::default();
        let message = [false; 77];
        let codeword = encode_174_91(&message);
        let llrs = std::array::from_fn(|i| if codeword[i] { -6.0 } else { 6.0 });

        let result = decoder.decode(&llrs);

        assert!(result.success);
        assert_eq!(result.unsatisfied_checks, 0);
        assert_eq!(result.codeword, codeword);
        assert!(verify_codeword_crc14(&result.codeword));
    }

    #[test]
    fn moderate_noise_still_returns_bounded_output() {
        let decoder = Decoder::new(Config {
            max_iterations: 10,
            ..Config::default()
        })
        .unwrap();
        let message = [true; 77];
        let codeword = encode_174_91(&message);
        let llrs = std::array::from_fn(|i| {
            let magnitude = 0.8 + ((i * 37) % 19) as f32 * 0.07;
            let sign = if codeword[i] { -1.0 } else { 1.0 };
            let flip = matches!(i, 3 | 41 | 97 | 122);
            if flip {
                -sign * magnitude
            } else {
                sign * magnitude
            }
        });

        let result = decoder.decode(&llrs);

        assert!(result.iterations > 0);
        assert!(result.unsatisfied_checks <= M);
        assert!(result.normalized_distance.is_finite());
        assert!(result.refined_llrs.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn batch_decoder_matches_scalar_per_lane() {
        let decoder = Decoder::new(Config {
            max_iterations: 6,
            ..Config::default()
        })
        .unwrap();
        let mut batch_decoder = BatchDecoder::<4>::new(BatchConfig {
            decoder: Config {
                max_iterations: 6,
                ..Config::default()
            },
            ..BatchConfig::default()
        })
        .unwrap();

        let lanes = std::array::from_fn(|lane| {
            std::array::from_fn(|i| {
                let base = ((i * 17 + lane * 29) % 41) as f32 - 20.0;
                base * 0.35
            })
        });

        let batch = batch_decoder.decode(&lanes);

        let mut lane = 0usize;
        while lane < 4 {
            let scalar = decoder.decode(&lanes[lane]);
            assert_eq!(batch[lane].codeword, scalar.codeword);
            assert_eq!(batch[lane].success, scalar.success);
            assert_eq!(batch[lane].iterations, scalar.iterations);
            assert_eq!(batch[lane].unsatisfied_checks, scalar.unsatisfied_checks);
            assert_eq!(batch[lane].hard_error_count, scalar.hard_error_count);
            lane += 1;
        }
    }

    #[test]
    fn batch_decoder_8_matches_scalar_per_lane() {
        let decoder = Decoder::new(Config {
            max_iterations: 6,
            ..Config::default()
        })
        .unwrap();
        let mut batch_decoder = BatchDecoder::<8>::new(BatchConfig {
            decoder: Config {
                max_iterations: 6,
                ..Config::default()
            },
            ..BatchConfig::default()
        })
        .unwrap();

        let lanes = std::array::from_fn(|lane| {
            std::array::from_fn(|i| {
                let base = ((i * 13 + lane * 31) % 53) as f32 - 26.0;
                base * 0.27
            })
        });

        let batch = batch_decoder.decode(&lanes);

        let mut lane = 0usize;
        while lane < 8 {
            let scalar = decoder.decode(&lanes[lane]);
            assert_eq!(batch[lane].codeword, scalar.codeword);
            assert_eq!(batch[lane].success, scalar.success);
            assert_eq!(batch[lane].iterations, scalar.iterations);
            assert_eq!(batch[lane].unsatisfied_checks, scalar.unsatisfied_checks);
            assert_eq!(batch[lane].hard_error_count, scalar.hard_error_count);
            lane += 1;
        }
    }

    #[test]
    fn default_gamma_is_the_mean_magnitude_over_four() {
        let decoder = Decoder::default();
        let llrs: [f32; N] = std::array::from_fn(|i| {
            if i % 3 == 0 {
                -2.5
            } else {
                3.5 + (i % 5) as f32 * 0.25
            }
        });
        let gamma = llrs.iter().map(|llr| llr.abs()).sum::<f32>() / (N as f32 * 4.0);

        let default = decoder.decode(&llrs);
        let explicit = Decoder::new(Config {
            gamma_estimate: Some(gamma),
            ..Config::default()
        })
        .unwrap()
        .decode(&llrs);

        assert_eq!(
            default.normalized_distance.to_bits(),
            explicit.normalized_distance.to_bits()
        );
    }
}
