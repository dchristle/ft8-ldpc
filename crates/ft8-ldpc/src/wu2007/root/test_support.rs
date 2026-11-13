//! Synthetic bases and stage runners shared by the unit tests.
use super::*;
use crate::wu2007::scoring::test_inputs::SplitMix64;

/// A decoder with a synthetic basis for testing kernels at arbitrary input widths.
pub(crate) struct Synthetic {
    pub decoder: Decoder,
    pub baseline: PackedCodeword,
    pub k: usize,
}

/// G[i] has bit i set, nothing else below k, and random bits at
/// positions k to 173 with `density`/1024 set. Δ₀ and the reliabilities are
/// random. The decoder's own k stays 91; stages are called with `k`.
pub(crate) fn synthetic(seed: u64, k: usize, density: u64, config: Config) -> Synthetic {
    let mut rng = SplitMix64(seed);
    let mut decoder = Decoder::new(config).unwrap();
    let random_bits = |rng: &mut SplitMix64, row: &mut PackedCodeword| {
        for n in k..N {
            if (rng.next() & 1023) < density {
                set_codeword_bit(row, n);
            }
        }
    };
    for n in 0..N {
        decoder.basis.sys_to_natural[n] = n as u16;
        decoder.basis.natural_to_sys[n] = n as u16;
    }
    for i in 0..k {
        let mut row = [0; 3];
        set_codeword_bit(&mut row, i);
        random_bits(&mut rng, &mut row);
        decoder.basis.standard_rows[i] = row;
        decoder.basis.encoded_basis_deltas[i] = row;
        decoder.basis.parity_basis_deltas[i] = extract_slice(&row, k, N - k);
    }
    let mut baseline = [0; 3];
    random_bits(&mut rng, &mut baseline);
    let alpha: [f32; N] = std::array::from_fn(|_| rng.unit() * 6.0 + 0.05);
    decoder.scorer.update(&alpha, [0; 3]);
    Synthetic {
        decoder,
        baseline,
        k,
    }
}

pub(in crate::wu2007::root) fn fresh_state() -> SearchState {
    SearchState {
        received: [0; 3],
        minimum: None,
        best_any: None,
        best_valid: None,
        candidates: Vec::new(),
        diagnostics: Diagnostics::default(),
        trace: Vec::new(),
    }
}

/// Everything a stage test compares: the ADMIT sequence in selection order,
/// and the result with all diagnostics and the candidate list.
#[derive(PartialEq, Eq, Debug)]
pub(crate) struct Outcome {
    pub trace: Vec<(PackedCodeword, u8, [u8; 6])>,
    pub summary: String,
    pub diagnostics: StageCounters,
}

/// Search-stage counters grouped for equality comparisons.
#[derive(PartialEq, Eq, Debug, Clone, Copy, Default)]
pub(crate) struct StageCounters {
    pub visited_steps: usize,
    pub enumerated_patterns: usize,
    pub recoded_candidates: usize,
    pub scored_candidates: usize,
    pub preprocess1_examined: usize,
    pub preprocess1_rejects: usize,
    pub preprocess2_enumerated_patterns: usize,
    pub preprocess2_scored_candidates: usize,
    pub preprocess2_probe_masks: usize,
    pub preprocess2_key_probes: usize,
    pub candidate_cap_hit: bool,
}

pub(crate) fn counters(d: &Diagnostics) -> StageCounters {
    StageCounters {
        visited_steps: d.visited_steps,
        enumerated_patterns: d.enumerated_patterns,
        recoded_candidates: d.recoded_candidates,
        scored_candidates: d.scored_candidates,
        preprocess1_examined: d.preprocess1_examined,
        preprocess1_rejects: d.preprocess1_rejects,
        preprocess2_enumerated_patterns: d.preprocess2_enumerated_patterns,
        preprocess2_scored_candidates: d.preprocess2_scored_candidates,
        preprocess2_probe_masks: d.preprocess2_probe_masks,
        preprocess2_key_probes: d.preprocess2_key_probes,
        candidate_cap_hit: d.candidate_cap_hit,
    }
}

pub(in crate::wu2007::root) fn outcome(mut state: SearchState, require_crc: bool) -> Outcome {
    let trace = std::mem::take(&mut state.trace);
    let diagnostics = counters(&state.diagnostics);
    Outcome {
        trace,
        summary: format!("{:?}", finalize(state, require_crc)),
        diagnostics,
    }
}

impl Synthetic {
    /// Run second preprocessing alone, as decode() would, from a state with
    /// the baseline already admitted `admitted_before` times.
    pub(in crate::wu2007::root) fn run_pre2(&mut self) -> SearchState {
        let mut state = fresh_state();
        self.decoder.batch.len = 0;
        let before_enumerated = state.diagnostics.enumerated_patterns;
        let before_scored = state.diagnostics.scored_candidates;
        self.decoder
            .second_preprocessing(&mut state, self.baseline, self.k);
        self.decoder.flush_batch(&mut state);
        state.diagnostics.preprocess2_enumerated_patterns =
            state.diagnostics.enumerated_patterns - before_enumerated;
        state.diagnostics.preprocess2_scored_candidates =
            state.diagnostics.scored_candidates - before_scored;
        state
    }
}

/// A configuration for stage-level tests: no first preprocessing, second
/// preprocessing on with the given w, τ and cap, and the list on.
pub(crate) fn pre2_config(w: usize, tau: Option<usize>, cap: usize) -> Config {
    let mut config = Config {
        cascade_bits: 0,
        ..Default::default()
    };
    config.search_order = w;
    config.enable_first_preprocessing = false;
    config.enable_second_preprocessing = true;
    config.second_preprocessing_prefix_len = tau;
    config.max_recoded_candidates = cap;
    config.collect_candidates = true;
    config.max_list_candidates = 64;
    config
}

/// A random payload with a valid CRC, encoded with the generator for `c`
/// cascaded bits, returned in natural codeword order.
pub(crate) fn random_codeword(rng: &mut SplitMix64, cascade_bits: usize) -> PackedCodeword {
    let payload: Vec<bool> = (0..77).map(|_| rng.next() & 1 == 1).collect();
    let crc = ft8_ldpc_core::crc::compute_crc14_from_bits(&payload);
    let mut message = [0u64; 2];
    for (i, &bit) in payload.iter().enumerate() {
        message[i / 64] |= (bit as u64) << (i % 64);
    }
    for j in 0..14 {
        let bit = (crc >> (13 - j)) & 1;
        message[(77 + j) / 64] |= (bit as u64) << ((77 + j) % 64);
    }
    assert!(verify_crc14_packed_message(message));
    let k = 91 - cascade_bits;
    let input = mask_message_words(message, active_message_mask(k));
    let generator = CascadedGeneratorMatrix::new(cascade_bits).unwrap();
    let codeword = generator.encode_packed(input);
    assert_eq!(extract_prefix(&codeword, 91), message);
    codeword
}

/// LLRs of amplitude `amplitude` whose signs follow the codeword.
pub(crate) fn llr_of(codeword: &PackedCodeword, amplitude: f32) -> [f32; N] {
    std::array::from_fn(|n| {
        if get_codeword_bit(codeword, n) {
            -amplitude
        } else {
            amplitude
        }
    })
}

/// Random-sign LLRs with magnitudes from 1/32 to 8 in increments of 1/32.
pub(crate) fn noise_llr(rng: &mut SplitMix64) -> [f32; N] {
    std::array::from_fn(|_| {
        let magnitude = (1 + (rng.next() >> 56)) as f32 / 32.;
        if rng.next() & 1 == 1 {
            -magnitude
        } else {
            magnitude
        }
    })
}

/// A codeword with `flips` hard errors at distinct random positions drawn
/// from `range` (natural indices), with the flipped LLR magnitudes halved.
pub(crate) fn perturbed_llr(
    rng: &mut SplitMix64,
    codeword: &PackedCodeword,
    flips: usize,
    range: std::ops::Range<usize>,
) -> [f32; N] {
    let mut llr: [f32; N] = std::array::from_fn(|n| {
        let magnitude = 3.1 + rng.unit() * 2.3;
        if get_codeword_bit(codeword, n) {
            -magnitude
        } else {
            magnitude
        }
    });
    let mut chosen = Vec::new();
    while chosen.len() < flips {
        let n = range.start + rng.below(range.len());
        if !chosen.contains(&n) {
            chosen.push(n);
            llr[n] = -llr[n] * 0.5;
        }
    }
    llr
}
