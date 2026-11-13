//! Second-preprocessing tests for the exhaustive oracle, box and probe ordering,
//! candidate caps, and agreement with box-free and general-step references.
use super::*;
use crate::wu2007::root::test_support::*;
use crate::wu2007::scoring::test_inputs::SplitMix64;

const HIGH_CAP: usize = 1 << 30;

fn choose(n: usize, r: usize) -> usize {
    (0..r).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

/// Parity density in 1/1024 so that a few key bits are set per row.
fn density_for(tau: usize) -> u64 {
    if tau == 0 {
        256
    } else {
        (600 / tau as u64).clamp(1, 256)
    }
}

fn rows(s: &Synthetic) -> Vec<PackedRedundancy> {
    s.decoder.basis.parity_basis_deltas[..s.k].to_vec()
}

fn masks_of(tau: usize) -> Vec<u128> {
    std::iter::once(0)
        .chain((1..=tau).map(|u| 1u128 << (tau - u)))
        .collect()
}

/// What a brute-force scan of every (step, probe, pair) finds, independently
/// of the boxes and walkers.
#[derive(Default, Debug)]
struct Analysis {
    /// Position arrays the stage must admit, as a sorted set.
    admitted: Vec<[u8; 6]>,
    steps_with_a_passing_pair: usize,
    steps_without: usize,
    largest_box: usize,
    box_with_failing_entry: bool,
    steps_with_two_one_bit_masks: usize,
    probes: usize,
}

fn analyze(s: &Synthetic, w: usize, tau: usize) -> Analysis {
    let k = s.k;
    let rows = rows(s);
    let baseline_parity = extract_slice(&s.baseline, k, N - k);
    let mut analysis = Analysis::default();
    enumerate(k, w, |step| {
        let floor = if w >= 1 { step[0] as usize } else { k };
        if floor < 2 {
            return true;
        }
        let mut parity = baseline_parity;
        for &q in &step[..w] {
            parity = xor_red(parity, rows[q as usize]);
        }
        let step_key = key_of(parity, tau);
        let (mut any, mut one_bit_masks) = (false, 0);
        for (u, mask) in masks_of(tau).into_iter().enumerate() {
            analysis.probes += 1;
            let (mut in_box, mut passing) = (0, 0);
            for i in 0..k {
                for j in i + 1..k {
                    if key_of(xor_red(rows[i], rows[j]), tau) == step_key ^ mask {
                        in_box += 1;
                        if j < floor {
                            passing += 1;
                            let mut positions = [0; 6];
                            positions[0] = i as u8;
                            positions[1] = j as u8;
                            positions[2..2 + w].copy_from_slice(&step[..w]);
                            analysis.admitted.push(positions);
                        }
                    }
                }
            }
            analysis.largest_box = analysis.largest_box.max(in_box);
            analysis.box_with_failing_entry |= in_box > passing;
            any |= passing > 0;
            one_bit_masks += usize::from(u > 0 && passing > 0);
        }
        analysis.steps_with_a_passing_pair += usize::from(any);
        analysis.steps_without += usize::from(!any);
        analysis.steps_with_two_one_bit_masks += usize::from(one_bit_masks >= 2);
        true
    });
    analysis
}

fn run(s: &mut Synthetic) -> Outcome {
    outcome(s.run_pre2(), true)
}

fn run_reference(s: &mut Synthetic, box_free: bool, general: bool) -> Outcome {
    s.decoder.reference.pre2_box_free = box_free;
    s.decoder.reference.pre2_general = general;
    let result = run(s);
    s.decoder.reference.pre2_box_free = false;
    s.decoder.reference.pre2_general = false;
    result
}

/// The stage admits exactly the TEPs whose key differs from the step key
/// in at most one bit, each once, for the configurations of the oracle.
#[test]
fn oracle_matches_brute_force_for_every_walker() {
    for (n, &(tau, w)) in [(0, 0), (1, 1), (14, 2), (16, 3), (17, 1), (83, 4)]
        .iter()
        .enumerate()
    {
        let mut s = synthetic(
            0xa11ce + n as u64,
            18,
            density_for(tau),
            pre2_config(w, Some(tau), HIGH_CAP),
        );
        let analysis = analyze(&s, w, tau);
        let outcome = run(&mut s);
        let mut got: Vec<[u8; 6]> = outcome.trace.iter().map(|t| t.2).collect();
        assert_eq!(got.len(), outcome.trace.len());
        got.sort();
        let before = got.len();
        got.dedup();
        assert_eq!(got.len(), before, "(tau {tau}, w {w}) repeats a pattern");
        let mut expected = analysis.admitted.clone();
        expected.sort();
        assert!(!expected.is_empty(), "(tau {tau}, w {w}) admits nothing");
        assert_eq!(got, expected, "(tau {tau}, w {w})");
        assert!(outcome.trace.iter().all(|t| t.1 as usize == w + 2));
        assert!(!outcome.diagnostics.candidate_cap_hit);
    }
}

/// With every parity bit zero all pairs share key 0, and the box lists
/// them in lexicographic order, in both structures.
#[test]
fn box_of_equal_keys_is_lexicographic_in_both_structures() {
    let all: Vec<[u8; 2]> = (0..91)
        .flat_map(|i| (i + 1..91).map(move |j| [i as u8, j as u8]))
        .collect();
    for tau in [0, 17] {
        let mut boxes = PairWorkspace::new(tau);
        boxes.build(&[[0; 2]; 91], 91);
        let (first, end) = boxes.range(0);
        let listed: Vec<[u8; 2]> = (first..end).map(|e| boxes.entry(e)).collect();
        assert_eq!(listed, all, "tau {tau}");
        assert_eq!(tau <= BUCKET_TAU_MAX, tau == 0);
    }
}

#[test]
fn rebuilt_boxes_forget_the_previous_basis() {
    let first = [[0b001, 0], [0b001, 0], [0b010, 0], [0b100, 0]];
    let second = [[0b001, 0], [0b000, 0], [0b010, 0], [0b100, 0]];
    for tau in [3, 17] {
        let mut boxes = PairWorkspace::new(tau);
        let listed = |boxes: &PairWorkspace, key: u128| -> Vec<[u8; 2]> {
            let (first, end) = boxes.range(key);
            (first..end).map(|e| boxes.entry(e)).collect()
        };
        boxes.build(&first, 4);
        assert_eq!(listed(&boxes, 0), vec![[0, 1]], "tau {tau}");
        boxes.build(&second, 4);
        assert_eq!(listed(&boxes, 0), Vec::<[u8; 2]>::new(), "tau {tau}");
        assert_eq!(listed(&boxes, 1), vec![[0, 1]], "tau {tau}");
    }
}

/// Checks the probe list for every τ and its visible effect on admission order.
#[test]
fn probe_list_follows_the_paper_for_every_tau() {
    for tau in 0..=97 {
        let boxes = PairWorkspace::new(tau);
        assert_eq!(boxes.masks.len(), tau + 1);
        assert_eq!(boxes.masks[0], 0);
        for u in 1..=tau {
            assert_eq!(boxes.masks[u], 1u128 << (tau - u), "tau {tau}, u {u}");
        }
    }
    assert_eq!(PairWorkspace::new(2).masks, vec![0, 2, 1]);
    assert_eq!(PairWorkspace::new(3).masks, vec![0, 4, 2, 1]);
    assert_eq!(PairWorkspace::new(4).masks, vec![0, 8, 4, 2, 1]);

    let probe_example = |cap: usize| {
        let mut s = synthetic(5, 3, 0, pre2_config(0, Some(2), cap));
        s.decoder.basis.parity_basis_deltas[..3].copy_from_slice(&[
            [0b00, 0],
            [0b01, 0],
            [0b11, 0],
        ]);
        s.baseline = [0; 3];
        let state = s.run_pre2();
        (
            state.trace.iter().map(|t| t.2).collect::<Vec<_>>(),
            counters(&state.diagnostics),
        )
    };
    let (order, all) = probe_example(HIGH_CAP);
    assert_eq!(order, vec![[1, 2, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]);
    assert_eq!(all.preprocess2_key_probes, 3);
    let (order, capped) = probe_example(1);
    assert_eq!(order, vec![[1, 2, 0, 0, 0, 0]]);
    assert_eq!(capped.preprocess2_key_probes, 3);
    assert_eq!(capped.enumerated_patterns, 2);
    assert_eq!(capped.recoded_candidates, 1);
    assert!(capped.candidate_cap_hit);
}

/// The cap binds on the second counted pair.
#[test]
fn cap_boundary_counts_the_refused_pair_and_nothing_after_it() {
    for (tau, probes) in [(0, 2), (17, 19)] {
        let mut s = synthetic(6, 4, 0, pre2_config(1, Some(tau), 1));
        s.decoder.basis.parity_basis_deltas[..4].fill([0; 2]);
        s.baseline = [0; 3];
        s.decoder.scorer.update(&[0.; N], [0; 3]);
        let state = s.run_pre2();
        assert_eq!(state.trace.len(), 1);
        assert_eq!(state.trace[0].2, [0, 1, 2, 0, 0, 0]);
        let c = counters(&state.diagnostics);
        assert_eq!(
            (c.visited_steps, c.enumerated_patterns),
            (2, 2),
            "tau {tau}"
        );
        assert_eq!((c.recoded_candidates, c.scored_candidates), (1, 1));
        assert!(c.candidate_cap_hit);
        assert_eq!(c.preprocess2_key_probes, probes, "tau {tau}");
    }
}

/// Checks counter identities with and without a binding cap.
#[test]
fn counters_satisfy_the_closed_forms() {
    for (n, &(k, w, tau)) in [
        (18, 0, 0),
        (18, 1, 1),
        (20, 2, 14),
        (16, 3, 16),
        (14, 4, 20),
        (18, 1, 17),
    ]
    .iter()
    .enumerate()
    {
        let seed = 0xc0de + n as u64;
        let density = (density_for(tau) / 2).max(1);
        let mut s = synthetic(seed, k, density, pre2_config(w, Some(tau), HIGH_CAP));
        let full = counters(&s.run_pre2().diagnostics);
        assert!(!full.candidate_cap_hit);
        assert_eq!(full.preprocess2_probe_masks, tau + 1);
        assert_eq!(
            full.preprocess2_key_probes,
            (tau + 1) * choose(k - 2, w),
            "(k {k}, w {w}, tau {tau})"
        );
        assert_eq!(
            full.preprocess2_enumerated_patterns,
            full.preprocess2_scored_candidates
        );
        assert_eq!(full.recoded_candidates, full.scored_candidates);
        let admitted = full.recoded_candidates;
        assert!(admitted >= 2, "(k {k}, w {w}, tau {tau}) admits {admitted}");
        for cap in [1, admitted / 2, admitted - 1, admitted] {
            let mut s = synthetic(seed, k, density, pre2_config(w, Some(tau), cap));
            let c = counters(&s.run_pre2().diagnostics);
            assert_eq!(c.recoded_candidates, cap);
            assert_eq!(
                c.candidate_cap_hit,
                cap < admitted,
                "cap {cap} of {admitted}"
            );
            assert_eq!(
                c.preprocess2_enumerated_patterns,
                c.preprocess2_scored_candidates + usize::from(c.candidate_cap_hit)
            );
            assert_eq!(c.visited_steps, c.enumerated_patterns);
            assert!(c.preprocess2_key_probes >= 1);
            assert!(c.preprocess2_key_probes <= (tau + 1) * choose(k - 2, w));
            assert_eq!(c.preprocess2_probe_masks, tau + 1);
        }
    }
}

fn noise_decoder_input(seed: u64) -> [f32; N] {
    noise_llr(&mut SplitMix64(seed))
}

/// The lazy step and the general step give identical decodes at every
/// τ, including τ = 17 (sorted boxes with the lazy step) and τ = 65.
#[test]
fn lazy_and_general_steps_decode_identically() {
    for (n, tau) in [0, 16, 64, 65, 17].into_iter().enumerate() {
        let llr = noise_decoder_input(900 + n as u64);
        let mut config = pre2_config(3, Some(tau), 1_000_000);
        config.collect_candidates = false;
        let mut lazy = Decoder::new(config.clone()).unwrap();
        let mut general = Decoder::new(config).unwrap();
        general.reference.pre2_general = true;
        let (a, b) = (lazy.decode(&llr), general.decode(&llr));
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "tau {tau}");
        assert!(a.diagnostics.preprocess2_key_probes > 0);
    }
}

/// A weight-three error pattern is found only by second preprocessing.
#[test]
fn second_preprocessing_finds_the_pattern_no_earlier_stage_visits() {
    let mut rng = SplitMix64(0x833);
    let codeword = random_codeword(&mut rng, 0);
    let mut llr = llr_of(&codeword, 8.0);
    for n in 0..3 {
        llr[n] = -llr[n];
    }
    let config = pre2_config(1, Some(0), 300_000);
    let mut config = config;
    config.collect_candidates = false;
    let result = Decoder::new(config).unwrap().decode(&llr);
    assert_eq!(result.status, Status::Success);
    assert_eq!(result.codeword, codeword);
    assert_eq!(result.hard_error_count, 3);
    assert!(result.diagnostics.recoded_candidates > 92);
    let mut off = pre2_config(1, Some(0), 300_000);
    off.enable_second_preprocessing = false;
    off.collect_candidates = false;
    let without = Decoder::new(off).unwrap().decode(&llr);
    assert_ne!(without.codeword, codeword);
}

#[derive(Clone, Copy, Debug)]
enum Cap {
    High,
    Exact,
    One,
    MidBox,
    Half,
}

#[derive(Clone, Copy, Debug)]
struct Input {
    seed: u64,
    k: usize,
    w: usize,
    tau: usize,
    cap: Cap,
}

const fn input(seed: u64, k: usize, w: usize, tau: usize, cap: Cap) -> Input {
    Input {
        seed,
        k,
        w,
        tau,
        cap,
    }
}

fn smoke_inputs() -> Vec<Input> {
    use Cap::*;
    vec![
        input(1, 24, 0, 0, High),
        input(2, 24, 0, 1, Exact),
        input(3, 20, 0, 16, MidBox),
        input(4, 22, 0, 83, High),
        input(5, 30, 1, 1, High),
        input(6, 30, 1, 8, Exact),
        input(7, 26, 1, 17, MidBox),
        input(8, 26, 1, 20, High),
        input(9, 18, 2, 2, MidBox),
        input(10, 18, 2, 14, Exact),
        input(11, 20, 2, 16, One),
        input(12, 20, 2, 64, High),
        input(13, 16, 3, 3, MidBox),
        input(14, 16, 3, 17, Exact),
        input(15, 14, 3, 65, High),
        input(16, 14, 3, 16, Half),
        input(17, 14, 4, 4, MidBox),
        input(18, 14, 4, 20, Exact),
        input(19, 12, 4, 83, High),
        input(20, 12, 4, 0, One),
        input(21, 16, 2, 5, MidBox),
        input(22, 20, 1, 6, High),
    ]
}

fn build(input: &Input, cap: usize) -> Synthetic {
    synthetic(
        0x7f1 + input.seed,
        input.k,
        density_for(input.tau),
        pre2_config(input.w, Some(input.tau), cap),
    )
}

/// The cap that makes the refused candidate the second of a box, for a
/// stage's uncapped trace: two consecutive admissions of one step whose pairs
/// have equal keys sit in the same box.
fn mid_box_cap(
    s: &Synthetic,
    input: &Input,
    trace: &[(PackedCodeword, u8, [u8; 6])],
) -> Option<usize> {
    let rows = rows(s);
    let pair_key =
        |p: &[u8; 6]| key_of(xor_red(rows[p[0] as usize], rows[p[1] as usize]), input.tau);
    (1..trace.len()).find(|&i| {
        trace[i].2[2..] == trace[i - 1].2[2..] && pair_key(&trace[i].2) == pair_key(&trace[i - 1].2)
    })
}

/// Compares the optimized stage with the box-free reference bit for bit.
#[test]
fn optimized_stage_matches_the_box_free_reference() {
    let inputs = smoke_inputs();
    assert!((16..=24).contains(&inputs.len()));
    let (mut multi_pair_box, mut failing_entry, mut two_masks) = (0, 0, 0);
    let (mut cap_in_box, mut exact, mut cap_one, mut early) = (0, 0, 0, 0);
    let mut by_w = [0; 5];
    let mut taus = std::collections::BTreeSet::new();
    for input in &inputs {
        by_w[input.w] += 1;
        taus.insert(input.tau);
        let mut uncapped = build(input, HIGH_CAP);
        let analysis = analyze(&uncapped, input.w, input.tau);
        let free = run_reference(&mut uncapped, true, false);
        let total = free.diagnostics.recoded_candidates;
        multi_pair_box += usize::from(analysis.largest_box >= 2);
        failing_entry += usize::from(analysis.box_with_failing_entry);
        two_masks += usize::from(input.tau >= 2 && analysis.steps_with_two_one_bit_masks > 0);
        let cap = match input.cap {
            Cap::High => HIGH_CAP,
            Cap::Exact => total.max(1),
            Cap::One => 1,
            Cap::Half => (total / 2).max(1),
            Cap::MidBox => match mid_box_cap(&uncapped, input, &free.trace) {
                Some(i) => {
                    cap_in_box += 1;
                    i
                }
                None => (total / 2).max(1),
            },
        };
        let mut optimized = build(input, cap);
        let mut reference = build(input, cap);
        let (a, b) = (
            run(&mut optimized),
            run_reference(&mut reference, true, false),
        );
        assert_eq!(a, b, "{input:?}");
        assert_eq!(
            a.trace.iter().map(|t| t.2).collect::<Vec<_>>().len(),
            a.diagnostics.recoded_candidates
        );
        assert_eq!(run(&mut build(input, HIGH_CAP)), free, "{input:?}");
        exact += usize::from(matches!(input.cap, Cap::Exact) && !a.diagnostics.candidate_cap_hit);
        cap_one += usize::from(cap == 1 && a.diagnostics.candidate_cap_hit);
    }
    // Earlier stages that already used the whole cap: the stage does not run.
    for (k_cascade, w) in [(0, 1), (14, 1)] {
        let llr = noise_decoder_input(0xea71 + k_cascade as u64);
        let k = 91 - k_cascade;
        let mut config = pre2_config(w, None, 40);
        config.cascade_bits = k_cascade;
        config.collect_candidates = false;
        let mut decoder = Decoder::new(config).unwrap();
        decoder.reference.pre2_box_free = true;
        let result = decoder.decode(&llr);
        let d = &result.diagnostics;
        assert!(d.candidate_cap_hit, "k {k}");
        assert_eq!(
            (
                d.preprocess2_enumerated_patterns,
                d.preprocess2_scored_candidates,
                d.preprocess2_probe_masks,
                d.preprocess2_key_probes
            ),
            (0, 0, 0, 0)
        );
        early += 1;
    }
    // Full decodes at k = 91 and k = 77 with the default τ.
    for (n, (cascade, w)) in [(0, 0), (0, 1), (14, 0), (14, 1)].into_iter().enumerate() {
        let llr = noise_decoder_input(0xf011 + n as u64);
        let mut config = pre2_config(w, None, 300_000);
        config.cascade_bits = cascade;
        let mut optimized = Decoder::new(config.clone()).unwrap();
        let mut reference = Decoder::new(config).unwrap();
        reference.reference.pre2_box_free = true;
        let (a, b) = (optimized.decode(&llr), reference.decode(&llr));
        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "cascade {cascade}, w {w}"
        );
        let tau = resolve_tau(91 - cascade, &pre2_config(w, None, 1));
        assert_eq!(a.diagnostics.preprocess2_probe_masks, tau + 1);
    }
    assert!(by_w.iter().all(|&n| n >= 2), "{by_w:?}");
    for tau in [0, 1, 8, 16, 17, 20, 64, 65, 83] {
        assert!(taus.contains(&tau), "tau {tau} not reached");
    }
    assert!(taus.iter().any(|&t| (2..16).contains(&t)));
    assert!(two_masks >= 2, "{two_masks}");
    assert!(multi_pair_box >= 1 && failing_entry >= 1);
    assert!(cap_in_box >= 1 && exact >= 1 && cap_one >= 1 && early >= 1);
}

/// Compares the lazy step with the general-step reference.
#[test]
fn lazy_step_matches_the_general_step_reference() {
    use Cap::*;
    let inputs = [
        input(101, 16, 3, 0, High),
        input(102, 16, 3, 1, MidBox),
        input(103, 18, 3, 15, High),
        input(104, 18, 3, 16, Exact),
        input(105, 18, 3, 17, High),
        input(106, 16, 3, 17, MidBox),
        input(107, 20, 3, 40, High),
        input(108, 16, 3, 64, Half),
        input(109, 15, 3, 16, MidBox),
        input(110, 22, 3, 12, Half),
        input(111, 14, 3, 2, One),
        input(112, 20, 3, 30, Exact),
        input(113, 17, 3, 15, High),
        input(114, 17, 3, 64, High),
        input(115, 19, 3, 8, MidBox),
        input(116, 13, 3, 0, Half),
        input(117, 21, 3, 1, High),
        input(118, 18, 3, 40, MidBox),
    ];
    let (mut built, mut never_built, mut capped_inside) = (0, 0, 0);
    let mut taus = std::collections::BTreeSet::new();
    for input in &inputs {
        taus.insert(input.tau);
        let mut uncapped = build(input, HIGH_CAP);
        let analysis = analyze(&uncapped, 3, input.tau);
        built += analysis.steps_with_a_passing_pair;
        never_built += analysis.steps_without;
        let general = run_reference(&mut uncapped, false, true);
        let total = general.diagnostics.recoded_candidates;
        let cap = match input.cap {
            High => HIGH_CAP,
            Exact => total.max(1),
            One => 1,
            Half => (total / 2).max(1),
            MidBox => mid_box_cap(&uncapped, input, &general.trace).unwrap_or((total / 2).max(1)),
        };
        let mut lazy = build(input, cap);
        let mut reference = build(input, cap);
        let (a, b) = (run(&mut lazy), run_reference(&mut reference, false, true));
        assert_eq!(a, b, "{input:?}");
        capped_inside += usize::from(a.diagnostics.candidate_cap_hit);
        assert_eq!(a.diagnostics.preprocess2_key_probes > 0, true);
    }
    for tau in [0, 1, 15, 16, 17, 40, 64] {
        assert!(taus.contains(&tau), "tau {tau} not reached");
    }
    assert!(built > 0 && never_built > 0 && capped_inside > 0);

    // k = 91 with the default τ.
    let llr = noise_decoder_input(0x91);
    let mut config = pre2_config(3, None, 300_000);
    config.collect_candidates = false;
    let mut lazy = Decoder::new(config.clone()).unwrap();
    let mut general = Decoder::new(config).unwrap();
    general.reference.pre2_general = true;
    let (a, b) = (lazy.decode(&llr), general.decode(&llr));
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert_eq!(a.diagnostics.preprocess2_probe_masks, 16);
}

#[test]
fn default_tau_follows_the_table() {
    let expected = |k: usize, w: usize| -> usize {
        let r = N - k;
        match (w, k) {
            (0, _) => r,
            (1, 77..=80) => 19,
            (1, _) => 20,
            (2, 77..=80) => 16,
            (2, _) => 17,
            (3, _) => 15,
            (4, 77..=80) => 13,
            (4, _) => 14,
            _ => unreachable!(),
        }
    };
    for cascade in 0..=14 {
        for w in 0..=4 {
            let mut config = pre2_config(w, None, 1000);
            config.cascade_bits = cascade;
            let k = 91 - cascade;
            assert_eq!(resolve_tau(k, &config), expected(k, w), "(k {k}, w {w})");
        }
    }
    let mut config = pre2_config(2, Some(500), 1000);
    assert_eq!(resolve_tau(91, &config), 83);
    config.cascade_bits = 14;
    assert_eq!(resolve_tau(77, &config), 97);
}
