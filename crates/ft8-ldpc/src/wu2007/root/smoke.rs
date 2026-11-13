//! Compares optimized stages with plain references using deterministic seeded inputs
//! and assertions that verify branch coverage.
use super::test_support::*;
use super::*;
use crate::wu2007::scoring::test_inputs::SplitMix64;
use std::collections::BTreeSet;

// ---------------------------------------------------------------- inputs

/// Approximately Gaussian, from exact dyadic arithmetic only.
fn gaussian(rng: &mut SplitMix64) -> f32 {
    (0..12).map(|_| rng.unit()).sum::<f32>() - 6.0
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Clean,
    /// `count` hard errors, from natural positions `from..to`.
    Flips(usize, usize, usize),
    /// Sign reversals that keep their magnitude, so the errors rank among the
    /// reliable positions and need a pattern of that order.
    Strong(usize, usize, usize),
    Awgn(f32),
    /// Strong message bits with `count` reversed, and parity bits of almost
    /// no weight: the order-0 candidate is the cheapest and fails the CRC.
    LowParity(usize),
    PureNoise,
    AllEqual,
    Integers,
    WithZeros,
    WideRange,
}

fn make_llr(rng: &mut SplitMix64, kind: Kind, cascade: usize) -> [f32; N] {
    let codeword = random_codeword(rng, cascade);
    match kind {
        Kind::Clean => llr_of(&codeword, 4.37),
        Kind::Flips(count, from, to) => perturbed_llr(rng, &codeword, count, from..to),
        Kind::Strong(count, from, to) => {
            let mut llr = llr_of(&codeword, 1.0);
            llr = std::array::from_fn(|n| llr[n] * (3.1 + rng.unit() * 2.3));
            let mut chosen = Vec::new();
            while chosen.len() < count {
                let n = from + rng.below(to - from);
                if !chosen.contains(&n) {
                    chosen.push(n);
                    llr[n] = -llr[n];
                }
            }
            llr
        }
        Kind::LowParity(count) => {
            let mut llr: [f32; N] = std::array::from_fn(|n| {
                let amplitude = if n < 91 {
                    3.1 + rng.unit() * 2.3
                } else {
                    0.01 + rng.unit() * 0.04
                };
                if get_codeword_bit(&codeword, n) {
                    -amplitude
                } else {
                    amplitude
                }
            });
            let mut chosen = Vec::new();
            while chosen.len() < count {
                let n = rng.below(91);
                if !chosen.contains(&n) {
                    chosen.push(n);
                    llr[n] = -llr[n];
                }
            }
            llr
        }
        Kind::Awgn(sigma) => {
            let clean = llr_of(&codeword, 4.1);
            std::array::from_fn(|n| clean[n] + sigma * gaussian(rng))
        }
        Kind::PureNoise => noise_llr(rng),
        Kind::AllEqual => {
            let mut llr = llr_of(&codeword, 3.3);
            for _ in 0..2 {
                let n = rng.below(N);
                llr[n] = -llr[n];
            }
            llr
        }
        Kind::Integers => {
            let clean = llr_of(&codeword, 4.0);
            std::array::from_fn(|n| (clean[n] + (rng.below(7) as f32 - 3.)).round())
        }
        Kind::WithZeros => {
            let mut llr = llr_of(&codeword, 3.9);
            for n in 0..N {
                match rng.below(40) {
                    0 => llr[n] = 0.0,
                    1 => llr[n] = -0.0,
                    _ => llr[n] *= 0.8 + 0.4 * rng.unit(),
                }
            }
            llr
        }
        Kind::WideRange => {
            let clean = llr_of(&codeword, 1.0);
            std::array::from_fn(|n| {
                clean[n] * (0.5 + rng.unit() * 0.5) * f32::powi(2.0, rng.below(21) as i32 - 10)
            })
        }
    }
}

// ----------------------------------------------------- decode-level cells

#[derive(Clone)]
struct Cell {
    config: Config,
    llr: [f32; N],
}

fn cell_config(
    cascade: usize,
    w: usize,
    pre1: bool,
    cap: usize,
    list: Option<usize>,
    require_crc: bool,
) -> Config {
    let mut config = Config {
        cascade_bits: cascade,
        ..Default::default()
    };
    config.search_order = w;
    config.enable_first_preprocessing = pre1;
    config.max_recoded_candidates = cap;
    config.collect_candidates = list.is_some();
    config.max_list_candidates = list.unwrap_or(32);
    config.require_crc = require_crc;
    config
}

/// Decode on the optimized decoder and on one with `stages` replaced, each
/// twice (the second time after a different input), and compare every field.
fn assert_decodes_agree(cell: &Cell, stages: ReferenceStages, what: &str) -> DecodeResult {
    let mut optimized = Decoder::new(cell.config.clone()).unwrap();
    let mut reference = Decoder::new(cell.config.clone()).unwrap();
    reference.reference = stages;
    let other = cell.llr.map(|x| -x);
    let _ = (optimized.decode(&other), reference.decode(&other));
    let (a, b) = (optimized.decode(&cell.llr), reference.decode(&cell.llr));
    assert_eq!(format!("{a:?}"), format!("{b:?}"), "{what}");
    a
}

/// The order of the winning record, read from a list of length one.
fn winner_order(cell: &Cell) -> Option<u8> {
    let mut config = cell.config.clone();
    config.collect_candidates = true;
    config.max_list_candidates = 1;
    let result = Decoder::new(config).unwrap().decode(&cell.llr);
    (result.status == Status::Success)
        .then(|| result.candidates.first().map(|c| c.order))
        .flatten()
}

/// The first seeded input whose CRC-valid winner has the given order.
fn input_with_winner_order(order: u8, config: &Config, seed: u64) -> Cell {
    let mut rng = SplitMix64(seed);
    for flips in std::iter::repeat([order as usize, order as usize + 1, 1])
        .take(60)
        .flatten()
    {
        let llr = make_llr(
            &mut rng,
            Kind::Strong(flips, 0, 91 - config.cascade_bits),
            config.cascade_bits,
        );
        let cell = Cell {
            config: config.clone(),
            llr,
        };
        if winner_order(&cell) == Some(order) {
            return cell;
        }
    }
    panic!("no input with a winner of order {order}");
}

fn cell(config: Config, rng: &mut SplitMix64, kind: Kind) -> Cell {
    let llr = make_llr(rng, kind, config.cascade_bits);
    Cell { config, llr }
}

/// Compares the per-candidate path with the batch path.
#[test]
fn per_candidate_path_matches_the_batch() {
    let stages = ReferenceStages {
        dispatch: true,
        ..Default::default()
    };
    let mut rng = SplitMix64(0x7200);
    let mut cells = Vec::new();
    let default = cell_config(0, 3, true, 300_000, None, true);
    cells.push(cell(default.clone(), &mut rng, Kind::Clean));
    for order in 1..=3 {
        cells.push(input_with_winner_order(
            order,
            &default,
            0x7210 + order as u64,
        ));
    }
    cells.push(input_with_winner_order(4, &default, 0x7220));
    let flips = Kind::Flips(3, 0, 91);
    for cap in [2, 3, 4, 5, 6] {
        let config = cell_config(0, 3, true, cap, None, true);
        cells.push(cell(config, &mut rng, flips));
    }
    // A cap inside first preprocessing, a few survivors after the main search.
    cells.push(cell(
        cell_config(0, 3, true, 1 + 125_671 + 3, None, true),
        &mut rng,
        Kind::PureNoise,
    ));
    cells.push(cell(
        cell_config(0, 3, true, 1, None, true),
        &mut rng,
        flips,
    ));
    cells.push(cell(
        cell_config(0, 3, true, 0, None, true),
        &mut rng,
        flips,
    ));
    cells.push(cell(
        cell_config(0, 2, true, 300_000, Some(8), true),
        &mut rng,
        flips,
    ));
    cells.push(cell(
        cell_config(0, 2, true, 300_000, Some(32), true),
        &mut rng,
        flips,
    ));
    cells.push(cell(
        cell_config(0, 2, true, 300_000, None, false),
        &mut rng,
        Kind::PureNoise,
    ));
    cells.push(cell(
        cell_config(14, 2, true, 300_000, None, true),
        &mut rng,
        flips,
    ));
    cells.push(cell(
        cell_config(0, 1, true, 300_000, None, true),
        &mut rng,
        Kind::PureNoise,
    ));
    // The lowest-distance candidate fails the CRC and a higher one passes.
    let config = cell_config(0, 2, false, 300_000, None, true);
    let found = cell(config.clone(), &mut rng, Kind::LowParity(2));
    {
        let result = Decoder::new(config).unwrap().decode(&found.llr);
        assert_eq!(result.status, Status::Success);
        assert!(result.diagnostics.best_any_llr_distance.unwrap() < result.llr_distance);
        assert!(result.diagnostics.crc_failures > 0);
    }
    let found = Some(found);
    cells.push(found.expect("an input whose best candidate fails the CRC"));
    assert!((16..=24).contains(&cells.len()));

    let (mut early, mut cap_hits, mut no_valid, mut listed) = (0, 0, 0, 0);
    for (n, cell) in cells.iter().enumerate() {
        let result = assert_decodes_agree(cell, stages, &format!("per-candidate cell {n}"));
        let d = &result.diagnostics;
        early += usize::from(
            d.recoded_candidates == 1 && d.main_enumerated_patterns == 0 && !d.candidate_cap_hit,
        );
        cap_hits += usize::from(d.candidate_cap_hit);
        no_valid += usize::from(result.status == Status::NoValidCrc);
        listed += usize::from(!result.candidates.is_empty());
    }
    assert!(
        early >= 1 && cap_hits >= 6 && no_valid >= 1 && listed >= 2,
        "early {early}, cap hits {cap_hits}, no valid {no_valid}, listed {listed}"
    );
    let mid = Decoder::new(cells[9 + 1].config.clone())
        .unwrap()
        .decode(&cells[10].llr);
    assert!(mid.diagnostics.candidate_cap_hit);
}

// ------------------------------------------------ kernel-level machinery

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Class {
    Lane4(usize),
    Lane2(usize),
    One,
    First,
    Last,
}

/// Classifies the survivor's position within the four-, two- or one-lane blocks.
fn class_of(positions: &[u8; 6], m: usize, k: usize) -> Class {
    let start = positions[m - 2] as usize + 1;
    let length = k - start;
    let offset = positions[m - 1] as usize - start;
    if offset < 4 * (length / 4) {
        Class::Lane4(offset % 4)
    } else {
        let rest = offset - 4 * (length / 4);
        if m != 3 && length % 4 >= 2 && rest < 2 {
            Class::Lane2(rest)
        } else {
            Class::One
        }
    }
}

fn kernel_config(m: usize) -> Config {
    let mut config = cell_config(0, m - 1, true, 1 << 30, Some(64), true);
    config.preprocessing_threshold = Some(usize::MAX);
    config
}

fn kernel_run(
    s: &mut Synthetic,
    m: usize,
    t: usize,
    theta: usize,
    cap: usize,
    reference: bool,
) -> Outcome {
    let k = s.k;
    s.decoder.config.max_recoded_candidates = cap;
    let mask = low_mask(t);
    for i in 0..k {
        s.decoder.prefix_words[i] = s.decoder.basis.parity_basis_deltas[i][0] & mask;
    }
    let b = extract_slice(&s.baseline, k, N - k)[0] & mask;
    let mut state = fresh_state();
    s.decoder.batch.len = 0;
    let d = &mut s.decoder;
    match (m, reference) {
        (2, false) => d.pre1_order2_vector(&mut state, s.baseline, k, b, theta),
        (3, false) => d.pre1_order3_vector(&mut state, s.baseline, k, b, theta),
        (4, false) => d.pre1_order4_vector(&mut state, s.baseline, k, b, theta),
        (2, true) => d.pre1_reference_order2(&mut state, s.baseline, k, b, theta),
        (3, true) => d.pre1_reference_order3(&mut state, s.baseline, k, b, theta),
        (4, true) => d.pre1_reference_order4(&mut state, s.baseline, k, b, theta),
        _ => unreachable!(),
    }
    d.flush_batch(&mut state);
    outcome(state, true)
}

/// Compares an order-`m` first-preprocessing kernel with its reference using survivor-derived caps.
fn preprocessing_kernel_smoke(m: usize, ks: &[usize], seed: u64) {
    let thetas = [0, m - 1, m, m + 1, m + 5, 32, usize::MAX];
    let mut classes = BTreeSet::new();
    let (mut none, mut few, mut all, mut below, mut equal, mut beyond) = (0, 0, 0, 0, 0, 0);
    for (n, &k) in ks.iter().enumerate() {
        let mut theta = thetas[n % thetas.len()];
        let t = [0, 6, 20, 41, 64][n % 5].min(if k > 20 { 64 } else { 20 });
        let density = [12, 128, 512, 700][n % 4];
        let make = || synthetic(seed + n as u64, k, density, kernel_config(m));
        if theta == m + 5 && t > 0 {
            // Raise θ from m until a handful of patterns survive.
            if let Some(tuned) = (m..m + t + 1).find(|&candidate| {
                let found = kernel_run(&mut make(), m, t, candidate, 1 << 30, true)
                    .trace
                    .len();
                (1..=8).contains(&found)
            }) {
                theta = tuned;
            }
        }
        let patterns = choose_usize(k, m);
        let mut s = make();
        let full = kernel_run(&mut s, m, t, theta, 1 << 30, true);
        assert_eq!(
            full,
            kernel_run(&mut make(), m, t, theta, 1 << 30, false),
            "(k {k}, theta {theta}, t {t})"
        );
        let survivors: Vec<[u8; 6]> = full.trace.iter().map(|x| x.2).collect();
        none += usize::from(survivors.is_empty());
        few += usize::from((1..=8).contains(&survivors.len()));
        all += usize::from(survivors.len() == patterns);
        below += usize::from(theta < m);
        equal += usize::from(theta == m);
        let mut caps = BTreeSet::from([0, survivors.len(), survivors.len() + 1]);
        if !survivors.is_empty() {
            caps.insert(survivors.len() - 1);
        }
        let mut seen = BTreeSet::new();
        for (index, positions) in survivors.iter().enumerate() {
            let class = class_of(positions, m, k);
            if seen.insert(class) {
                caps.insert(index);
                classes.insert(class);
            }
        }
        if survivors.len() == patterns && !survivors.is_empty() {
            classes.insert(Class::First);
            classes.insert(Class::Last);
        }
        beyond += 1;
        // Large inputs are the expensive ones: a few caps only.
        let caps: Vec<usize> = if k > 20 {
            caps.into_iter().take(3).collect()
        } else {
            caps.into_iter().collect()
        };
        for cap in caps {
            let a = kernel_run(&mut make(), m, t, theta, cap, false);
            let b = kernel_run(&mut make(), m, t, theta, cap, true);
            assert_eq!(a, b, "(m {m}, k {k}, theta {theta}, t {t}, cap {cap})");
        }
    }
    assert!(
        none >= 1 && few >= 1 && all >= 1 && below >= 1 && equal >= 1 && beyond >= 1,
        "m {m}: none {none}, few {few}, all {all}, below {below}, equal {equal}"
    );
    for lane in 0..4 {
        assert!(
            classes.contains(&Class::Lane4(lane)),
            "m {m}: lane {lane} of a 4-block not reached"
        );
    }
    assert!(classes.contains(&Class::One), "m {m}: no 1-block");
    if m != 3 {
        assert!(
            classes.contains(&Class::Lane2(0)) && classes.contains(&Class::Lane2(1)),
            "m {m}: 2-block lanes"
        );
    }
    assert!(classes.contains(&Class::First) && classes.contains(&Class::Last));
}

fn choose_usize(n: usize, r: usize) -> usize {
    (0..r).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

/// Four decode-level inputs through the decoder's dispatch to the kernel.
fn preprocessing_decode_smoke(w: usize) {
    let stages = ReferenceStages {
        pre1: true,
        ..Default::default()
    };
    let mut rng = SplitMix64(0x7300 + w as u64);
    let m = w + 1;
    let mut cells = vec![
        cell(
            cell_config(0, w, true, 300_000, None, true),
            &mut rng,
            Kind::Flips(m, 0, 91),
        ),
        cell(
            cell_config(14, w, true, 300_000, None, true),
            &mut rng,
            Kind::Flips(m, 0, 77),
        ),
        cell(
            cell_config(0, w, true, 300_000, Some(8), true),
            &mut rng,
            Kind::PureNoise,
        ),
    ];
    // A cap that lands inside the stage, after its first admissions.
    let mut capped_config = cell_config(0, w, true, 300_000, None, true);
    capped_config.parity_prefix_len = Some(10);
    capped_config.preprocessing_threshold = Some(m + 6);
    let mut capped = cell(capped_config, &mut rng, Kind::PureNoise);
    let uncapped = Decoder::new(capped.config.clone())
        .unwrap()
        .decode(&capped.llr);
    let d = &uncapped.diagnostics;
    capped.config.max_recoded_candidates =
        d.recoded_candidates - d.preprocess1_scored_candidates.min(3) + 1;
    cells.push(capped);
    let mut hits = 0;
    for (n, cell) in cells.iter().enumerate() {
        let result = assert_decodes_agree(cell, stages, &format!("order {m} decode cell {n}"));
        assert!(result.diagnostics.preprocess1_examined > 0);
        hits += usize::from(result.diagnostics.candidate_cap_hit);
    }
    assert!(hits >= 1);
}

#[test]
fn order_two_kernel_matches_its_reference() {
    let ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 9, 77, 91];
    preprocessing_kernel_smoke(2, &ks, 0x7301);
    preprocessing_decode_smoke(1);
}

#[test]
fn order_three_kernel_matches_its_reference() {
    let ks = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 11, 12, 77, 91];
    preprocessing_kernel_smoke(3, &ks, 0x7401);
    preprocessing_decode_smoke(2);
}

#[test]
fn order_four_kernel_matches_its_reference() {
    let ks = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 77, 91];
    preprocessing_kernel_smoke(4, &ks, 0x7501);
    preprocessing_decode_smoke(3);
}

/// Compares dedicated main-search loops of orders 1 to 3 with the generic enumerator.
#[test]
fn main_orders_match_the_generic_enumerator() {
    let ks = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 6, 7, 77, 91];
    for m in 1..=3 {
        for (n, &k) in ks.iter().enumerate() {
            if k < m || (k > 20 && m == 3 && k == 91) && false {
                continue;
            }
            let make = || {
                let mut config = cell_config(0, m, false, 1 << 30, Some(64), true);
                config.enable_second_preprocessing = false;
                synthetic(0x7800 + (10 * m + n) as u64, k, 300, config)
            };
            let run = |cap: usize, generic: bool| {
                let mut s = make();
                s.decoder.config.max_recoded_candidates = cap;
                s.decoder.reference.main = generic;
                let mut state = fresh_state();
                s.decoder.batch.len = 0;
                s.decoder.main_order(&mut state, s.baseline, k, m);
                s.decoder.flush_batch(&mut state);
                outcome(state, true)
            };
            let patterns = choose_usize(k, m);
            let mut caps = BTreeSet::from([0, 1, patterns - 1, patterns, patterns + 1]);
            if m >= 2 {
                // The last pattern of the first row and the first of the next.
                let first_row = k - m + 1;
                caps.insert(first_row - 1);
                caps.insert(first_row);
            }
            if m == 3 {
                // The last pattern of an i1 row inside the first i0.
                caps.insert(k - 3);
                caps.insert(k - 2);
            }
            let caps: Vec<usize> = if k > 20 {
                vec![0, patterns - 1, patterns, patterns / 2]
            } else {
                caps.into_iter().collect()
            };
            for cap in caps {
                if k > 20 && m == 3 && cap == patterns {
                    continue;
                }
                assert_eq!(run(cap, false), run(cap, true), "(m {m}, k {k}, cap {cap})");
            }
        }
    }
    // Decode level: w = 1, 2, 3 with first preprocessing off.
    let stages = ReferenceStages {
        main: true,
        ..Default::default()
    };
    let mut rng = SplitMix64(0x7880);
    for w in 1..=3 {
        let c = cell(
            cell_config(0, w, false, 300_000, None, true),
            &mut rng,
            Kind::Flips(w, 0, 91),
        );
        assert_decodes_agree(&c, stages, &format!("main w {w}"));
    }
    let capped = cell(
        cell_config(14, 3, false, 5_000, Some(8), true),
        &mut rng,
        Kind::PureNoise,
    );
    let result = assert_decodes_agree(&capped, stages, "main capped");
    assert!(result.diagnostics.candidate_cap_hit);
}

fn basis_fields_equal(a: &SystematicBasis, b: &SystematicBasis, k: usize) {
    assert_eq!(a.rank, b.rank);
    assert_eq!(a.sys_to_natural, b.sys_to_natural);
    assert_eq!(a.natural_to_sys, b.natural_to_sys);
    assert_eq!(a.standard_rows[..k], b.standard_rows[..k]);
    assert_eq!(a.encoded_basis_deltas[..k], b.encoded_basis_deltas[..k]);
    assert_eq!(a.parity_basis_deltas[..k], b.parity_basis_deltas[..k]);
}

/// Compares the optimized basis build with Gauss–Jordan elimination.
#[test]
fn basis_build_matches_gauss_jordan() {
    let mut rng = SplitMix64(0x7900);
    let mut recipes: Vec<(usize, [f32; N])> = Vec::new();
    let cascades = [0, 1, 7, 13, 14];
    for (n, &c) in cascades.iter().cycle().take(20).enumerate() {
        let llr: [f32; N] = match n % 10 {
            0 | 1 => noise_llr(&mut rng),
            2 => [2.5; N],
            3 => std::array::from_fn(|_| (rng.below(9) as f32) - 4.),
            4 => std::array::from_fn(|i| {
                if i % 7 == 0 {
                    0.
                } else {
                    noise_llr(&mut rng)[i]
                }
            }),
            5 => std::array::from_fn(|i| (i + 1) as f32 / 16.),
            6 => std::array::from_fn(|i| (N - i) as f32 / 16.),
            // Reliable parity side: most greedy picks come from it.
            7 | 8 => std::array::from_fn(|i| {
                if i >= 91 {
                    5. + rng.unit()
                } else {
                    rng.unit() * 0.5
                }
            }),
            _ => std::array::from_fn(|i| {
                if i < 91 - c && i % 3 == 0 {
                    0.01
                } else {
                    1. + rng.unit()
                }
            }),
        };
        recipes.push((c, llr));
    }
    let mut scratch = BasisBuildScratch::new();
    let mut scratch_reference = BasisBuildScratch::new();
    let (mut optimized, mut reference) = (SystematicBasis::new(), SystematicBasis::new());
    let (mut saw_no_cascade, mut saw_max_cascade, mut skipped_column) = (false, false, false);
    let mut previous: Option<(usize, [f32; N])> = None;
    for (c, llr) in recipes {
        let generator = CascadedGeneratorMatrix::new(c).unwrap();
        let k = 91 - c;
        // Each build runs on objects that built a different input before.
        if let Some((pc, pllr)) = previous {
            let before = CascadedGeneratorMatrix::new(pc).unwrap();
            let _ = optimized.rebuild_from_masks(&mut scratch, before.generator(), &pllr, 91 - pc);
            let _ = reference.rebuild_reference(
                &mut scratch_reference,
                before.generator(),
                &pllr,
                91 - pc,
            );
        }
        let a = optimized.rebuild_from_masks(&mut scratch, generator.generator(), &llr, k);
        let b = reference.rebuild_reference(&mut scratch_reference, generator.generator(), &llr, k);
        assert_eq!(a, b, "c {c}");
        assert!(a.is_ok());
        basis_fields_equal(&optimized, &reference, k);
        let mut order: Vec<usize> = (0..N).collect();
        order.sort_by(|&x, &y| {
            llr[y]
                .abs()
                .partial_cmp(&llr[x].abs())
                .unwrap()
                .then(x.cmp(&y))
        });
        skipped_column |= (0..k).any(|s| optimized.sys_to_natural[s] as usize != order[s]);
        saw_no_cascade |= c == 0;
        saw_max_cascade |= c == 14;
        previous = Some((c, llr));
    }
    assert!(saw_no_cascade && saw_max_cascade && skipped_column);

    // Hand-made generators of rank below k: only the returned rank is compared.
    let k = 20;
    let llr = noise_llr(&mut rng);
    let duplicated: Vec<[u64; 2]> = (0..N).map(|n| [1u64 << (n % 12), 0]).collect();
    let zero_columns: Vec<[u64; 2]> = (0..N)
        .map(|n| {
            if n % 3 == 0 {
                [0, 0]
            } else {
                [1u64 << (n % 19), 0]
            }
        })
        .collect();
    let mut failures = 0;
    for masks in [&duplicated, &zero_columns] {
        let a = optimized.rebuild_from_masks(&mut scratch, masks, &llr, k);
        let b = reference.rebuild_reference(&mut scratch_reference, masks, &llr, k);
        assert_eq!(a, b);
        failures += usize::from(a.is_err());
    }
    assert_eq!(failures, 2);
    // At decode level the rank is the same field of the diagnostics.
    let c = cell(
        cell_config(7, 1, true, 10_000, None, true),
        &mut rng,
        Kind::Flips(2, 0, 84),
    );
    let stages = ReferenceStages {
        basis: true,
        ..Default::default()
    };
    assert_decodes_agree(&c, stages, "basis decode");
}

fn record(i: usize, distance: f32, hard_errors: u16, crc_valid: bool) -> Candidate {
    Candidate {
        codeword: [i as u64, 0, 0],
        payload: [0; 2],
        message_with_crc: [0; 2],
        llr_distance: distance,
        hard_error_count: hard_errors,
        crc_valid,
        order: (i % 7) as u8,
        positions: [i as u8; 6],
        num_positions: (i % 7) as u8,
    }
}

/// Compares binary-search list insertion with linear insertion.
#[test]
fn list_insertion_matches_straight_insertion() {
    let mut rng = SplitMix64(0x7b00);
    let (mut zero, mut larger, mut overflow, mut mixed) = (0, 0, 0, 0);
    for n in 0..20 {
        let ties = n >= 10;
        let bound = [
            0, 1, 8, 32, 50, 8, 50, 32, 1, 0, 0, 1, 8, 32, 50, 50, 32, 8, 1, 8,
        ][n];
        let length = [
            40, 300, 120, 45, 40, 200, 70, 250, 90, 49, 150, 43, 240, 100, 45, 220, 41, 300, 80, 60,
        ][n];
        let records: Vec<_> = (0..length)
            .map(|i| {
                let crc = rng.below(3) == 0;
                if ties {
                    record(i, rng.below(4) as f32, rng.below(2) as u16, crc)
                } else {
                    // Distinct distances: the index is in the low bits.
                    record(
                        i,
                        rng.below(1000) as f32 + i as f32 / 512.,
                        rng.below(3) as u16,
                        crc,
                    )
                }
            })
            .collect();
        let (mut optimized, mut reference) = (Vec::new(), Vec::new());
        for r in &records {
            list_insert(&mut optimized, r.clone(), bound);
            list_insert_reference(&mut reference, r.clone(), bound);
            assert!(optimized.len() <= bound && reference.len() <= bound);
        }
        zero += usize::from(bound == 0);
        larger += usize::from(bound > length);
        overflow += usize::from(length > bound);
        mixed += usize::from(
            records.iter().any(|r| r.crc_valid) && records.iter().any(|r| !r.crc_valid),
        );
        assert_eq!(optimized.len(), reference.len());
        if !ties {
            assert_eq!(
                format!("{optimized:?}"),
                format!("{reference:?}"),
                "sequence {n}"
            );
        } else {
            let key = |r: &Candidate| (r.crc_valid, r.llr_distance.to_bits(), r.hard_error_count);
            let keys = |l: &[Candidate]| l.iter().map(key).collect::<Vec<_>>();
            assert_eq!(keys(&optimized), keys(&reference), "sequence {n}");
            if let Some(cutoff) = optimized.last() {
                // Records strictly better than the cutoff are all kept, but
                // tied ones may appear in any order.
                let better = |l: &[Candidate]| {
                    let mut kept: Vec<String> = l
                        .iter()
                        .filter(|r| compare_candidates(r, cutoff).is_lt())
                        .map(|r| format!("{r:?}"))
                        .collect();
                    kept.sort();
                    kept
                };
                assert_eq!(better(&optimized), better(&reference), "sequence {n}");
            }
        }
        for pair in optimized.windows(2) {
            assert!(compare_candidates(&pair[0], &pair[1]).is_le());
        }
    }
    assert!(zero >= 1 && larger >= 1 && overflow >= 1 && mixed >= 1);
}

/// Compares the decoder with every stage replaced by its reference implementation.
#[test]
fn all_references_together_match_the_optimized_decoder() {
    let all = ReferenceStages {
        fill: true,
        dispatch: true,
        basis: true,
        main: true,
        pre1: true,
        list: true,
        pre2_box_free: false,
        pre2_general: false,
    };
    let mut rng = SplitMix64(0x7c00);
    let mut cells = Vec::new();
    let mut push =
        |config: Config, rng: &mut SplitMix64, kind: Kind| cells.push(cell(config, rng, kind));
    for (c, w, kind) in [
        (0, 0, Kind::Flips(1, 0, 91)),
        (0, 1, Kind::Flips(2, 0, 91)),
        (7, 2, Kind::Flips(3, 0, 84)),
        (14, 3, Kind::Flips(4, 0, 77)),
        (0, 3, Kind::Clean),
        (0, 2, Kind::Awgn(1.2)),
        (14, 1, Kind::Awgn(3.0)),
        (7, 3, Kind::Awgn(8.0)),
        (0, 1, Kind::PureNoise),
    ] {
        push(cell_config(c, w, true, 300_000, None, true), &mut rng, kind);
    }
    // t on both paths and θ at its extremes.
    for (t, theta) in [
        (Some(0), None),
        (Some(64), Some(6)),
        (Some(70), Some(usize::MAX)),
        (Some(41), Some(1)),
        (None, Some(usize::MAX)),
    ] {
        let mut config = cell_config(0, 1, true, 300_000, Some(8), true);
        config.parity_prefix_len = t;
        config.preprocessing_threshold = theta;
        push(config, &mut rng, Kind::Flips(3, 0, 91));
    }
    // Caps, lists and the CRC requirement, with integer and zero-bearing input.
    push(
        cell_config(0, 3, true, 0, None, true),
        &mut rng,
        Kind::PureNoise,
    );
    push(
        cell_config(0, 3, true, 1, None, true),
        &mut rng,
        Kind::Flips(2, 0, 91),
    );
    push(
        cell_config(0, 3, true, 5, None, true),
        &mut rng,
        Kind::Awgn(2.0),
    );
    push(
        cell_config(7, 2, true, 40_000, Some(32), false),
        &mut rng,
        Kind::Awgn(3.0),
    );
    push(
        cell_config(0, 3, true, 300_000, None, false),
        &mut rng,
        Kind::Integers,
    );
    push(
        cell_config(14, 2, true, 300_000, None, true),
        &mut rng,
        Kind::WithZeros,
    );
    push(
        cell_config(0, 2, true, 300_000, None, true),
        &mut rng,
        Kind::AllEqual,
    );
    push(
        cell_config(0, 2, false, 300_000, Some(0), true),
        &mut rng,
        Kind::WideRange,
    );
    assert!((16..=40).contains(&cells.len()));
    let (mut statuses, mut cap_hits, mut lists, mut two_word) = (BTreeSet::new(), 0, 0, 0);
    for (n, cell) in cells.iter().enumerate() {
        // Exact ties may order list entries differently, so
        // degenerate inputs run with the list off.
        let mut cell = cell.clone();
        if cell.config.collect_candidates && n >= 14 + 5 && n != 18 {
            cell.config.collect_candidates = false;
        }
        let result = assert_decodes_agree(&cell, all, &format!("all-reference cell {n}"));
        statuses.insert(format!("{:?}", result.status));
        cap_hits += usize::from(result.diagnostics.candidate_cap_hit);
        lists += usize::from(!result.candidates.is_empty());
        two_word += usize::from(cell.config.parity_prefix_len.is_some_and(|t| t > 64));
    }
    assert!(statuses.len() >= 3, "{statuses:?}");
    assert!(cap_hits >= 3 && lists >= 3 && two_word >= 1);
}
