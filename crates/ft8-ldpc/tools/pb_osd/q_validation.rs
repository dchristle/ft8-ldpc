//! Reproducible numerical baselines and interleaved timing, run as ignored unit tests.

use crate::pb_osd::encoder::scheduler_tests::decoder_with_llrs;
use crate::pb_osd::probability::q_function;
use crate::pb_osd::probability::q_scaled;
use crate::pb_osd::{PromisingProbabilityModel, SearchStrategy};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/q-validation")
}

fn inputs() -> Vec<f32> {
    let mut values = Vec::with_capacity(4_194_312);
    for i in -1_048_576..=1_048_576 {
        values.push(i as f32 / 65_536.0);
    }
    let mut rng = StdRng::seed_from_u64(0x5146_554e_4354_494f);
    for _ in 0..1_048_576 {
        values.push(f32::from_bits(rng.random()));
    }
    for _ in 0..1_048_576 {
        values.push(rng.random_range(-16.0..=16.0));
    }
    values.extend([0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
    values
}

fn baseline_mode() -> bool {
    std::env::var("Q_VALIDATION_MODE").as_deref() == Ok("baseline")
}

#[test]
fn arbitrary_precision_reference() {
    let mut max_positive = 0.0f64;
    let mut max_negative = 0.0f64;
    let mut max_f32_relative = 0.0f64;
    let mut max_normal_f32_relative = 0.0f64;
    let mut count = 0;
    for row in include_bytes!("q_reference.bin").chunks_exact(20) {
        let x = f32::from_bits(u32::from_le_bytes(row[..4].try_into().unwrap()));
        if x > 14.0 { continue; }
        let y = (x / 2.0_f32.sqrt()) as f64;
        let positive = f64::from_le_bytes(row[4..12].try_into().unwrap());
        let negative = f64::from_le_bytes(row[12..20].try_into().unwrap());
        let got = q_scaled(y);
        max_positive = max_positive.max(((got-positive)/positive).abs());
        max_negative = max_negative.max(((q_scaled(-y)-negative)/negative).abs());
        max_f32_relative = max_f32_relative.max(((q_function(x) as f64-positive)/positive).abs());
        if q_function(x).is_normal() {
            max_normal_f32_relative = max_normal_f32_relative.max(((q_function(x) as f64-positive)/positive).abs());
        }
        assert!(((got-positive)/positive).abs() <= 1e-13, "x={x}: {got} vs {positive}");
        assert!(((q_scaled(-y)-negative)/negative).abs() <= 1e-13);
        assert_eq!(q_scaled(-y), 1.0-got);
        count += 1;
    }
    let report = format!("{count} independent references: max f64 relative error positive={max_positive:.17e}, negative={max_negative:.17e}; max f32 relative error={max_f32_relative:.17e}, normal outputs only={max_normal_f32_relative:.17e}\n");
    let dir = directory();
    if let Err(error) = fs::create_dir_all(&dir)
        .and_then(|()| fs::write(dir.join("accuracy.txt"), &report))
    {
        eprintln!("could not save accuracy report: {error}");
    }
    println!("{report}");
}

#[test]
fn q_properties_and_boundary_neighbors() {
    assert_eq!(q_function(0.0), 0.5);
    assert_eq!(q_function(-0.0), 0.5);
    assert!(q_function(f32::NAN).is_nan());
    assert_eq!(q_function(f32::INFINITY), 0.0);
    assert_eq!(q_function(f32::NEG_INFINITY), 1.0);
    let mut values = inputs();
    for segment in 0..=21 {
        let boundary = (segment as f32 * 0.5 * 2.0_f32.sqrt()).to_bits();
        for bits in boundary.saturating_sub(64)..=boundary+64 {
            values.push(f32::from_bits(bits));
            values.push(-f32::from_bits(bits));
        }
    }
    values.retain(|x| !x.is_nan());
    values.sort_unstable_by(f32::total_cmp);
    let mut previous = 1.0;
    for x in values {
        let q = q_function(x);
        assert!((0.0..=1.0).contains(&q));
        assert!(q <= previous, "nonmonotonic at x={x}: {previous} -> {q}");
        if x >= 0.0 {
            // Complementarity is exact before rounding; independent f32 rounding
            // can differ from subtracting an already-rounded positive tail.
            assert!(q_function(-x).to_bits().abs_diff((1.0-q).to_bits()) <= 1);
        }
        previous = q;
    }
    assert_eq!(q_function(16.0), 0.0);
    assert_eq!(q_function(-16.0), 1.0);
    let (mut lo, mut hi) = (14.0f32.to_bits(), 15.0f32.to_bits());
    while hi-lo > 1 {
        let mid = lo+(hi-lo)/2;
        if q_function(f32::from_bits(mid)) == 0.0 { hi = mid; }
        else { lo = mid; }
    }
    println!("last nonzero x={}, first zero x={}", f32::from_bits(lo), f32::from_bits(hi));
}

#[test]
#[ignore = "writes or compares the saved numerical and decode corpus"]
fn saved_baselines() {
    let dir = directory();
    fs::create_dir_all(&dir).unwrap();
    let baseline = baseline_mode();
    let label = if baseline { "baseline" } else { "candidate" };
    let path = dir.join(format!("q-{label}.bin"));
    if baseline {
        assert!(!path.exists(), "refusing to overwrite the baseline");
    }
    let mut out = BufWriter::new(File::create(path).unwrap());
    let old = if baseline { Vec::new() } else { fs::read(dir.join("q-baseline.bin")).unwrap() };
    let mut counts = [[0u64; 4]; 9];
    let mut relative = [0.0f64; 9];
    let values = inputs();
    if !baseline { assert_eq!(old.len(), values.len()*8); }
    for (i, &x) in values.iter().enumerate() {
        let q = q_function(x);
        out.write_all(&x.to_bits().to_le_bytes()).unwrap();
        out.write_all(&q.to_bits().to_le_bytes()).unwrap();
        if baseline || !x.is_finite() { continue; }
        assert_eq!(x.to_bits(), u32::from_le_bytes(old[8*i..8*i+4].try_into().unwrap()));
        let expected = u32::from_le_bytes(old[8*i+4..8*i+8].try_into().unwrap());
        let bin = match x.abs() {
            a if a <= 1.0 => 0, a if a <= 2.0 => 1, a if a <= 3.0 => 2,
            a if a <= 4.0 => 3, a if a <= 5.0 => 4, a if a <= 6.0 => 5,
            a if a <= 8.0 => 6, a if a <= 14.0 => 7, _ => 8,
        };
        let ulps = q.to_bits().abs_diff(expected);
        counts[bin][0] += 1;
        counts[bin][1] += u64::from(ulps != 0);
        counts[bin][2] = counts[bin][2].max(ulps as u64);
        if ulps != 0 && expected == 0 { counts[bin][3] += 1; }
        let reference = f32::from_bits(expected) as f64;
        if reference != 0.0 {
            relative[bin] = relative[bin].max(((q as f64 - reference) / reference).abs());
        }
    }
    out.flush().unwrap();
    if !baseline {
        let mut report = String::from("bin,count,mismatches,max_f32_ulps,old_zero_new_nonzero,max_relative_to_nonzero_old\n");
        for i in 0..9 {
            report += &format!("{i},{},{},{},{},{:.17e}\n", counts[i][0], counts[i][1], counts[i][2], counts[i][3], relative[i]);
        }
        fs::write(dir.join("mismatches.csv"), &report).unwrap();
        println!("{report}");
        assert!(counts[..6].iter().all(|row| row[2] <= 1));
    }

    let mut corpus = String::new();
    let mut rng = StdRng::seed_from_u64(0x4445_434f_4445_5146);
    let generator = ft8_ldpc_core::fec::CascadedGeneratorMatrix::new(0).unwrap();
    let strategies = [SearchStrategy::LayeredBranchPruned, SearchStrategy::SimpleByWeight,
        SearchStrategy::EagerProbabilitySort, SearchStrategy::LazyProbabilityBestFirst];
    let mut successes = 0;
    let mut teps = 0u64;
    for case in 0..512 {
        let message = std::array::from_fn(|_| rng.random_bool(0.5));
        let codeword = generator.encode(&ft8_ldpc_core::crc::add_crc14(&message));
        let sigma = [0.45, 0.7, 0.95, 1.2, 1.5, 2.0, 3.0, 5.0][case % 8];
        let llrs = std::array::from_fn(|i| {
            let noise: f32 = rng.sample(StandardNormal);
            (if codeword[i] { -1.0 } else { 1.0 }) + sigma * noise
        });
        let mut decoder = decoder_with_llrs(llrs, 2 + usize::from(case % 16 == 0), strategies[(case / 8) % 4]);
        decoder.set_diagnostics_enabled(true);
        if case % 3 == 0 {
            decoder.set_promising_probability_model(PromisingProbabilityModel::WeightedNormal);
        }
        if case % 11 == 0 { decoder.set_calibration_trace_enabled(true); }
        let result = decoder.decode();
        successes += usize::from(result.success);
        teps += result.num_teps_processed as u64;
        // Debug includes every field; append float bits to preserve exact representations.
        corpus += &format!("case {case} llrs {:?}\n{result:?}\n", llrs.map(f32::to_bits));
        corpus += &format!("distance bits {} {}\n", result.distance.to_bits(), result.best_valid_distance.to_bits());
        if let Some(d) = &result.diagnostics {
            corpus += &format!("diagnostic bits {} {} {} {:?}\n", d.gamma.to_bits(), d.best_distance.to_bits(),
                d.best_valid_distance.to_bits(), d.distance_improvements.iter().map(|(n, f)| (*n, f.to_bits())).collect::<Vec<_>>());
        }
        if let Some(t) = &result.calibration_trace {
            corpus += &format!("trace bits {}\n", t.brier_sum.to_bits());
            for b in &t.bins {
                corpus += &format!("bin bounds bits {} {}\n", b.lower_probability.to_bits(), b.upper_probability.to_bits());
                corpus += &format!("bin bits {:?}\n", [b.predicted_sum,
                    b.brier_sum, b.incumbent_distance_sum, b.message_distance_sum, b.candidate_distance_sum].map(f64::to_bits));
            }
        }
    }
    let path = dir.join(format!("decodes-{label}.txt"));
    if baseline { assert!(!path.exists()); }
    fs::write(path, &corpus).unwrap();
    println!("512 decodes: {successes} successes, {teps} processed TEPs");
    if !baseline {
        let expected = fs::read_to_string(dir.join("decodes-baseline.txt")).unwrap();
        if corpus != expected {
            let line = corpus.lines().zip(expected.lines()).position(|(a,b)| a != b);
            panic!("decode corpus changed at line {line:?}; candidate saved for inspection");
        }
    }
}

#[inline(never)]
fn old_q(x: f32) -> f32 {
    (0.5 * (1.0 - libm::erf((x / 2.0_f32.sqrt()) as f64))) as f32
}

#[inline(never)]
fn new_q(x: f32) -> f32 { q_function(x) }

#[test]
#[ignore = "interleaved release microbenchmark"]
fn timing() {
    use std::hint::black_box;
    use std::time::Instant;
    assert!(!cfg!(debug_assertions), "use --release");
    let mut rng = StdRng::seed_from_u64(0x4245_4e43_4851);
    let values: Vec<f32> = (0..262_144).map(|_| rng.random_range(-8.0..8.0)).collect();
    fn run(f: fn(f32) -> f32, values: &[f32]) -> f64 {
        let start = Instant::now();
        for _ in 0..16 {
            for &x in values { black_box(f(black_box(x))); }
        }
        start.elapsed().as_secs_f64() * 1e9 / (values.len() * 16) as f64
    }
    run(old_q, &values); run(new_q, &values);
    let mut report = String::from("pair,old_ns,new_ns\n");
    for pair in 0..12 {
        let (old, new) = if pair % 2 == 0 {
            (run(old_q, &values), run(new_q, &values))
        } else {
            let new = run(new_q, &values);
            (run(old_q, &values), new)
        };
        report += &format!("{pair},{old:.6},{new:.6}\n");
    }
    fs::create_dir_all(directory()).unwrap();
    fs::write(directory().join("timings.csv"), &report).unwrap();
    println!("{report}");
}
