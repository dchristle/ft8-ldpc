//! Checks allocation-free decoding after construction with candidate collection disabled.
use ft8_ldpc::wu2007::{Config, Decoder, SelectionPolicy, Status};
use ft8_ldpc_core::{crc::add_crc14, fec::BASE_GENERATOR_MATRIX};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct CountAlloc;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

#[test]
fn collection_off_decodes_allocate_nothing_after_construction() {
    let mut seed = 0x52a9_73c1_u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed
    };
    let noise: [f32; 174] = std::array::from_fn(|_| {
        let bits = next();
        let magnitude = (1 + (bits >> 56)) as f32 / 32.0;
        if bits & (1 << 32) == 0 {
            magnitude
        } else {
            -magnitude
        }
    });
    let payload = std::array::from_fn(|_| next() & (1 << 32) != 0);
    let message = add_crc14(&payload);
    let codeword = BASE_GENERATOR_MATRIX.encode(&message);
    let clean: [f32; 174] = std::array::from_fn(|i| {
        let magnitude = (1 + (next() >> 56)) as f32 / 32.0;
        if codeword[i] {
            -magnitude
        } else {
            magnitude
        }
    });
    let mut positions: [usize; 174] = std::array::from_fn(|i| i);
    positions.sort_by(|&a, &b| clean[b].abs().total_cmp(&clean[a].abs()).then(a.cmp(&b)));
    let perturbed = |count: usize| {
        let mut llr = clean;
        for (i, &position) in positions[..count].iter().enumerate() {
            // Keep one error in the reliable basis; weaken the remaining reversals.
            llr[position] *= if i == 0 { -1.0 } else { -1.0 / 1024.0 };
        }
        llr
    };
    let inputs = [
        ("noise", noise, None),
        ("clean", clean, Some(0)),
        ("3 reversals", perturbed(3), Some(3)),
        ("8 reversals", perturbed(8), Some(8)),
    ];
    let pack = |bits: &[bool]| {
        let mut words = [0; 3];
        for (i, &bit) in bits.iter().enumerate() {
            words[i / 64] |= (bit as u64) << (i % 64);
        }
        words
    };
    let expected_codeword = pack(&codeword);
    let expected_payload = [pack(&payload)[0], pack(&payload)[1]];
    let expected_message = [pack(&message)[0], pack(&message)[1]];
    let interference = noise.map(|x| -x);
    let mut searched_recoveries = [false; 2];
    for (cascade_bits, selection_policy) in [0, 14].into_iter().flat_map(|c| {
        [
            SelectionPolicy::BestCrcValid,
            SelectionPolicy::MinMetricThenCrc,
        ]
        .map(|p| (c, p))
    }) {
        for search_order in 0..=4 {
            for (first, second, tau) in [
                (false, false, None),
                (true, false, None),
                (true, true, None),
                (true, true, Some(16)),
            ] {
                for require_crc in [false, true] {
                    let config = Config {
                        selection_policy,
                        cascade_bits,
                        search_order,
                        enable_first_preprocessing: first,
                        enable_second_preprocessing: second,
                        second_preprocessing_prefix_len: tau,
                        collect_candidates: false,
                        require_crc,
                        ..Default::default()
                    };
                    let mut successes = [0; 2];
                    let mut second_preprocessing_seen = false;
                    for (name, llr, reversals) in &inputs {
                        let mut decoder = Decoder::new(config.clone()).unwrap();
                        let decode = |decoder: &mut Decoder, llr: &[f32; 174], phase: &str| {
                            ALLOCATIONS.store(0, Ordering::Relaxed);
                            TRACK.store(true, Ordering::Relaxed);
                            let result = std::hint::black_box(decoder.decode(llr));
                            TRACK.store(false, Ordering::Relaxed);
                            assert_eq!(
                                ALLOCATIONS.load(Ordering::Relaxed),
                                0,
                                "{config:?}, {name}, {phase}"
                            );
                            assert!(result.candidates.is_empty());
                            result
                        };
                        let fresh = decode(&mut decoder, llr, "fresh");
                        decode(&mut decoder, &interference, "intervening noise");
                        let reused = decode(&mut decoder, llr, "reused");
                        assert_eq!(
                            format!("{fresh:?}"),
                            format!("{reused:?}"),
                            "{config:?}, {name}"
                        );
                        for (phase, result) in [fresh, reused].iter().enumerate() {
                            second_preprocessing_seen |=
                                result.diagnostics.preprocess2_key_probes > 0;
                            if reversals.is_some()
                                && (search_order > 0 || first || *reversals == Some(0))
                            {
                                assert!(result.crc_valid, "{config:?}, {name}");
                                assert_eq!(
                                    result.codeword, expected_codeword,
                                    "{config:?}, {name}"
                                );
                                assert_eq!(result.payload, expected_payload);
                                assert_eq!(result.message_with_crc, expected_message);
                                assert_eq!(Some(result.hard_error_count as usize), *reversals);
                            } else if reversals.is_some() {
                                assert_ne!(
                                    result.codeword, expected_codeword,
                                    "search is required: {config:?}, {name}"
                                );
                            }
                            if result.status == Status::Success
                                && result.crc_valid
                                && reversals.is_some()
                            {
                                successes[phase] += 1;
                                if let Some(index) = match reversals {
                                    Some(3) => Some(0),
                                    Some(8) => Some(1),
                                    _ => None,
                                } {
                                    let d = &result.diagnostics;
                                    searched_recoveries[index] |= d.main_scored_candidates
                                        + d.preprocess1_scored_candidates
                                        + d.preprocess2_scored_candidates
                                        > 0;
                                }
                            }
                        }
                    }
                    assert!(
                        successes.iter().all(|&count| count > 0),
                        "no CRC-valid success: {config:?}"
                    );
                    if second && search_order <= 3 {
                        assert!(
                            second_preprocessing_seen,
                            "second preprocessing was not reached: {config:?}"
                        );
                    }
                }
            }
        }
    }
    assert!(searched_recoveries.into_iter().all(|recovered| recovered));
}
