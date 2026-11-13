//! Public API and configuration validation.
use ft8_ldpc::wu2007::*;

fn send_sync_clone<T: Send + Sync + Clone>() {}

#[test]
fn configuration_fields_match_defaults() {
    let config = Config {
        cascade_bits: 0,
        redundancy_order: RedundancyOrder::ReliabilityDescending,
        search_order: 3,
        enable_first_preprocessing: true,
        enable_second_preprocessing: false,
        parity_prefix_len: None,
        preprocessing_threshold: None,
        second_preprocessing_prefix_len: None,
        max_recoded_candidates: 300_000,
        max_list_candidates: 32,
        collect_candidates: false,
        require_crc: true,
    };
    assert_eq!(config, Config::default());
    assert_eq!(config.validate(), Ok(()));
    let llr = [4.0; 174];
    assert_eq!(
        Decoder::default().decode(&llr).codeword,
        Decoder::new(config).unwrap().decode(&llr).codeword
    );
}

#[test]
fn every_public_type_is_send_sync_and_clone() {
    send_sync_clone::<Config>();
    send_sync_clone::<Decoder>();
    send_sync_clone::<ConfigError>();
    send_sync_clone::<DecodeResult>();
    send_sync_clone::<Candidate>();
    send_sync_clone::<Diagnostics>();
    send_sync_clone::<Status>();
    send_sync_clone::<RedundancyOrder>();
}

fn rejected(config: Config) -> ConfigError {
    let error = config.validate().unwrap_err();
    assert_eq!(Decoder::new(config).err().unwrap(), error);
    error
}

#[test]
fn validation_and_construction_report_the_same_errors_in_order() {
    let mut config = Config {
        cascade_bits: 15,
        search_order: usize::MAX,
        ..Default::default()
    };
    let error = rejected(config.clone());
    assert!(matches!(
        error,
        ConfigError::CascadeBitsOutOfRange { value: 15, max: 14 }
    ));
    assert_eq!(error.to_string(), "cascade_bits must be in 0..=14, got 15");
    assert!(std::error::Error::source(&error).is_none());
    config.cascade_bits = 0;
    for first in [false, true] {
        for second in [false, true] {
            config.enable_first_preprocessing = first;
            config.enable_second_preprocessing = second;
            let layers = if second { 2 } else { usize::from(first) };
            for order in [7 - layers, usize::MAX] {
                config.search_order = order;
                let error = rejected(config.clone());
                assert!(
                    matches!(error, ConfigError::CandidatePositionCapacityExceeded {
                    search_order, preprocessing_extra_layers, capacity: 6
                } if search_order == order && preprocessing_extra_layers == layers)
                );
                assert_eq!(
                    error.to_string(),
                    "search_order plus preprocessing layers must fit 6 candidate positions"
                );
                assert!(std::error::Error::source(&error).is_none());
            }
            config.search_order = 6 - layers;
            assert_eq!(config.validate(), Ok(()));
            assert!(Decoder::new(config.clone()).is_ok());
        }
    }
    for cascade_bits in 0..=14 {
        let config = Config {
            cascade_bits,
            ..Default::default()
        };
        assert_eq!(config.validate(), Ok(()));
        assert!(Decoder::new(config).is_ok());
    }
}

#[test]
fn prefix_clamping_and_disabled_options_preserve_results() {
    let llr = std::array::from_fn(|i| if i % 5 == 0 { -1.5 } else { 2.25 });
    for first in [false, true] {
        for second in [false, true] {
            let exact = Config {
                search_order: 1,
                enable_first_preprocessing: first,
                enable_second_preprocessing: second,
                parity_prefix_len: Some(83),
                second_preprocessing_prefix_len: Some(83),
                max_recoded_candidates: 256,
                collect_candidates: true,
                max_list_candidates: 0,
                ..Default::default()
            };
            let mut over = exact.clone();
            over.parity_prefix_len = Some(usize::MAX);
            over.second_preprocessing_prefix_len = Some(usize::MAX);
            if !first {
                over.preprocessing_threshold = Some(usize::MAX);
            }
            assert_eq!(over.validate(), Ok(()));
            let a = Decoder::new(exact).unwrap().decode(&llr);
            let b = Decoder::new(over).unwrap().decode(&llr);
            assert_eq!(a.llr_distance.to_bits(), b.llr_distance.to_bits());
            assert_eq!(
                a.diagnostics.best_any_llr_distance.map(f32::to_bits),
                b.diagnostics.best_any_llr_distance.map(f32::to_bits)
            );
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            assert!(a.candidates.is_empty());
        }
    }
}

#[test]
fn default_diagnostic_counters_are_empty() {
    let d = Diagnostics::default();
    let Diagnostics {
        rank,
        cascade_bits,
        active_k,
        active_r,
        visited_steps,
        enumerated_patterns,
        recoded_candidates,
        scored_candidates,
        main_enumerated_patterns,
        main_scored_candidates,
        materialized_candidates,
        crc_checked_candidates,
        preprocess1_examined,
        preprocess1_rejects,
        preprocess1_scored_candidates,
        preprocess2_enumerated_patterns,
        preprocess2_scored_candidates,
        preprocess2_probe_masks,
        preprocess2_key_probes,
        crc_failures,
        candidate_cap_hit,
        best_any_llr_distance,
        best_any_crc_valid,
        ..
    } = d;
    assert_eq!(
        [
            rank,
            cascade_bits,
            active_k,
            active_r,
            visited_steps,
            enumerated_patterns,
            recoded_candidates,
            scored_candidates,
            main_enumerated_patterns,
            main_scored_candidates,
            materialized_candidates,
            crc_checked_candidates,
            preprocess1_examined,
            preprocess1_rejects,
            preprocess1_scored_candidates,
            preprocess2_enumerated_patterns,
            preprocess2_scored_candidates,
            preprocess2_probe_masks,
            preprocess2_key_probes,
            crc_failures
        ],
        [0; 20]
    );
    assert!(!candidate_cap_hit && !best_any_crc_valid && best_any_llr_distance.is_none());
}

/// A zero candidate cap returns an empty result with `CandidateCapHit`.
#[test]
fn a_zero_cap_gives_the_empty_result() {
    let llr: [f32; 174] = std::array::from_fn(|n| if n % 5 == 0 { -1.5 } else { 2.25 });
    for second in [false, true] {
        let mut config = Config {
            cascade_bits: 0,
            ..Default::default()
        };
        config.max_recoded_candidates = 0;
        config.enable_second_preprocessing = second;
        let result = Decoder::new(config).unwrap().decode(&llr);
        assert_eq!(result.status, Status::CandidateCapHit);
        assert_eq!(result.codeword, [0; 3]);
        assert_eq!(result.payload, [0; 2]);
        assert_eq!(result.message_with_crc, [0; 2]);
        assert_eq!(result.llr_distance, f32::INFINITY);
        assert_eq!(result.hard_error_count, 0);
        assert!(!result.crc_valid && result.candidates.is_empty());
        let d = result.diagnostics;
        assert_eq!((d.scored_candidates, d.recoded_candidates), (0, 0));
        assert!(d.candidate_cap_hit);
        assert_eq!(d.preprocess2_probe_masks, 0);
        assert_eq!((d.rank, d.active_k, d.active_r), (91, 91, 83));
    }
}

#[test]
fn candidate_records_keep_the_fixed_layout() {
    let mut config = Config {
        cascade_bits: 0,
        ..Default::default()
    };
    config.collect_candidates = true;
    config.search_order = 1;
    let llr: [f32; 174] = std::array::from_fn(|n| ((n * 37) % 11) as f32 / 4. + 0.5);
    let result = Decoder::new(config).unwrap().decode(&llr);
    for c in &result.candidates {
        assert!(c.num_positions as usize <= CANDIDATE_POSITION_CAPACITY_FOR_TEST);
        assert_eq!(c.order, c.num_positions);
        assert!(c.positions[c.num_positions as usize..]
            .iter()
            .all(|&p| p == 0));
    }
}
const CANDIDATE_POSITION_CAPACITY_FOR_TEST: usize = ft8_ldpc::wu2007::CANDIDATE_POSITION_CAPACITY;
