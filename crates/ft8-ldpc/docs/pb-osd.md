# pb_osd

An implementation of the probability-based ordered-statistics decoder (PB-OSD) of Yue et al. (2021) for the FT8 (174,91) code. The LDPC tables and CRC definition are from the FT4/FT8 QEX paper (Franke, Somerville and Taylor).

The decoder takes 174 log-likelihood ratios and searches codewords by flipping selected positions in a reliability-ordered basis. It can fold 0 to 14 CRC bits into the code ("cascade bits").

The search stops early when a success probability exceeds its threshold and discards patterns whose promising probability is below its threshold.

## Example

```rust
use ft8_ldpc_core::crc::add_crc14;
use ft8_ldpc_core::fec::CascadedGeneratorMatrix;
use ft8_ldpc::pb_osd::{Config, Decoder};
use ft8_ldpc_core::protocol::{PAYLOAD_BITS, N};

fn main() {
    // k = 91: the CRC is checked separately from the code.
    let generator = CascadedGeneratorMatrix::new(0).unwrap();
    let payload: [bool; PAYLOAD_BITS] = std::array::from_fn(|i| i % 3 == 0);
    let codeword = generator.encode(&add_crc14(&payload));

    // Corrupt three bits.
    let mut llr: [f32; N] = std::array::from_fn(|i| if codeword[i] { -4.0 } else { 4.0 });
    for i in [3, 40, 100] {
        llr[i] = -llr[i] * 0.25;
    }

    // gamma scales the LLRs to the reliabilities alpha_i = |LLR_i| / (4 gamma).
    let gamma = llr.iter().map(|v| v.abs()).sum::<f32>() / (4.0 * N as f32);
    let alpha: [f32; N] = std::array::from_fn(|i| llr[i].abs() / (4.0 * gamma));

    let decoder = Decoder::new(Config {
        gamma,
        reliabilities: Some(alpha),
        max_order: 2,
        ..Config::default()
    }).unwrap();
    let result = decoder.decode(&llr);
    assert!(result.success);
    assert_eq!(result.codeword.to_vec(), codeword);
}
```

Run this example with `cargo run --release -p ft8-ldpc --example pb_osd_decode_noisy_codeword`. Reuse a decoder across received words. Each `decode(&llr)` constructs its basis and runs an independent search.

## Results

When `success` is true, the primary `codeword` and `llr_distance` report the lowest-distance candidate found, which may fail the CRC. Read its payload from `codeword[..77]`. When `success` is false, `codeword` holds the input hard decisions, `hard_error_count` is zero, and `llr_distance` is infinity (NaN for zero gamma); these are placeholders rather than candidate measurements.

`best_valid: Option<ValidCandidate>` holds the lowest-distance CRC-valid candidate accepted by the [message gate](#message-gate). `Some` supplies its natural-order `codeword` and `llr_distance`; `None` means no such candidate was selected. Distance scaling can overflow or produce NaN even for a selected candidate, so use the option to determine availability.

## Conventions

- **LLR sign:** follows the [crate-level convention](../src/lib.rs).
- **Distances** in the result are in LLR units. Internally the decoder works with the reliabilities `alpha_i = |LLR_i| / (4 gamma)`.
- **Order:** `max_order` is the largest number of flipped positions in a test error pattern.

## Configuration

The default `LayeredBranchPruned` strategy visits test error patterns in breadth layers without a global probability sort. `SimpleByWeight` enumerates by weight; `EagerProbabilitySort` and `LazyProbabilityBestFirst` visit patterns in descending probability order.

`Config` has public fields, implements `Default`, and provides `validate()`. Pass it to `Decoder::new(config)`; invalid generator, scaling, or threshold settings return a `ConfigError`.

Its policies are `SearchStrategy`, `EpsilonModel`, `PromisingProbabilityModel`, and `Threshold` (automatic, scaled, explicit, or disabled). `BasisReductionKernel::Auto` selects blocked elimination with Gray-code combination tables (the Method of Four Russians); the other kernels produce the same basis.

`gamma` defaults to 1.0. With `reliabilities: None`, each frame derives `alpha_i = abs(LLR_i) / (4 * gamma)` in `f32`. Supply `Some(alpha)` to use exact caller-provided reliabilities unchanged. Diagnostic distances and message-gate distances use this internal scale; result distances multiply it by `4 * gamma`.

## Message gate

Set `Config::message_gate` to `Arc::new(my_gate)` to install a `MessageGate`. It receives a CRC-valid 77-bit payload and its internal-scale distance and returns whether to accept it.

During search, the gate is called only for CRC-valid candidates whose internal-scale distance improves on the best accepted candidate. Rejection prevents an update to `best_valid`; the primary result and `success` are unaffected. The default `AcceptAll` gate accepts every CRC-valid distance improvement.

An input that already satisfies the code and CRC is returned immediately as the primary codeword and as `Some(ValidCandidate)` at zero distance without calling the gate.

## Tests

`cargo test --release -p ft8-ldpc --no-default-features --features pb-osd` runs the unit tests: probability formulas against explicit sums, the packed distance tables against a scalar reference, the basis reductions against each other, and decoder runs at several orders and cascade levels.

## References

- C. Yue, M. Shirvanimoghaddam, G. Park, O.-S. Park, B. Vucetic and Y. Li, "Probability-Based Ordered-Statistics Decoding for Short Block Codes," IEEE Communications Letters, 2021, DOI 10.1109/LCOMM.2021.3058978.
- M. P. C. Fossorier and S. Lin, "Soft-Decision Decoding of Linear Block Codes Based on Ordered Statistics," IEEE Transactions on Information Theory, vol. 41, no. 5, September 1995.
- V. L. Arlazarov, E. A. Dinic, M. A. Kronrod and I. A. Faradzev, "On economical construction of the transitive closure of an oriented graph," 1970; and M. Albrecht, G. Bard and C. Pernet, "Efficient Dense Gaussian Elimination over the Finite Field with Two Elements," arXiv:1111.6549, for the Gray-code table construction.
- M. López-Benítez and F. Casadevall, "Versatile, Accurate, and Analytically Tractable Approximation for the Gaussian Q-Function," IEEE Transactions on Communications, vol. 59, no. 4, April 2011, for the Q-function approximation constants.
- S. Franke, B. Somerville and J. Taylor, "The FT4 and FT8 Communication Protocols," QEX, July/August 2020.
