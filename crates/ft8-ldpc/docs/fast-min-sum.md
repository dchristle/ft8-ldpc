# fast_min_sum

A fixed-point layered offset-min-sum decoder for the FT8 (174,91) LDPC code, with single-word and batch decoding. It uses 16-bit integers and the code graph from the FT4/FT8 QEX paper (Franke, Somerville and Taylor).

The decoder takes 174 log-likelihood ratios and iteratively updates bit beliefs to satisfy the parity checks. Results include hard decisions, refined LLRs, and the number of unsatisfied checks.

## Example

```rust
use ft8_ldpc_core::crc::verify_codeword_crc14;
use ft8_ldpc::fast_min_sum::Decoder;
use ft8_ldpc_core::fec::encode_174_91;
use ft8_ldpc_core::protocol::{PAYLOAD_BITS, N};

fn main() {
    let payload: [bool; PAYLOAD_BITS] = std::array::from_fn(|i| i % 3 == 0);
    let codeword = encode_174_91(&payload);
    assert!(verify_codeword_crc14(&codeword));

    // Weaken and flip four bits.
    let mut llrs: [f32; N] = std::array::from_fn(|i| if codeword[i] { -4.0 } else { 4.0 });
    for i in [3, 40, 100, 150] {
        llrs[i] = -llrs[i] * 0.25;
    }

    let result = Decoder::default().decode(&llrs);

    // Check CRC validity separately from parity-check convergence.
    assert!(result.success);
    assert!(verify_codeword_crc14(&result.codeword));
    assert_eq!(result.codeword, codeword);
}
```

Run this example with `cargo run --release -p ft8-ldpc --example fast_min_sum_decode_noisy_codeword`.

## How it works

- **Quantization.** Each LLR is negated, because the kernels treat a non-negative value as bit 1, then multiplied by `llr_scale`, rounded, and clamped to `±max_channel_value`, giving an `i16`.
- **Layered schedule.** The checks are processed one after another. Each check updates the beliefs of its bits immediately, so later checks in the same iteration already see the new values.
- **Check update.** For each check the outgoing magnitude is the smallest magnitude among the other inputs, less `offset`, floored at zero and capped at `max_message_value`; the sign is the product of the other inputs' signs. Only the smallest and second-smallest magnitudes are tracked. Beliefs saturate at `max_belief_value`. The 59 checks of degree 6 and the 24 of degree 7 have separate code paths.
- **Stopping.** After each iteration the unsatisfied checks are counted. Decoding stops when the count is at most `max_unsatisfied_checks`, when the lowest count so far has not been beaten for `stall_limit` iterations in a row (after at least three iterations), or at `max_iterations`. With `max_iterations` of 0 no iteration runs and `unsatisfied_checks` is reported as 83 without being computed, so `success` is false for any `max_unsatisfied_checks` below 83.

## Results

Input LLR conventions are defined in the [crate documentation](../src/lib.rs).

- **`success`:** the reported `unsatisfied_checks` is at most `max_unsatisfied_checks` (default 0). With zero iterations, the reported count is the sentinel 83 rather than a measurement. Success does not establish CRC validity, so check it with `ft8_ldpc_core::crc::verify_codeword_crc14`.
- **`normalized_distance`:** zero when `gamma < 1e-9`; otherwise the sum of `|LLR| / (4 gamma)` over the bits where the final decision differs from the input's hard decision. `Config::gamma_estimate` supplies `gamma`; with `None` it is a quarter of the mean `|LLR|`. In the sum branch, multiply by `4 gamma` to get LLR units.
- **`hard_error_count`:** the number of bits where the final decision differs from the input's hard decision.
- **`refined_llrs`:** the final beliefs, negated back to the public sign and divided by `llr_scale`.

## Batch decoding

`BatchDecoder<LANES>` decodes `LANES` received words together and keeps its working buffers between calls:

```rust
use ft8_ldpc::fast_min_sum::BatchDecoder;
use ft8_ldpc_core::protocol::N;

let mut batch = BatchDecoder::<8>::default();
let llrs = [[0.0f32; N]; 8]; // eight received words
let results = batch.decode(&llrs); // one result per lane
```

Each lane gives the same codeword, iteration count and unsatisfied-check count as the single-word decoder on the same input. A lane that has converged or stalled is frozen while the others continue.

With 8 lanes on aarch64 the check updates use NEON only when `max_message_value` is 63 and `max_belief_value` is 255, their defaults. Other bounds, targets and lane counts use portable scalar updates.

`BatchConfig<LANES>` holds the shared `decoder: Config` and optional `gamma_estimates: [f32; LANES]`. Per-lane estimates override the shared `Config::gamma_estimate`; without either override, each lane estimates gamma from its own input.

## Configuration

Set public fields on `Config::default()`, then pass the configuration to `Decoder::new(config)`. Both `Config` and `BatchConfig` provide `validate()`, and constructors validate automatically. `Decoder::default()` and `BatchDecoder::default()` use default settings.

| Field | Default | Meaning |
|---|---|---|
| `max_iterations` | 8 | Iteration limit |
| `stall_limit` | 3 | Iterations in a row without a new lowest unsatisfied-check count before stopping |
| `max_unsatisfied_checks` | 0 | Unsatisfied checks allowed for `success` |
| `llr_scale` | 8.0 | Multiplier from LLR to quantized units |
| `offset` | 2 | Offset subtracted from check magnitudes |
| `max_channel_value` | 127 | Clamp on quantized input LLRs |
| `max_message_value` | 63 | Clamp on check-to-bit messages |
| `max_belief_value` | 255 | Clamp on beliefs |
| `gamma_estimate` | `None` | Optional distance normalizer |

Validation rejects negative channel bounds and negative belief bounds when iterations are enabled. Negative offsets are treated as zero. Scale and gamma values are used as supplied; zero scale can produce nonfinite refined LLRs.

## Tests

`cargo test --release -p ft8-ldpc --no-default-features --features fast-min-sum` runs the unit tests: a clean codeword decodes with no unsatisfied checks, noisy input gives bounded output, the batch decoder matches the scalar decoder lane by lane (4 and 8 lanes), and the default `gamma` is a quarter of the mean `|LLR|`.

## References

- M. P. C. Fossorier, M. Mihaljević and H. Imai, "Reduced Complexity Iterative Decoding of Low-Density Parity Check Codes Based on Belief Propagation," IEEE Transactions on Communications, vol. 47, no. 5, May 1999, DOI 10.1109/26.768759, for the min-sum approximation.
- J. Chen, A. Dholakia, E. Eleftheriou, M. P. C. Fossorier and X.-Y. Hu, "Reduced-Complexity Decoding of LDPC Codes," IEEE Transactions on Communications, vol. 53, no. 8, August 2005, DOI 10.1109/TCOMM.2005.852852, for the normalized and offset forms.
- D. E. Hocevar, "A Reduced Complexity Decoder Architecture via Layered Decoding of LDPC Codes," IEEE Workshop on Signal Processing Systems (SiPS), 2004, pp. 107-112, and M. M. Mansour and N. R. Shanbhag, "High-Throughput LDPC Decoders," IEEE Transactions on VLSI Systems, vol. 11, no. 6, December 2003, pp. 976-996, for layered scheduling.
- S. Franke, B. Somerville and J. Taylor, "The FT4 and FT8 Communication Protocols," QEX, July/August 2020, for the code.
