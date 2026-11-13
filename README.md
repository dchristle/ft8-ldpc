# ft8-ldpc

LDPC decoders and coding primitives in Rust for the FT8 and FT4 (174,91) code:
77 payload bits, 14 CRC bits and 83 parity bits.

The decoders take 174 log-likelihood ratios and return a codeword.
Synchronization, demodulation, signal subtraction and message unpacking are
not part of this repository.

## Crates

- `ft8-ldpc-core`: modules `protocol` (code dimensions), `crc` (CRC-14) and `fec`
  (LDPC tables, generator matrices including the CRC-cascaded variants, and
  packed GF(2) matrices).
- `ft8-ldpc`: three LDPC decoders, each behind a cargo feature. All are on
  by default.

| Module | Feature | Decoder |
|---|---|---|
| [`wu2007`](crates/ft8-ldpc/docs/wu2007.md) | `wu2007` | Ordered-statistics decoder after Wu and Hadjicostis (2007), with first and second preprocessing |
| [`pb_osd`](crates/ft8-ldpc/docs/pb-osd.md) | `pb-osd` | Probability-based ordered-statistics decoder after Yue et al. (2021) |
| [`fast_min_sum`](crates/ft8-ldpc/docs/fast-min-sum.md) | `fast-min-sum` | Fixed-point layered offset-min-sum decoder, scalar and batched |

## Conventions

The [crate documentation](crates/ft8-ldpc/src/lib.rs) defines the LLR sign
convention and each decoder's exact-zero decision.

## Building

`rust-toolchain.toml` pins stable Rust 1.96.1.

```sh
cargo test --workspace --release
cargo run --release -p ft8-ldpc --example pb_osd_decode_noisy_codeword
```

CI runs the tests on x86-64 Linux and Apple Silicon.

Tests include seeded decoder checks, comparisons against reference kernels,
and CRC and encoder vectors in `tests/fixtures/known-answer/`.

## Protocol reference

Steve Franke, Bill Somerville and Joe Taylor, *The FT4 and FT8 Communication
Protocols*, QEX, July/August 2020, §3 describes the CRC and LDPC definitions.
Reference [14] in the paper identifies the companion matrices in
[`ft4_ft8_protocols.tgz`](http://physics.princeton.edu/pulsar/k1jt/ft4_ft8_protocols.tgz).
For the authors' statement concerning the protocol description and companion
resources, see §9, "Concluding Remarks and Software License," printed page 15
(PDF page 9).

## Contributing

Contributions must be your own work and must not be copied from software under
incompatible licenses.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
