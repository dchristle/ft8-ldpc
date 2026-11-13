# ft8-ldpc

Three LDPC decoders for the FT8 and FT4 (174,91) code, one module and one cargo
feature each; all features are on by default. Each takes 174 log-likelihood
ratios and returns the decoded codeword. The code tables, CRC and matrix types
come from `ft8-ldpc-core`.

| Module | Feature | Page |
|---|---|---|
| `wu2007` | `wu2007` | [docs/wu2007.md](docs/wu2007.md) |
| `pb_osd` | `pb-osd` | [docs/pb-osd.md](docs/pb-osd.md) |
| `fast_min_sum` | `fast-min-sum` | [docs/fast-min-sum.md](docs/fast-min-sum.md) |

Runnable examples are in `examples/`.
