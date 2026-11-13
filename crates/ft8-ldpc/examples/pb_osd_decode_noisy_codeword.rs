use ft8_ldpc::pb_osd::{Config, Decoder};
use ft8_ldpc_core::crc::add_crc14;
use ft8_ldpc_core::fec::CascadedGeneratorMatrix;
use ft8_ldpc_core::protocol::{N, PAYLOAD_BITS};

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
    })
    .unwrap();
    let result = decoder.decode(&llr);
    assert!(result.success);
    assert_eq!(result.codeword.to_vec(), codeword);
    println!(
        "decoded; {} TEPs processed, distance {:.2}",
        result.num_teps_processed, result.llr_distance
    );
}
