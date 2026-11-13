use ft8_ldpc::fast_min_sum::Decoder;
use ft8_ldpc_core::crc::verify_codeword_crc14;
use ft8_ldpc_core::fec::encode_174_91;
use ft8_ldpc_core::protocol::{N, PAYLOAD_BITS};

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

    // With the default threshold of zero, success requires every parity check to pass.
    // The CRC is the caller's check.
    assert!(result.success);
    assert!(verify_codeword_crc14(&result.codeword));
    assert_eq!(result.codeword, codeword);
    println!(
        "decoded in {} iterations, {} hard decisions changed",
        result.iterations, result.hard_error_count
    );
}
