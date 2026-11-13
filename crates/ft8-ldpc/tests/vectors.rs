//! Known-answer tests for the CRC and encoder.
use ft8_ldpc_core::crc::{compute_crc14_from_bits, verify_crc14_packed_message};
use ft8_ldpc_core::fec::CascadedGeneratorMatrix;
use std::path::PathBuf;

fn vectors(file: &str) -> Vec<[String; 3]> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/known-answer")
        .join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "cannot read known-answer vectors {}: {error}",
            path.display()
        )
    });
    text.lines()
        .enumerate()
        .map(|(line_number, line)| {
            let fields: Vec<_> = line.split_whitespace().map(str::to_owned).collect();
            fields.try_into().unwrap_or_else(|fields: Vec<String>| {
                panic!(
                    "{}:{}: expected 3 fields, got {}",
                    path.display(),
                    line_number + 1,
                    fields.len()
                )
            })
        })
        .collect()
}

fn bits(text: &str) -> Vec<bool> {
    text.bytes()
        .map(|byte| match byte {
            b'0' => false,
            b'1' => true,
            _ => panic!("expected a binary digit, got byte {byte:#04x}"),
        })
        .collect()
}

fn pack(bits: &[bool]) -> [u64; 2] {
    let mut words = [0u64; 2];
    for (i, &bit) in bits.iter().enumerate() {
        words[i / 64] |= (bit as u64) << (i % 64);
    }
    words
}

/// The 14-bit remainder of the 77 payload bits with no zero-extension to 82
/// bits: only the 14 flushing zeros follow them.
fn crc_without_zero_extension(payload: &[bool]) -> u16 {
    let mut remainder = 0u16;
    for &bit in payload.iter().chain(&[false; 14]) {
        remainder = (remainder << 1) | bit as u16;
        if remainder >> 14 != 0 {
            remainder ^= 0x2757;
        }
        remainder &= 0x3fff;
    }
    remainder
}

#[test]
fn the_crc_matches_all_known_answer_vectors() {
    let vectors = vectors("crc14.txt");
    assert_eq!(vectors.len(), 301);
    let mut zero_extension_matters = false;
    for (n, fields) in vectors.iter().enumerate() {
        let payload = bits(&fields[0]);
        let crc_bits = bits(&fields[1]);
        assert_eq!((payload.len(), crc_bits.len()), (77, 14));
        let expected: u16 = fields[2].parse().expect("CRC value must be a decimal u16");
        let computed = compute_crc14_from_bits(&payload);
        assert_eq!(computed, expected, "vector {n}");
        // Bit 77 is the most significant CRC bit.
        let msb_first = crc_bits.iter().fold(0u16, |acc, &b| (acc << 1) | b as u16);
        assert_eq!(msb_first, expected, "vector {n}");
        zero_extension_matters |= crc_without_zero_extension(&payload) != expected;

        let mut message = payload.clone();
        message.extend(&crc_bits);
        assert!(verify_crc14_packed_message(pack(&message)), "vector {n}");
        for flipped in 0..91 {
            let mut broken = message.clone();
            broken[flipped] = !broken[flipped];
            assert!(
                !verify_crc14_packed_message(pack(&broken)),
                "vector {n}, bit {flipped}"
            );
        }
    }
    assert!(
        zero_extension_matters,
        "no vector detects a missing zero-extension"
    );
    // Reversing the CRC bit order fails for the vectors.
    let reversed_fail = vectors
        .iter()
        .filter(|fields| {
            let mut message = bits(&fields[0]);
            message.extend(bits(&fields[1]).iter().rev());
            !verify_crc14_packed_message(pack(&message))
        })
        .count();
    assert!(reversed_fail > 250, "{reversed_fail}");
}

#[test]
fn the_encoder_matches_known_answer_vectors_for_every_cascade() {
    let vectors = vectors("ldpc17491.txt");
    assert_eq!(vectors.len(), 400);
    let base = CascadedGeneratorMatrix::new(0).unwrap();
    let cascaded: Vec<_> = (1..=14)
        .map(|c| CascadedGeneratorMatrix::new(c).unwrap())
        .collect();
    for (n, fields) in vectors.iter().enumerate() {
        let payload = bits(&fields[0]);
        let crc = bits(&fields[1]);
        let parity = bits(&fields[2]);
        assert_eq!((payload.len(), crc.len(), parity.len()), (77, 14, 83));
        let mut message = payload.clone();
        message.extend(&crc);
        assert!(verify_crc14_packed_message(pack(&message)), "vector {n}");
        let codeword = base.encode_packed(pack(&message));
        for (i, &bit) in message.iter().enumerate() {
            assert_eq!(
                (codeword[i / 64] >> (i % 64)) & 1 == 1,
                bit,
                "vector {n}, bit {i}"
            );
        }
        for (i, &bit) in parity.iter().enumerate() {
            let n_bit = 91 + i;
            assert_eq!(
                (codeword[n_bit / 64] >> (n_bit % 64)) & 1 == 1,
                bit,
                "vector {n}, parity {i}"
            );
        }
        // Encoding the cascaded input gives the same codeword.
        for (index, generator) in cascaded.iter().enumerate() {
            let c = index + 1;
            let k = 91 - c;
            let input = pack(&message[..k]);
            assert_eq!(
                generator.encode_packed(input),
                codeword,
                "vector {n}, c {c}"
            );
        }
    }
}
