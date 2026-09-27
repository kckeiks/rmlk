//! Multi-op graph integration tests.
//!
//! Expected outputs are produced by `scripts/gen_graph_fixtures.py` and checked
//! in under `runtime/tests/fixtures/`.

mod attention;
mod conv_block;
mod serialize_roundtrip;

use std::path::PathBuf;

fn fixture_bytes(name: &str) -> Vec<u8> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn load_f32(name: &str) -> Vec<f32> {
    let bytes = fixture_bytes(name);
    assert_eq!(bytes.len() % 4, 0, "{name}: not a multiple of 4 bytes");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual={actual:?} expected={expected:?}"
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= eps,
            "mismatch at {i}: actual={a} expected={e} (eps={eps})"
        );
    }
}
