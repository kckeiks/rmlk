//! rmlk inference server.

pub mod engine;
pub mod http;
pub mod protocol;
pub mod session;
pub mod worker;

/// Crate version from Cargo.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `true` when built with `--features ort` (or `cuda`).
pub const FEATURE_ORT: bool = cfg!(feature = "ort");

/// `true` when built with `--features cuda`.
pub const FEATURE_CUDA: bool = cfg!(feature = "cuda");

#[cfg(test)]
mod feature_tests {
    #[test]
    fn default_build_excludes_ort_and_cuda() {
        assert!(
            !crate::FEATURE_ORT,
            "default `cargo test -p rmlk-server` must not enable feature `ort`"
        );
        assert!(
            !crate::FEATURE_CUDA,
            "default `cargo test -p rmlk-server` must not enable feature `cuda`"
        );
    }
}
