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

/// Best-effort text of a panic payload for logging.
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else {
        "non-string panic payload"
    }
}

/// Log every panic at error level with its thread, location, and a backtrace.
///
/// Replaces the default hook, which only prints the message to stderr and
/// only includes a backtrace when `RUST_BACKTRACE` is set. Call once at
/// process start after the logger is initialized.
pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let thread = std::thread::current();
        let thread = thread.name().unwrap_or("unnamed");
        let location = info
            .location()
            .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
            .unwrap_or_else(|| "unknown location".to_owned());
        let backtrace = std::backtrace::Backtrace::force_capture();
        log::error!(
            "panic in thread `{thread}` at {location}: {}\n{backtrace}",
            panic_message(info.payload())
        );
    }));
}

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
