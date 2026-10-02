use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use rmlk_server::http::{ENGINE_MOCK, ENGINE_NEMO, ENGINE_ORT};
use tokio::net::TcpListener;

#[derive(Debug, Clone, Default, ValueEnum)]
enum EngineChoice {
    /// Deterministic mock (default; used by tests).
    #[default]
    Mock,
    /// Nemotron / parakeet-rs + ORT (requires `--features ort`).
    Ort,
    /// In-process NeMo SDK with GGUF model (requires `--features nemo`).
    Nemo,
}

impl EngineChoice {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Mock => ENGINE_MOCK,
            Self::Ort => ENGINE_ORT,
            Self::Nemo => ENGINE_NEMO,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "infer-server", version = rmlk_server::VERSION)]
struct Args {
    /// Listen address (`host:port`).
    #[arg(long, env = "RMLK_BIND", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// Inference backend: `mock`, `ort`, or `nemo`.
    #[arg(long, env = "RMLK_ENGINE", value_enum, default_value_t = EngineChoice::Mock)]
    engine: EngineChoice,

    /// Nemotron ONNX directory (required when `--engine ort`).
    #[arg(long, env = "RMLK_NEMOTRON_MODEL_DIR")]
    model_dir: Option<PathBuf>,
    /// Absolute path to a trusted NeMo ASR SDK shared library.
    #[arg(long, env = "RMLK_NEMO_LIBRARY")]
    nemo_library: Option<PathBuf>,
    /// Nemotron streaming GGUF model file.
    #[arg(long, env = "RMLK_NEMO_MODEL")]
    nemo_model: Option<PathBuf>,
    /// NeMo device: -1 for CPU, or a GPU index (CUDA SDK required).
    #[arg(
        long,
        env = "RMLK_NEMO_DEVICE",
        default_value_t = 0,
        allow_hyphen_values = true
    )]
    nemo_device: i32,
    /// Maximum NeMo sessions; reserves one blocking worker per slot (1..=64).
    #[arg(long, env = "RMLK_NEMO_MAX_SESSIONS", default_value_t = 8)]
    nemo_max_sessions: usize,
    /// Maximum native physical batch size (1 disables native batching).
    #[arg(long, env = "RMLK_NEMO_BATCH_SIZE", default_value_t = 8)]
    nemo_batch_size: u32,
    /// Maximum native batch queue delay in microseconds.
    #[arg(long, env = "RMLK_NEMO_QUEUE_DELAY_US", default_value_t = 5_000)]
    nemo_queue_delay_us: u32,
    /// RNNT lookahead frames; 6 selects the Nemotron 560 ms operating point.
    #[arg(
        long,
        env = "RMLK_NEMO_RIGHT_CONTEXT",
        default_value_t = 6,
        allow_hyphen_values = true
    )]
    nemo_right_context: i32,
}

/// Panic policy: a connection panic is isolated to that connection and logged
/// by the HTTP layer. A panic on the engine worker takes every session with
/// it, so the process logs it and exits non-zero rather than serving with no
/// engine behind it.
#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    rmlk_server::install_panic_hook();

    let args = Args::parse();
    let (state, worker) = if matches!(args.engine, EngineChoice::Nemo) {
        #[cfg(feature = "nemo")]
        {
            use nemo_speech::Device;
            use rmlk_server::engine::NemoConfig;
            let library = args.nemo_library.ok_or_else(|| {
                anyhow::anyhow!("--engine nemo requires --nemo-library / RMLK_NEMO_LIBRARY")
            })?;
            let model = args.nemo_model.ok_or_else(|| {
                anyhow::anyhow!("--engine nemo requires --nemo-model / RMLK_NEMO_MODEL")
            })?;
            let mut config = NemoConfig::new(library, model);
            config.device = match args.nemo_device {
                -1 => Device::Cpu,
                index if index >= 0 => Device::Gpu(index as u32),
                _ => bail!("--nemo-device must be -1 (CPU) or a nonnegative GPU index"),
            };
            config.max_sessions = args.nemo_max_sessions;
            config.batch_size = args.nemo_batch_size;
            config.queue_delay_us = args.nemo_queue_delay_us;
            config.right_context = args.nemo_right_context;
            // SAFETY: the operator selects a trusted SDK built from the pinned
            // upstream revision, as documented in docs/nemo.md.
            unsafe { rmlk_server::http::app_state_from_nemo_config(&config) }?
        }
        #[cfg(not(feature = "nemo"))]
        bail!("engine `nemo` requires building with `--features nemo`");
    } else {
        rmlk_server::http::app_state_from_config(args.engine.as_str(), args.model_dir.as_deref())?
    };
    let listener = TcpListener::bind(args.bind).await?;
    println!(
        "infer-server {} listening on {} (engine={})",
        rmlk_server::VERSION,
        listener.local_addr()?,
        state.engine_name()
    );

    tokio::select! {
        served = rmlk_server::http::serve_with_state(
            listener,
            rmlk_server::http::shutdown_on_ctrl_c(),
            state,
        ) => served,
        joined = worker => match joined {
            Ok(()) => bail!("engine worker stopped while the server was still running"),
            Err(err) if err.is_panic() => {
                let payload = err.into_panic();
                bail!(
                    "engine worker panicked: {}",
                    rmlk_server::panic_message(payload.as_ref())
                )
            }
            Err(err) => bail!("engine worker task failed: {err}"),
        },
    }
}
