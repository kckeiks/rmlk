use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use rmlk_server::http::{ENGINE_MOCK, ENGINE_ORT};
use tokio::net::TcpListener;

#[derive(Debug, Clone, Default, ValueEnum)]
enum EngineChoice {
    /// Deterministic mock (default; used by tests).
    #[default]
    Mock,
    /// Nemotron / parakeet-rs + ORT (requires `--features ort`).
    Ort,
}

impl EngineChoice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Mock => ENGINE_MOCK,
            Self::Ort => ENGINE_ORT,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "infer-server", version = rmlk_server::VERSION)]
struct Args {
    /// Listen address (`host:port`).
    #[arg(long, env = "RMLK_BIND", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// Inference backend: `mock` (default) or `ort`.
    #[arg(long, env = "RMLK_ENGINE", value_enum, default_value_t = EngineChoice::Mock)]
    engine: EngineChoice,

    /// Nemotron ONNX directory (required when `--engine ort`).
    #[arg(long, env = "RMLK_NEMOTRON_MODEL_DIR")]
    model_dir: Option<PathBuf>,
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
    let (state, worker) =
        rmlk_server::http::app_state_from_config(args.engine.as_str(), args.model_dir.as_deref())?;
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
