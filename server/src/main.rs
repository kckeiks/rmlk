use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
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

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let state =
        rmlk_server::http::app_state_from_config(args.engine.as_str(), args.model_dir.as_deref())?;
    let listener = TcpListener::bind(args.bind).await?;
    println!(
        "infer-server {} listening on {} (engine={})",
        rmlk_server::VERSION,
        listener.local_addr()?,
        state.engine_name()
    );
    rmlk_server::http::serve_with_state(listener, rmlk_server::http::shutdown_on_ctrl_c(), state)
        .await?;
    Ok(())
}
