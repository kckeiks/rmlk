use std::net::SocketAddr;

use anyhow::Result;
use clap::Parser;
use tokio::net::TcpListener;

#[derive(Debug, Parser)]
#[command(name = "infer-server", version = rmlk_server::VERSION)]
struct Args {
    /// Listen address (`host:port`).
    #[arg(long, env = "RMLK_BIND", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let listener = TcpListener::bind(args.bind).await?;
    println!(
        "infer-server {} listening on {}",
        rmlk_server::VERSION,
        listener.local_addr()?
    );
    axum::serve(listener, rmlk_server::http::router()).await?;
    Ok(())
}
