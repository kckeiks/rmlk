use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Phases land here: config, listener, session router, GPU worker.
    // See server/TODO.md.
    println!("rmlk-server {}", rmlk_server::VERSION);
    Ok(())
}
