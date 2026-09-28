use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("rmlk-server {}", rmlk_server::VERSION);
    Ok(())
}
