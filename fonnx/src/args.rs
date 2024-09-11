use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about, name = "fonnx")]
pub struct Args {
    pub path: PathBuf,
}
