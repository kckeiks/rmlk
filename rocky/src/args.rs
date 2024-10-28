use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about, name = "fonnx")]
pub struct Args {
    #[command(subcommand)]
    pub cmd: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Find { path: PathBuf, target: String },
    Transform { path: PathBuf },
}
