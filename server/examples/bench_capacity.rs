//! Continuous-stream capacity comparison with Dirigo's nemotron_triton_560 profile.
//! See server/docs/capacity.md for methodology and remaining comparison limits.
mod capacity;

use anyhow::{bail, Context, Result};
use capacity::{audio, gpu, metrics::Level, stream};
use clap::Parser;
use serde::Serialize;
use std::{fs, io::Write, path::PathBuf, sync::Arc};

#[derive(Clone, Debug, Parser, Serialize)]
#[command(about = "Ramp and soak continuous realtime streams; write JSON/Markdown reports")]
struct Args {
    #[arg(long, default_value = "ws://127.0.0.1:8080/ws")]
    url: String,
    #[arg(long, default_value_t = 1)]
    start_streams: usize,
    #[arg(long, default_value_t = 32, alias = "clients")]
    max_streams: usize,
    #[arg(long, default_value_t = 1)]
    ramp_step: usize,
    #[arg(long, default_value_t = 30.0)]
    hold_seconds: f64,
    #[arg(long, default_value_t = 120.0)]
    soak_seconds: f64,
    /// One stream before the ramp, excluded from the capacity score (Dirigo default).
    #[arg(long, default_value_t = 8.0)]
    warmup_seconds: f64,
    #[arg(long, default_value_t = 20)]
    frame_ms: u64,
    #[arg(long, default_value_t = 560)]
    server_chunk_ms: usize,
    #[arg(long, default_value_t = 0.95)]
    min_realtime_ratio: f64,
    #[arg(long, default_value_t = 500.0)]
    max_p99_client_e2e_ms: f64,
    #[arg(long, default_value_t = 0.05)]
    max_late_frame_ratio: f64,
    /// Bound queued chunks without slowing the audio clock. Overflow fails the level.
    #[arg(long, default_value_t = 256)]
    max_pending_chunks: usize,
    #[arg(long, default_value_t = 2.0)]
    drain_timeout_seconds: f64,
    #[arg(long, default_value_t = 5.0)]
    open_timeout_seconds: f64,
    /// Flat WAV/PCM16LE directory; otherwise use the checksum-pinned capacity pack.
    #[arg(long)]
    data_dir: Option<PathBuf>,
    /// New directory for config.json, levels.jsonl, summary.json/md, and gpu.csv.
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Operator-supplied backend/model/batch/context details; recorded verbatim.
    #[arg(long, default_value = "")]
    notes: String,
    #[arg(long, default_value_t = 0)]
    gpu_index: u32,
    #[arg(long, default_value_t = 1.0)]
    gpu_sample_interval: f64,
    #[arg(long)]
    no_gpu_sampling: bool,
}

impl Args {
    fn validate(&self) -> Result<()> {
        if !self.warmup_seconds.is_finite()
            || !(0.0..=86400.0).contains(&self.warmup_seconds)
            || (self.warmup_seconds > 0.0
                && self.warmup_seconds * 1000.0 < self.server_chunk_ms as f64)
        {
            bail!("warmup-seconds must be zero or cover at least one chunk, and <= 86400");
        }
        if self.start_streams == 0 || self.max_streams < self.start_streams || self.ramp_step == 0 {
            bail!("require 1 <= start-streams <= max-streams and ramp-step > 0");
        }
        for (name, value) in [
            ("hold-seconds", self.hold_seconds),
            ("soak-seconds", self.soak_seconds),
            ("drain-timeout-seconds", self.drain_timeout_seconds),
            ("open-timeout-seconds", self.open_timeout_seconds),
            ("gpu-sample-interval", self.gpu_sample_interval),
            ("max-p99-client-e2e-ms", self.max_p99_client_e2e_ms),
        ] {
            if !value.is_finite() || value <= 0.0 || value > 86400.0 {
                bail!("{name} must be finite and in (0, 86400]");
            }
        }
        if !(0.0..=1.0).contains(&self.min_realtime_ratio)
            || !(0.0..=1.0).contains(&self.max_late_frame_ratio)
        {
            bail!("ratio gates must be finite and in [0, 1]");
        }
        if self.frame_ms == 0
            || self.frame_ms > 1000
            || self.server_chunk_ms == 0
            || self.server_chunk_ms > 10000
            || self.max_pending_chunks == 0
        {
            bail!(
                "require frame-ms in 1..=1000, server-chunk-ms in 1..=10000, and a nonzero queue"
            );
        }
        if self.hold_seconds * 1000.0 < self.server_chunk_ms as f64
            || self.soak_seconds * 1000.0 < self.server_chunk_ms as f64
        {
            bail!("hold and soak must each cover at least one complete inference chunk");
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct Summary<'a> {
    schema_version: u32,
    status: &'a str,
    config: &'a Args,
    clips: Vec<audio::ClipInfo>,
    ramp_safe_n: usize,
    /// None means no sustained level has been verified, including an interrupted run.
    safe_n: Option<usize>,
    reached_ceiling: bool,
    levels: &'a [Level],
    gpu: gpu::Snapshot,
    server_queue_wait_ms: Option<f64>,
    server_inference_ms: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let code = execute(Args::parse()).await?;
    if code != 0 {
        std::process::exit(code);
    }
    Ok(())
}

async fn execute(args: Args) -> Result<i32> {
    args.validate()?;
    let clips = Arc::new(audio::load(&args)?);
    let run_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    let output = args.output_dir.clone().unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../.cache/rmlk/benchmarks/capacity-{run_id}"))
    });
    if output.exists() {
        bail!(
            "output directory already exists: {} (choose a new directory)",
            output.display()
        );
    }
    fs::create_dir_all(output.parent().context("output directory has no parent")?)?;
    fs::create_dir(&output)?;
    let git = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty());
    fs::write(
        output.join("config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1, "unix_start_ms": run_id, "git_sha": git,
            "git_dirty": dirty, "args": args, "clips": clips.iter().map(|c| &c.info).collect::<Vec<_>>()
        }))?,
    )?;
    let mut journal = fs::File::create(output.join("levels.jsonl"))?;
    let sampler = gpu::Sampler::start(&args, &output)?;
    let mut levels = Vec::new();
    let mut ramp_safe_n = 0;
    let mut safe_n = None;
    println!("Reports: {}", output.display());
    println!(
        "Clip assignment: {}",
        clips
            .iter()
            .map(|c| c.info.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let run = async {
        if args.warmup_seconds > 0.0 {
            sampler.set_phase("warmup", 1);
            let level =
                stream::run_level(&args, clips.clone(), "warmup", 1, args.warmup_seconds).await;
            let failed = level.dropped_streams > 0 || level.unacked_chunks > 0;
            record(&mut journal, &mut levels, level)?;
            if failed {
                return Ok("warmup_failed");
            }
        }
        for n in (args.start_streams..=args.max_streams).step_by(args.ramp_step) {
            sampler.set_phase("ramp", n);
            let level = stream::run_level(&args, clips.clone(), "ramp", n, args.hold_seconds).await;
            record(&mut journal, &mut levels, level)?;
            if !levels.last().unwrap().safe {
                break;
            }
            ramp_safe_n = n;
        }
        if ramp_safe_n > 0 {
            sampler.set_phase("soak", ramp_safe_n);
            let level =
                stream::run_level(&args, clips.clone(), "soak", ramp_safe_n, args.soak_seconds)
                    .await;
            if level.safe {
                safe_n = Some(ramp_safe_n);
            }
            record(&mut journal, &mut levels, level)?;
        }
        Ok::<&str, anyhow::Error>("complete")
    };
    let status = tokio::select! {
        result = run => result?,
        signal = tokio::signal::ctrl_c() => { signal?; "interrupted" }
    };
    let gpu = sampler.finish().await;
    let summary = Summary {
        schema_version: 1,
        status,
        config: &args,
        clips: clips.iter().map(|c| c.info.clone()).collect(),
        ramp_safe_n,
        safe_n,
        reached_ceiling: ramp_safe_n == args.max_streams,
        levels: &levels,
        gpu,
        server_queue_wait_ms: None,
        server_inference_ms: None,
    };
    fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    fs::write(
        output.join("summary.md"),
        capacity::report::markdown(&summary),
    )?;
    println!("ramp_safe_n={ramp_safe_n}  sustained_safe_n={safe_n:?}  status={status}");
    println!("Summary: {}", output.join("summary.md").display());
    Ok(if status != "complete" || safe_n.is_none() {
        2
    } else {
        0
    })
}

fn record(journal: &mut fs::File, levels: &mut Vec<Level>, level: Level) -> Result<()> {
    println!(
        "{} N={} rt={:.4} late={:.3}% e2e_p99={:?}ms safe={} {}",
        level.phase,
        level.streams,
        level.realtime_ratio,
        level.late_frame_ratio * 100.0,
        level.client_e2e.p99_ms,
        level.safe,
        level.unsafe_reasons.join("; ")
    );
    serde_json::to_writer(&mut *journal, &level)?;
    writeln!(journal)?;
    journal.flush()?;
    levels.push(level);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_timing_before_opening_sockets() {
        for flag in [
            "--hold-seconds",
            "--soak-seconds",
            "--frame-ms",
            "--ramp-step",
            "--max-pending-chunks",
        ] {
            let args = Args::parse_from(["test", flag, "0"]);
            assert!(args.validate().is_err(), "{flag}");
        }
        assert!(Args::parse_from(["test", "--hold-seconds", "NaN"])
            .validate()
            .is_err());
        assert!(Args::parse_from(["test"]).validate().is_ok());
    }

    #[tokio::test]
    async fn mock_ramp_and_soak_write_reproducible_reports() {
        let temp = tempfile::tempdir().unwrap();
        let audio_dir = temp.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();
        fs::write(audio_dir.join("b.pcm"), [0, 0, 1, 0]).unwrap();
        fs::write(audio_dir.join("a.pcm"), [0, 0, 2, 0]).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let state = rmlk_server::http::new_app_state();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(rmlk_server::http::serve_with_state(
            listener,
            async {
                let _ = rx.await;
            },
            state.clone(),
        ));
        let output = temp.path().join("report");
        let args = Args::parse_from([
            "test",
            "--max-streams",
            "2",
            "--hold-seconds",
            "0.16",
            "--soak-seconds",
            "0.16",
            "--warmup-seconds",
            "0.16",
            "--server-chunk-ms",
            "40",
            "--min-realtime-ratio",
            "0.5",
            "--no-gpu-sampling",
        ]);
        let args = Args {
            url: format!("ws://{address}/ws"),
            data_dir: Some(audio_dir),
            output_dir: Some(output.clone()),
            ..args
        };
        assert_eq!(execute(args.clone()).await.unwrap(), 0);
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("summary.json")).unwrap()).unwrap();
        assert_eq!(report["ramp_safe_n"], 2);
        assert_eq!(report["safe_n"], 2);
        assert_eq!(report["levels"].as_array().unwrap().len(), 4);
        assert_eq!(report["levels"][3]["phase"], "soak");
        assert_eq!(report["levels"][2]["per_stream"][0]["clip"], "a.pcm");
        assert_eq!(report["levels"][2]["per_stream"][1]["clip"], "b.pcm");
        assert!(report["gpu"]["overall"]["mean_utilization_pct"].is_null());
        assert!(report["server_inference_ms"].is_null());
        for file in ["config.json", "levels.jsonl", "summary.md", "gpu.csv"] {
            assert!(output.join(file).is_file());
        }
        assert_eq!(
            fs::read_to_string(output.join("levels.jsonl"))
                .unwrap()
                .lines()
                .count(),
            4
        );
        assert!(execute(args)
            .await
            .unwrap_err()
            .to_string()
            .contains("already exists"));
        assert_eq!(state.live_session_count(), 0);
        let _ = tx.send(());
        server.await.unwrap().unwrap();
    }
}
