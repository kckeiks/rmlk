//! N=1 chunk e2e latency client against a running `rmlk-server`.
//!
//! Measures the current single-stream path (including the interim shared
//! engine mutex). Default pacing is realtime (one model chunk per 560 ms of
//! wall clock), matching Dirigo capacity-style loadgen. Not for N>1 load
//! numbers until the Phase 7 scheduler lands.
//!
//! ```text
//! # generate Dirigo-identical capacity_load WAVs (needs numpy):
//! python3 server/scripts/gen_capacity_load.py --write-manifest
//!
//! # mock smoke:
//! cargo run -p rmlk-server -- --bind 127.0.0.1:8080
//! cargo run -p rmlk-server --example bench_latency -- \
//!   --url ws://127.0.0.1:8080/ws --silence-chunks 8
//!
//! # capacity clip (realtime pacing):
//! cargo run -p rmlk-server --example bench_latency -- \
//!   --url ws://127.0.0.1:8080/ws --clip loop_speechish
//! ```

use std::env;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use futures_util::{SinkExt, StreamExt};
use rmlk_server::protocol::{ClientFrame, ServerFrame};
use tokio::time::{sleep, timeout};
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Nemotron / Parakeet streaming step: 560 ms mono @ 16 kHz.
const CHUNK_SAMPLES: usize = 8960;
const SAMPLE_RATE_HZ: u32 = 16_000;
const TESTDATA_CACHE_ENV: &str = "RMLK_TESTDATA_CACHE";

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum Pace {
    /// Send each model chunk on a wall-clock 560 ms schedule (Dirigo-style).
    #[default]
    Realtime,
    /// Send as fast as Partial waits allow (mutex-path smoke only).
    Asap,
}

#[derive(Debug, Parser)]
#[command(name = "bench_latency", about = "N=1 WS chunk e2e latency against rmlk-server")]
struct Args {
    /// WebSocket URL (`ws://host:port/ws`).
    #[arg(long, default_value = "ws://127.0.0.1:8080/ws")]
    url: String,

    /// Capacity / testdata clip id (e.g. `loop_speechish`). Resolves
    /// `{RMLK_TESTDATA_CACHE|/.cache/rmlk/testdata}/{id}.wav`.
    #[arg(long, conflicts_with_all = ["wav", "silence_chunks"])]
    clip: Option<String>,

    /// 16 kHz mono PCM16 WAV to stream.
    #[arg(long, conflicts_with_all = ["clip", "silence_chunks"])]
    wav: Option<PathBuf>,

    /// Synthetic zeroed chunks (mock smoke; no WAV file).
    #[arg(long, conflicts_with_all = ["clip", "wav"])]
    silence_chunks: Option<usize>,

    /// Audio send pacing.
    #[arg(long, value_enum, default_value_t = Pace::Realtime)]
    pace: Pace,

    /// Discard this many full utterances before measuring.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Measured utterance repeats after warmup.
    #[arg(long, default_value_t = 3)]
    iters: usize,

    /// How long to wait for a Partial after each Audio (ms). Chunks with no
    /// Partial within this window are skipped for Partial percentiles (ORT may
    /// emit empty steps). Under realtime pacing the wait is also capped by the
    /// time remaining until the next chunk slot.
    #[arg(long, default_value_t = 5_000)]
    partial_timeout_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let chunks = load_chunks(&args)?;
    if chunks.is_empty() {
        bail!("no audio chunks to send");
    }

    let chunk_secs = CHUNK_SAMPLES as f64 / f64::from(SAMPLE_RATE_HZ);
    let audio_secs = chunks.len() as f64 * chunk_secs;
    println!(
        "bench_latency N=1 url={} pace={:?} chunks={} audio={audio_secs:.2}s warmup={} iters={}",
        args.url,
        args.pace,
        chunks.len(),
        args.warmup,
        args.iters
    );

    for i in 0..args.warmup {
        let stats = run_utterance(&args, &chunks).await?;
        println!(
            "warmup[{i}]: wall={:.1}ms final={:.1}ms partials={} rt_ratio={:.4} late={}",
            stats.wall_ms,
            stats.final_ms,
            stats.partial_ms.len(),
            stats.realtime_ratio,
            stats.late_chunks
        );
    }

    let mut all_partial_ms = Vec::new();
    let mut final_ms = Vec::new();
    let mut wall_ms = Vec::new();
    let mut rt_ratios = Vec::new();

    for i in 0..args.iters {
        let stats = run_utterance(&args, &chunks).await?;
        println!(
            "iter[{i}]: wall={:.1}ms final={:.1}ms partials={} rt_ratio={:.4} late={} rtf={:.3}",
            stats.wall_ms,
            stats.final_ms,
            stats.partial_ms.len(),
            stats.realtime_ratio,
            stats.late_chunks,
            (stats.wall_ms / 1000.0) / audio_secs
        );
        all_partial_ms.extend(stats.partial_ms);
        final_ms.push(stats.final_ms);
        wall_ms.push(stats.wall_ms);
        rt_ratios.push(stats.realtime_ratio);
    }

    println!();
    println!("=== N=1 summary (mutex path; local only) ===");
    print_percentile_block("partial_e2e_ms", &mut all_partial_ms);
    print_percentile_block("final_e2e_ms", &mut final_ms);
    print_percentile_block("utterance_wall_ms", &mut wall_ms);
    if let Some(p50_rt) = percentile_sorted(&mut rt_ratios.clone(), 50.0) {
        println!("realtime_ratio_p50={p50_rt:.4}  (~1.0 means keep-up under paced send)");
    }
    if let Some(p50_wall) = percentile_sorted(&mut wall_ms.clone(), 50.0) {
        println!(
            "rtf_from_wall_p50={:.3}  (wall_p50_s / audio_s)",
            (p50_wall / 1000.0) / audio_secs
        );
    }
    Ok(())
}

struct UtteranceStats {
    partial_ms: Vec<f64>,
    final_ms: f64,
    wall_ms: f64,
    realtime_ratio: f64,
    late_chunks: u64,
}

async fn run_utterance(args: &Args, chunks: &[Vec<i16>]) -> Result<UtteranceStats> {
    let (mut ws, _) = tokio_tungstenite::connect_async(&args.url)
        .await
        .with_context(|| format!("connect {}", args.url))?;

    let wall_start = Instant::now();
    send_frame(&mut ws, &ClientFrame::Open).await?;
    match recv_frame(&mut ws).await? {
        ServerFrame::OpenAck { .. } => {}
        other => bail!("expected OpenAck, got {other:?}"),
    }

    let chunk_dur = Duration::from_secs_f64(CHUNK_SAMPLES as f64 / f64::from(SAMPLE_RATE_HZ));
    let partial_timeout = Duration::from_millis(args.partial_timeout_ms);
    let mut partial_ms = Vec::new();
    let mut late_chunks = 0u64;
    let pace_origin = Instant::now();

    for (idx, chunk) in chunks.iter().enumerate() {
        if matches!(args.pace, Pace::Realtime) {
            late_chunks += wait_next_chunk_slot(pace_origin, idx, chunk_dur).await;
        }

        let t0 = Instant::now();
        send_frame(
            &mut ws,
            &ClientFrame::Audio {
                pcm16: chunk.clone(),
            },
        )
        .await?;

        let wait_budget = match args.pace {
            Pace::Realtime => {
                let next_slot = pace_origin + chunk_dur * (idx as u32 + 2);
                let until_next = next_slot.saturating_duration_since(Instant::now());
                until_next.min(partial_timeout).max(Duration::from_millis(1))
            }
            Pace::Asap => partial_timeout,
        };

        // Drain one response window. ORT may send nothing (empty step); mock
        // always sends a Partial. Timed-out windows are omitted from percentiles.
        match timeout(wait_budget, recv_frame(&mut ws)).await {
            Ok(Ok(ServerFrame::Partial { .. })) => {
                partial_ms.push(duration_ms(t0.elapsed()));
            }
            Ok(Ok(ServerFrame::Final { text })) => {
                bail!("unexpected Final during audio (chunk {idx}): {text}");
            }
            Ok(Ok(other)) => bail!("unexpected frame during audio (chunk {idx}): {other:?}"),
            Ok(Err(err)) => return Err(err),
            Err(_) => {}
        }
    }

    let audio_wall = pace_origin.elapsed();
    let audio_secs = chunks.len() as f64 * chunk_dur.as_secs_f64();
    let realtime_ratio = if audio_wall.as_secs_f64() > 0.0 {
        audio_secs / audio_wall.as_secs_f64()
    } else {
        1.0
    };

    let t_final = Instant::now();
    send_frame(&mut ws, &ClientFrame::Finalize).await?;
    let final_ms = loop {
        match recv_frame(&mut ws).await? {
            ServerFrame::Partial { .. } => {}
            ServerFrame::Final { .. } => break duration_ms(t_final.elapsed()),
            other => bail!("unexpected frame after Finalize: {other:?}"),
        }
    };

    let _ = timeout(Duration::from_secs(2), ws.next()).await;
    let _ = ws.close(None).await;

    Ok(UtteranceStats {
        partial_ms,
        final_ms,
        wall_ms: duration_ms(wall_start.elapsed()),
        realtime_ratio,
        late_chunks,
    })
}

/// Sleep until the wall-clock slot for chunk `idx` (0-based). Returns 1 if late.
async fn wait_next_chunk_slot(origin: Instant, idx: usize, chunk_dur: Duration) -> u64 {
    let target = origin + chunk_dur * (idx as u32 + 1);
    let now = Instant::now();
    if now < target {
        sleep(target - now).await;
        0
    } else {
        1
    }
}

fn load_chunks(args: &Args) -> Result<Vec<Vec<i16>>> {
    if let Some(n) = args.silence_chunks {
        return Ok((0..n).map(|_| vec![0i16; CHUNK_SAMPLES]).collect());
    }
    let path = if let Some(clip) = &args.clip {
        resolve_clip_wav(clip)?
    } else if let Some(wav) = &args.wav {
        wav.clone()
    } else {
        bail!("pass --clip ID, --wav PATH, or --silence-chunks N");
    };
    let pcm = load_wav_pcm16(&path)?;
    Ok(pcm_chunks(&pcm, CHUNK_SAMPLES))
}

fn resolve_clip_wav(clip_id: &str) -> Result<PathBuf> {
    let dir = testdata_dir();
    let path = dir.join(format!("{clip_id}.wav"));
    if !path.is_file() {
        bail!(
            "clip WAV not found: {} (run server/scripts/gen_capacity_load.py or set {TESTDATA_CACHE_ENV})",
            path.display()
        );
    }
    Ok(path)
}

fn testdata_dir() -> PathBuf {
    if let Ok(dir) = env::var(TESTDATA_CACHE_ENV) {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("server crate has parent")
        .join(".cache/rmlk/testdata")
}

fn load_wav_pcm16(path: &Path) -> Result<Vec<i16>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("open WAV {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        bail!("WAV must be mono (got {} channels)", spec.channels);
    }
    if spec.sample_rate != SAMPLE_RATE_HZ {
        bail!(
            "WAV must be {SAMPLE_RATE_HZ} Hz (got {})",
            spec.sample_rate
        );
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        bail!("WAV must be PCM16");
    }
    reader
        .samples::<i16>()
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("read WAV samples")
}

fn pcm_chunks(pcm: &[i16], chunk_samples: usize) -> Vec<Vec<i16>> {
    pcm.chunks(chunk_samples)
        .map(|chunk| {
            let mut padded = chunk.to_vec();
            if padded.len() < chunk_samples {
                padded.resize(chunk_samples, 0);
            }
            padded
        })
        .collect()
}

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn send_frame(ws: &mut WsStream, frame: &ClientFrame) -> Result<()> {
    let bytes = frame.encode().context("encode client frame")?;
    ws.send(WsMessage::Binary(bytes.into()))
        .await
        .context("send ws binary")?;
    Ok(())
}

async fn recv_frame(ws: &mut WsStream) -> Result<ServerFrame> {
    let msg = ws
        .next()
        .await
        .context("ws stream ended")?
        .context("ws message error")?;
    let WsMessage::Binary(bytes) = msg else {
        bail!("expected binary server frame, got {msg:?}");
    };
    ServerFrame::decode(&bytes).context("decode server frame")
}

fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn print_percentile_block(label: &str, values: &mut [f64]) {
    if values.is_empty() {
        println!("{label}: (no samples)");
        return;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile_sorted(values, 50.0).unwrap();
    let p95 = percentile_sorted(values, 95.0).unwrap();
    let p99 = percentile_sorted(values, 99.0).unwrap();
    let n = values.len();
    println!("{label}: n={n} p50={p50:.1} p95={p95:.1} p99={p99:.1}");
}

fn percentile_sorted(sorted: &[f64], pct: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = ((pct / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    Some(sorted[rank.min(sorted.len() - 1)])
}
