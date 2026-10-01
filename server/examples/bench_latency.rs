//! Chunk e2e latency client against a running `rmlk-server`.
//!
//! Starts at one stream and adds one stream per ramp stage. A stage is safe
//! under the same gate as Dirigo: realtime ratio at least 0.95, client e2e
//! p99 at most 500 ms, and at most 5 percent late chunks. After the first
//! unsafe stage it soaks at the last N that passed. Per-chunk latency is
//! `client_e2e`: from sending `Audio` to receiving `AudioProcessed`.
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
//! # find safe N up to 8, then soak:
//! cargo run -p rmlk-server --example bench_latency -- \
//!   --url ws://127.0.0.1:8080/ws --clip loop_speechish --clients 8
//! ```

use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use futures_util::{SinkExt, StreamExt};
use rmlk_server::protocol::{ClientFrame, ServerFrame};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Nemotron / Parakeet streaming step: 560 ms mono @ 16 kHz.
const CHUNK_SAMPLES: usize = 8960;
const SAMPLE_RATE_HZ: u32 = 16_000;
const TESTDATA_CACHE_ENV: &str = "RMLK_TESTDATA_CACHE";
const DEFAULT_MAX_CLIENTS: usize = 1;
const DEFAULT_STAGE_ITERS: usize = 1;
const DEFAULT_SOAK_ITERS: usize = 3;
const DEFAULT_WARMUP: usize = 1;
const DEFAULT_STEP_TIMEOUT_MS: u64 = 30_000;
/// Same defaults as Dirigo `RampConfig` in `benchmarks/common/ramp.py`.
const MIN_REALTIME_RATIO: f64 = 0.95;
const MAX_P99_CLIENT_E2E_MS: f64 = 500.0;
const MAX_LATE_CHUNK_RATIO: f64 = 0.05;

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum Pace {
    /// Send each model chunk on a wall-clock 560 ms schedule (Dirigo-style).
    #[default]
    Realtime,
    /// Send the next chunk as soon as the previous one is processed.
    Asap,
}

#[derive(Debug, Parser)]
#[command(
    name = "bench_latency",
    about = "Ramp-then-soak WS chunk e2e latency against rmlk-server"
)]
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

    /// Maximum concurrent streams to try. Adds one stream per ramp stage.
    #[arg(long, default_value_t = DEFAULT_MAX_CLIENTS)]
    clients: usize,

    /// Audio send pacing.
    #[arg(long, value_enum, default_value_t = Pace::Realtime)]
    pace: Pace,

    /// Utterances each live client runs at every ramp stage.
    #[arg(long, default_value_t = DEFAULT_STAGE_ITERS)]
    stage_iters: usize,

    /// Discard this many soak utterances (all live clients) before measuring.
    #[arg(long, default_value_t = DEFAULT_WARMUP)]
    warmup: usize,

    /// Measured soak utterances per live client after ramp.
    #[arg(long, visible_alias = "iters", default_value_t = DEFAULT_SOAK_ITERS)]
    soak_iters: usize,

    /// Ramp all the way to `--clients` and soak there, even if a stage fails
    /// the Dirigo safety gate.
    #[arg(long, visible_alias = "no-stop-on-late", default_value_t = false)]
    no_stop_on_unsafe: bool,

    /// How long to wait for AudioProcessed after each Audio (ms).
    #[arg(long, default_value_t = DEFAULT_STEP_TIMEOUT_MS)]
    step_timeout_ms: u64,

    /// Minimum send-clock realtime ratio (Dirigo `min_realtime_ratio`).
    #[arg(long, default_value_t = MIN_REALTIME_RATIO)]
    min_realtime_ratio: f64,

    /// Maximum client e2e p99 in ms (Dirigo `max_p99_client_e2e_ms`).
    #[arg(long, default_value_t = MAX_P99_CLIENT_E2E_MS)]
    max_p99_client_e2e_ms: f64,

    /// Maximum late-chunk ratio (Dirigo `max_late_frame_ratio`).
    #[arg(long, default_value_t = MAX_LATE_CHUNK_RATIO)]
    max_late_frame_ratio: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Arc::new(Args::parse());
    if args.clients == 0 {
        bail!("--clients must be at least 1");
    }
    if args.stage_iters == 0 {
        bail!("--stage-iters must be at least 1");
    }
    let chunks = Arc::new(load_chunks(&args)?);
    if chunks.is_empty() {
        bail!("no audio chunks to send");
    }

    let chunk_secs = CHUNK_SAMPLES as f64 / f64::from(SAMPLE_RATE_HZ);
    let audio_secs = chunks.len() as f64 * chunk_secs;
    let gate = SafetyGate {
        min_realtime_ratio: args.min_realtime_ratio,
        max_p99_client_e2e_ms: args.max_p99_client_e2e_ms,
        max_late_chunk_ratio: args.max_late_frame_ratio,
    };
    let stop_on_unsafe = !args.no_stop_on_unsafe;
    println!(
        "bench_latency max_n={} url={} pace={:?} chunks={} audio={audio_secs:.2}s stage_iters={} soak_iters={} warmup={} stop_on_unsafe={stop_on_unsafe} gate=rt>={:.2} e2e_p99<={:.0}ms late_ratio<={:.2}",
        args.clients,
        args.url,
        args.pace,
        chunks.len(),
        args.stage_iters,
        args.soak_iters,
        args.warmup,
        gate.min_realtime_ratio,
        gate.max_p99_client_e2e_ms,
        gate.max_late_chunk_ratio
    );

    let mut live: Vec<ClientWorker> = Vec::new();
    let mut last_safe_n = 0usize;
    for n in 1..=args.clients {
        live.push(spawn_client(n - 1, Arc::clone(&args), Arc::clone(&chunks)));
        let mut stage = run_stage(&live, args.stage_iters, "ramp", audio_secs).await?;
        let reasons = evaluate_safety(&stage, &gate, audio_secs);
        print_stage_summary("ramp", n, &mut stage, audio_secs, &reasons);
        if reasons.is_empty() {
            last_safe_n = n;
            continue;
        }
        if !stop_on_unsafe {
            continue;
        }
        if n > 1 {
            let extra = live.pop().expect("just spawned the unsafe client");
            extra.shutdown().await?;
            println!("first unsafe at N={n}; soaking at N={}", n - 1);
        } else {
            println!("unsafe at N=1; soaking at N=1");
        }
        break;
    }

    let soak_n = live.len();
    if args.warmup > 0 {
        let _ = run_stage(&live, args.warmup, "soak-warmup", audio_secs).await?;
    }
    if args.soak_iters > 0 {
        let mut soak = run_stage(&live, args.soak_iters, "soak", audio_secs).await?;
        let soak_reasons = evaluate_safety(&soak, &gate, audio_secs);
        print_stage_summary("soak", soak_n, &mut soak, audio_secs, &soak_reasons);
    }

    if last_safe_n == 0 {
        println!("safe_n=none");
    } else {
        println!("safe_n={last_safe_n}");
    }

    for client in live {
        client.shutdown().await?;
    }
    Ok(())
}

/// Samples gathered across every measured utterance of one or more clients.
#[derive(Default)]
struct ClientStats {
    client_e2e_ms: Vec<f64>,
    final_ms: Vec<f64>,
    wall_ms: Vec<f64>,
    rt_ratios: Vec<f64>,
    audio_wall_secs: f64,
    late_chunks: u64,
    chunks_sent: u64,
    partial_count: usize,
}

impl ClientStats {
    fn merge_utterance(&mut self, stats: UtteranceStats) {
        self.chunks_sent += stats.client_e2e_ms.len() as u64;
        self.client_e2e_ms.extend(stats.client_e2e_ms);
        self.final_ms.push(stats.final_ms);
        self.wall_ms.push(stats.wall_ms);
        self.rt_ratios.push(stats.realtime_ratio);
        self.audio_wall_secs += stats.audio_wall_secs;
        self.late_chunks += stats.late_chunks;
        self.partial_count += stats.partial_count;
    }
}

/// Dirigo `evaluate_safety` thresholds that we can score from this bench.
struct SafetyGate {
    min_realtime_ratio: f64,
    max_p99_client_e2e_ms: f64,
    max_late_chunk_ratio: f64,
}

/// Returns Dirigo-style unsafe reasons. Empty means the stage is safe.
fn evaluate_safety(stats: &ClientStats, gate: &SafetyGate, audio_secs: f64) -> Vec<String> {
    let mut reasons = Vec::new();
    if let Some(rt) = overall_realtime_ratio(stats, audio_secs) {
        if rt < gate.min_realtime_ratio {
            reasons.push(format!(
                "realtime_ratio {rt:.3} < {}",
                gate.min_realtime_ratio
            ));
        }
    }
    let mut e2e = stats.client_e2e_ms.clone();
    e2e.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if let Some(p99) = percentile_sorted(&e2e, 99.0) {
        if p99 > gate.max_p99_client_e2e_ms {
            reasons.push(format!(
                "client_e2e_p99_ms {p99:.1} > {}",
                gate.max_p99_client_e2e_ms
            ));
        }
    }
    if stats.chunks_sent > 0 {
        let late_ratio = stats.late_chunks as f64 / stats.chunks_sent as f64;
        if late_ratio > gate.max_late_chunk_ratio {
            reasons.push(format!(
                "late_frame_ratio {late_ratio:.3} > {}",
                gate.max_late_chunk_ratio
            ));
        }
    }
    reasons
}

fn overall_realtime_ratio(stats: &ClientStats, audio_secs: f64) -> Option<f64> {
    if stats.rt_ratios.is_empty() || stats.audio_wall_secs <= 0.0 {
        return None;
    }
    let total_audio = audio_secs * stats.rt_ratios.len() as f64;
    Some(total_audio / stats.audio_wall_secs)
}

enum Work {
    Utterance {
        reply: oneshot::Sender<Result<UtteranceStats>>,
    },
    Shutdown,
}

struct ClientWorker {
    idx: usize,
    tx: mpsc::Sender<Work>,
    task: JoinHandle<Result<()>>,
}

impl ClientWorker {
    async fn shutdown(self) -> Result<()> {
        let _ = self.tx.send(Work::Shutdown).await;
        self.task.await.context("client task panicked")??;
        Ok(())
    }
}

fn spawn_client(idx: usize, args: Arc<Args>, chunks: Arc<Vec<Vec<i16>>>) -> ClientWorker {
    let (tx, mut rx) = mpsc::channel(1);
    let task = tokio::spawn(async move {
        while let Some(work) = rx.recv().await {
            match work {
                Work::Utterance { reply } => {
                    let result = run_utterance(&args, &chunks).await;
                    if reply.send(result).is_err() {
                        break;
                    }
                }
                Work::Shutdown => break,
            }
        }
        Ok(())
    });
    ClientWorker { idx, tx, task }
}

/// All live clients run `iters` utterances in lockstep. Each iteration starts
/// every client before waiting, so the stage is actually concurrent.
async fn run_stage(
    live: &[ClientWorker],
    iters: usize,
    phase: &str,
    audio_secs: f64,
) -> Result<ClientStats> {
    let mut total = ClientStats::default();
    for i in 0..iters {
        let mut replies = Vec::with_capacity(live.len());
        for client in live {
            let (reply_tx, reply_rx) = oneshot::channel();
            client
                .tx
                .send(Work::Utterance { reply: reply_tx })
                .await
                .with_context(|| format!("c{} worker stopped", client.idx))?;
            replies.push((client.idx, reply_rx));
        }
        for (client_idx, reply_rx) in replies {
            let stats = reply_rx
                .await
                .unwrap_or_else(|_| panic!("c{client_idx} worker dropped the utterance reply"))?;
            println!(
                "c{client_idx} {phase}[{i}]: wall={:.1}ms final={:.1}ms steps={} partials={} rt_ratio={:.4} late={} rtf={:.3}",
                stats.wall_ms,
                stats.final_ms,
                stats.client_e2e_ms.len(),
                stats.partial_count,
                stats.realtime_ratio,
                stats.late_chunks,
                (stats.wall_ms / 1000.0) / audio_secs
            );
            total.merge_utterance(stats);
        }
    }
    Ok(total)
}

fn print_stage_summary(
    kind: &str,
    n: usize,
    stats: &mut ClientStats,
    audio_secs: f64,
    reasons: &[String],
) {
    println!();
    println!("=== {kind} N={n} (local only) ===");
    print_percentile_block("client_e2e_ms", &mut stats.client_e2e_ms);
    print_percentile_block("final_e2e_ms", &mut stats.final_ms);
    print_percentile_block("utterance_wall_ms", &mut stats.wall_ms);
    stats.rt_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let late_ratio = if stats.chunks_sent == 0 {
        0.0
    } else {
        stats.late_chunks as f64 / stats.chunks_sent as f64
    };
    if let Some(rt) = overall_realtime_ratio(stats, audio_secs) {
        let safe = if reasons.is_empty() { "yes" } else { "no" };
        println!(
            "realtime_ratio={rt:.4}  late_chunks={}  late_ratio={late_ratio:.3}  safe={safe}",
            stats.late_chunks
        );
    }
    if let Some(p50_wall) = percentile_sorted(&stats.wall_ms, 50.0) {
        println!(
            "rtf_from_wall_p50={:.3}  (wall_p50_s / audio_s)",
            (p50_wall / 1000.0) / audio_secs
        );
    }
    println!("partials={}", stats.partial_count);
    if reasons.is_empty() {
        println!("unsafe_reasons: (none)");
    } else {
        println!("unsafe_reasons: {}", reasons.join("; "));
    }
    println!();
}

struct UtteranceStats {
    client_e2e_ms: Vec<f64>,
    partial_count: usize,
    final_ms: f64,
    wall_ms: f64,
    realtime_ratio: f64,
    audio_wall_secs: f64,
    late_chunks: u64,
}

async fn run_utterance(args: &Args, chunks: &[Vec<i16>]) -> Result<UtteranceStats> {
    // Nagle off: with it on, the server's back-to-back Partial and
    // AudioProcessed writes wait on this client's delayed ACK and every step
    // measures about 40 ms too long.
    let disable_nagle = true;
    let (mut ws, _) = tokio_tungstenite::connect_async_with_config(&args.url, None, disable_nagle)
        .await
        .with_context(|| format!("connect {}", args.url))?;

    let wall_start = Instant::now();
    send_frame(&mut ws, &ClientFrame::Open).await?;
    match recv_frame(&mut ws).await? {
        ServerFrame::OpenAck { .. } => {}
        other => bail!("expected OpenAck, got {other:?}"),
    }

    let chunk_dur = Duration::from_secs_f64(CHUNK_SAMPLES as f64 / f64::from(SAMPLE_RATE_HZ));
    let step_timeout = Duration::from_millis(args.step_timeout_ms);
    let mut client_e2e_ms = Vec::with_capacity(chunks.len());
    let mut partial_count = 0usize;
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

        let (partials, e2e) = timeout(step_timeout, drain_until_processed(&mut ws, t0))
            .await
            .with_context(|| format!("AudioProcessed timeout on chunk {idx}"))??;
        partial_count += partials;
        client_e2e_ms.push(e2e);
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
            ServerFrame::Partial { .. } | ServerFrame::AudioProcessed => {}
            ServerFrame::Final { .. } => break duration_ms(t_final.elapsed()),
            other => bail!("unexpected frame after Finalize: {other:?}"),
        }
    };

    let _ = timeout(Duration::from_secs(2), ws.next()).await;
    let _ = ws.close(None).await;

    Ok(UtteranceStats {
        client_e2e_ms,
        partial_count,
        final_ms,
        wall_ms: duration_ms(wall_start.elapsed()),
        realtime_ratio,
        audio_wall_secs: audio_wall.as_secs_f64(),
        late_chunks,
    })
}

/// Drain Partials until AudioProcessed. Returns (partial count, e2e ms from `t0`).
async fn drain_until_processed(ws: &mut WsStream, t0: Instant) -> Result<(usize, f64)> {
    let mut partials = 0usize;
    loop {
        match recv_frame(ws).await? {
            ServerFrame::Partial { .. } => partials += 1,
            ServerFrame::AudioProcessed => return Ok((partials, duration_ms(t0.elapsed()))),
            ServerFrame::Final { text } => {
                bail!("unexpected Final while waiting for AudioProcessed: {text}")
            }
            other => bail!("unexpected frame while waiting for AudioProcessed: {other:?}"),
        }
    }
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
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open WAV {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        bail!("WAV must be mono (got {} channels)", spec.channels);
    }
    if spec.sample_rate != SAMPLE_RATE_HZ {
        bail!("WAV must be {SAMPLE_RATE_HZ} Hz (got {})", spec.sample_rate);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> SafetyGate {
        SafetyGate {
            min_realtime_ratio: MIN_REALTIME_RATIO,
            max_p99_client_e2e_ms: MAX_P99_CLIENT_E2E_MS,
            max_late_chunk_ratio: MAX_LATE_CHUNK_RATIO,
        }
    }

    fn stats_with(
        e2e_ms: Vec<f64>,
        late_chunks: u64,
        utterance_audio_secs: f64,
        audio_wall_secs: f64,
    ) -> ClientStats {
        let chunks_sent = e2e_ms.len() as u64;
        ClientStats {
            client_e2e_ms: e2e_ms,
            final_ms: vec![1.0],
            wall_ms: vec![audio_wall_secs * 1000.0],
            rt_ratios: vec![utterance_audio_secs / audio_wall_secs],
            audio_wall_secs,
            late_chunks,
            chunks_sent,
            partial_count: 0,
        }
    }

    #[test]
    fn dirigo_gate_rejects_p99_over_500() {
        let stats = stats_with(vec![400.0, 538.0], 0, 6.16, 6.16);
        let reasons = evaluate_safety(&stats, &gate(), 6.16);
        assert!(
            reasons.iter().any(|r| r.contains("client_e2e_p99_ms")),
            "{reasons:?}"
        );
    }

    #[test]
    fn dirigo_gate_allows_five_percent_late() {
        let e2e = vec![100.0; 20];
        let stats = stats_with(e2e, 1, 6.16, 6.16);
        let reasons = evaluate_safety(&stats, &gate(), 6.16);
        assert!(reasons.is_empty(), "{reasons:?}");
    }

    #[test]
    fn dirigo_gate_rejects_late_ratio_above_five_percent() {
        let e2e = vec![100.0; 20];
        let stats = stats_with(e2e, 2, 6.16, 6.16);
        let reasons = evaluate_safety(&stats, &gate(), 6.16);
        assert!(
            reasons.iter().any(|r| r.contains("late_frame_ratio")),
            "{reasons:?}"
        );
    }

    #[test]
    fn dirigo_gate_rejects_realtime_ratio_below_095() {
        let stats = stats_with(vec![100.0; 11], 0, 6.16, 6.16 / 0.94);
        let reasons = evaluate_safety(&stats, &gate(), 6.16);
        assert!(
            reasons.iter().any(|r| r.contains("realtime_ratio")),
            "{reasons:?}"
        );
    }

    #[test]
    fn dirigo_gate_passes_n6_like_keep_up_under_p99() {
        let stats = stats_with(vec![348.0; 198], 0, 6.16, 6.16 / 0.9578);
        let reasons = evaluate_safety(&stats, &gate(), 6.16);
        assert!(reasons.is_empty(), "{reasons:?}");
    }
}
