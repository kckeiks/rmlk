use super::{
    audio::{append_loop, Clip},
    metrics::{Level, Stream},
};
use crate::Args;
use anyhow::{bail, Context, Result};
use futures_util::{SinkExt, StreamExt};
use rmlk_server::protocol::{ClientFrame, ServerFrame};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::{
    sync::mpsc,
    task::{JoinHandle, JoinSet},
    time::{sleep, timeout},
};
use tokio_tungstenite::{tungstenite::Message, MaybeTlsStream, WebSocketStream};

type Ws = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;
struct Chunk {
    pcm: Vec<i16>,
    enqueued: Instant,
}

// Dropping a level on Ctrl-C must also drop sockets held by child RPC tasks.
struct AbortOnDrop<T>(JoinHandle<T>);
impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

async fn send(ws: &mut Ws, frame: ClientFrame) -> Result<()> {
    ws.send(Message::Binary(frame.encode()?.into())).await?;
    Ok(())
}
async fn receive(ws: &mut Ws) -> Result<ServerFrame> {
    loop {
        match ws
            .next()
            .await
            .context("WebSocket ended before acknowledgment")??
        {
            Message::Binary(bytes) => {
                return ServerFrame::decode(&bytes).context("decode server frame")
            }
            Message::Ping(bytes) => ws.send(Message::Pong(bytes)).await?,
            Message::Pong(_) => {}
            other => bail!("unexpected WebSocket message: {other:?}"),
        }
    }
}

async fn open(args: &Args) -> Result<Ws> {
    let (mut ws, _) = tokio_tungstenite::connect_async_with_config(&args.url, None, true).await?;
    send(&mut ws, ClientFrame::Open).await?;
    match receive(&mut ws).await? {
        ServerFrame::OpenAck { .. } => Ok(ws),
        other => bail!("expected OpenAck, got {other:?}"),
    }
}

async fn rpc_worker(
    mut ws: Ws,
    mut rx: mpsc::Receiver<Chunk>,
    stats: Arc<Mutex<Stream>>,
) -> Result<()> {
    // One request in flight per stream, as in Dirigo's Triton client. The producer
    // keeps its own frame clock and queues subsequent chunks independently.
    while let Some(chunk) = rx.recv().await {
        send(&mut ws, ClientFrame::Audio { pcm16: chunk.pcm }).await?;
        loop {
            match receive(&mut ws).await? {
                ServerFrame::AudioProcessed => {
                    let elapsed = chunk.enqueued.elapsed().as_secs_f64() * 1000.0;
                    let mut stats = stats.lock().unwrap();
                    stats.latencies.push(elapsed);
                    stats.chunks_acked += 1;
                    break;
                }
                ServerFrame::Partial { .. } => stats.lock().unwrap().partials += 1,
                other => bail!("expected AudioProcessed, got {other:?}"),
            }
        }
    }
    // Dirigo ends a phase without flushing an utterance. Cancel releases the
    // session; no padding, Finalize, or synthetic EOS work enters these metrics.
    send(&mut ws, ClientFrame::Cancel).await?;
    loop {
        match ws.next().await {
            Some(Ok(Message::Close(_))) | None => break,
            Some(Ok(Message::Ping(bytes))) => ws.send(Message::Pong(bytes)).await?,
            Some(Ok(Message::Pong(_))) => {}
            Some(Err(err)) => return Err(err.into()),
            other => bail!("unexpected message after Cancel: {other:?}"),
        }
    }
    Ok(())
}

async fn run_stream(args: Args, clip: Arc<Vec<i16>>, name: String, seconds: f64) -> Stream {
    // Like Dirigo, the clock starts before open and the hold duration after open.
    let origin = Instant::now();
    let stats = Arc::new(Mutex::new(Stream {
        clip: name,
        ..Stream::default()
    }));
    let ws = match timeout(
        Duration::from_secs_f64(args.open_timeout_seconds),
        open(&args),
    )
    .await
    {
        Ok(Ok(ws)) => ws,
        other => {
            stats.lock().unwrap().errors.push(match other {
                Ok(Err(err)) => format!("open: {err:#}"),
                _ => "open timeout".into(),
            });
            return stats.lock().unwrap().clone();
        }
    };
    let (tx, rx) = mpsc::channel(args.max_pending_chunks);
    let mut worker = AbortOnDrop(tokio::spawn(rpc_worker(ws, rx, stats.clone())));
    let end = Instant::now() + Duration::from_secs_f64(seconds);
    let frame = Duration::from_millis(args.frame_ms);
    let frame_samples = args.frame_ms as usize * 16;
    let chunk_samples = args.server_chunk_ms * 16;
    let mut frames = 0;
    let mut cursor = 0;
    let mut buffered = Vec::with_capacity(chunk_samples + frame_samples);
    while Instant::now() < end {
        let target = origin + frame.mul_f64((frames + 1) as f64);
        let now = Instant::now();
        if now < target {
            sleep(target - now).await;
        } else {
            let mut s = stats.lock().unwrap();
            s.late_frames += 1;
            s.max_lateness_ms = s.max_lateness_ms.max((now - target).as_secs_f64() * 1000.0);
        }
        frames += 1;
        stats.lock().unwrap().frames_sent = frames;
        append_loop(&clip, &mut cursor, frame_samples, &mut buffered);
        let mut failed = false;
        while buffered.len() >= chunk_samples {
            let pcm = buffered.drain(..chunk_samples).collect();
            // Timestamp at queue admission, not at eventual network send: all
            // client backlog is included in E2E latency rather than hidden.
            let chunk = Chunk {
                pcm,
                enqueued: Instant::now(),
            };
            // Serialize counter publication with ACK accounting.
            let mut s = stats.lock().unwrap();
            match tx.try_send(chunk) {
                Ok(()) => {
                    s.chunks_queued += 1;
                    s.peak_pending_chunks =
                        s.peak_pending_chunks.max(s.chunks_queued - s.chunks_acked);
                }
                Err(err) => {
                    s.errors.push(format!("client queue: {err}"));
                    failed = true;
                    break;
                }
            }
        }
        if failed {
            break;
        }
    }
    {
        let mut s = stats.lock().unwrap();
        s.sender_realtime_ratio =
            frames as f64 * frame.as_secs_f64() / origin.elapsed().as_secs_f64();
        s.tail_samples = buffered.len();
    }
    drop(tx);
    let error = match timeout(
        Duration::from_secs_f64(args.drain_timeout_seconds),
        &mut worker.0,
    )
    .await
    {
        Ok(Ok(Ok(()))) => None,
        Ok(Ok(Err(err))) => Some(format!("RPC/close: {err:#}")),
        Ok(Err(err)) => Some(format!("RPC task: {err}")),
        Err(_) => {
            worker.0.abort();
            // Await drop of the socket before the next level starts.
            let _ = (&mut worker.0).await;
            Some("drain/close timeout; pending audio is not counted as successful".into())
        }
    };
    let mut s = stats.lock().unwrap();
    s.realtime_ratio = frames as f64 * frame.as_secs_f64() / origin.elapsed().as_secs_f64();
    if let Some(error) = error {
        s.errors.push(error);
    }
    s.clone()
}

pub async fn run_level(
    args: &Args,
    clips: Arc<Vec<Clip>>,
    phase: &str,
    n: usize,
    seconds: f64,
) -> Level {
    let start = Instant::now();
    let mut tasks = JoinSet::new();
    for i in 0..n {
        let clip = &clips[i % clips.len()];
        let args = args.clone();
        let pcm = clip.pcm.clone();
        let name = clip.info.name.clone();
        tasks.spawn(async move { (i, run_stream(args, pcm, name, seconds).await) });
    }
    let mut results = Vec::new();
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(result) => results.push(result),
            Err(err) => results.push((
                usize::MAX,
                Stream {
                    errors: vec![format!("stream task: {err}")],
                    ..Stream::default()
                },
            )),
        }
    }
    results.sort_by_key(|(i, _)| *i);
    Level::new(
        args,
        phase,
        n,
        seconds,
        start.elapsed().as_secs_f64(),
        results.into_iter().map(|(_, s)| s).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use rmlk_server::{
        engine::MockEngine,
        http::{new_mock_app_state, serve_with_state},
        worker::WorkerLimits,
    };

    #[tokio::test]
    async fn delayed_ack_does_not_throttle_clock_and_backlog_fails_gate() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let state = new_mock_app_state(
            MockEngine::with_step_delay(Duration::from_millis(100)),
            WorkerLimits::default(),
            None,
        );
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(serve_with_state(
            listener,
            async {
                let _ = shutdown_rx.await;
            },
            state.clone(),
        ));
        let args = Args::parse_from([
            "test",
            "--server-chunk-ms",
            "40",
            "--max-p99-client-e2e-ms",
            "150",
        ]);
        let args = Args {
            url: format!("ws://{addr}/ws"),
            ..args
        };
        let s = run_stream(args.clone(), Arc::new(vec![1; 320]), "test".into(), 0.6).await;
        assert!(s.errors.is_empty(), "{:?}", s.errors);
        assert!(
            s.frames_sent >= 28,
            "clock was throttled: {} frames",
            s.frames_sent
        );
        assert!(s.peak_pending_chunks > 1);
        assert_eq!(s.chunks_acked, s.chunks_queued);
        let level = Level::new(&args, "ramp", 1, 0.6, 2., vec![s]);
        assert!(level.client_e2e.p99_ms.unwrap() > 500.0);
        assert!(!level.safe);
        assert_eq!(state.live_session_count(), 0);
        let _ = shutdown_tx.send(());
        server.await.unwrap().unwrap();
    }
}
