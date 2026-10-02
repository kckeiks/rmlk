//! HTTP routes served by the binary.
//!
//! [`AppState`] is cloned into each WebSocket upgrade task. A connection task
//! owns its socket and a [`SessionHandle`]; it never touches the engine. The
//! engine worker owns inference and the state of every live stream.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::serve::ListenerExt;
use axum::{routing::get, Router};
use futures_util::FutureExt;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

#[cfg(feature = "ort")]
use crate::engine::OrtParakeetEngine;
use crate::engine::{Engine, MockEngine};
use crate::protocol::{error_code, error_message, ClientFrame, ServerFrame};
use crate::session::{SessionError, SessionId, StreamState};
use crate::worker::{ConnEvent, EngineWorker, SessionHandle, WorkerLimits};

/// Backend name for the deterministic mock engine.
pub const ENGINE_MOCK: &str = "mock";
/// Backend name for the Nemotron engine running on ONNX Runtime.
pub const ENGINE_ORT: &str = "ort";
/// Backend name for the in-process native NeMo SDK.
pub const ENGINE_NEMO: &str = "nemo";

/// Shared inference backend selected at process start.
#[derive(Clone)]
enum AppBackend {
    Mock(EngineWorker<MockEngine>),
    #[cfg(feature = "nemo")]
    Nemo(EngineWorker<crate::engine::NemoEngine>),
    #[cfg(feature = "ort")]
    Ort(EngineWorker<OrtParakeetEngine>),
}

#[derive(Clone)]
pub struct AppState {
    /// Engine worker handle (connection tasks send work items only).
    backend: AppBackend,
    /// When set, each opened stream increments this on drop (tests).
    drop_counter: Option<Arc<AtomicUsize>>,
}

impl AppState {
    /// Number of sessions the engine worker currently holds.
    pub fn live_session_count(&self) -> usize {
        match &self.backend {
            AppBackend::Mock(worker) => worker.live_sessions(),
            #[cfg(feature = "nemo")]
            AppBackend::Nemo(worker) => worker.live_sessions(),
            #[cfg(feature = "ort")]
            AppBackend::Ort(worker) => worker.live_sessions(),
        }
    }

    /// Human-readable backend name (`mock`, `ort`, or `nemo`).
    pub fn engine_name(&self) -> &'static str {
        match &self.backend {
            AppBackend::Mock(_) => ENGINE_MOCK,
            #[cfg(feature = "nemo")]
            AppBackend::Nemo(_) => ENGINE_NEMO,
            #[cfg(feature = "ort")]
            AppBackend::Ort(_) => ENGINE_ORT,
        }
    }
}

/// Application router backed by the shared engine worker.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ws", get(ws_upgrade))
        .with_state(state)
}

/// Default in-process app state (mock engine, default limits).
pub fn new_app_state() -> AppState {
    new_mock_app_state(MockEngine::new(), WorkerLimits::default(), None)
}

/// App state that counts `StreamState` drops (integration tests).
pub fn new_app_state_with_drop_counter(drop_counter: Arc<AtomicUsize>) -> AppState {
    new_mock_app_state(
        MockEngine::new(),
        WorkerLimits::default(),
        Some(drop_counter),
    )
}

/// App state over a specific mock engine and worker limits (tests).
pub fn new_mock_app_state(
    engine: MockEngine,
    limits: WorkerLimits,
    drop_counter: Option<Arc<AtomicUsize>>,
) -> AppState {
    mock_app_state_with_worker(engine, limits, drop_counter).0
}

fn mock_app_state_with_worker(
    engine: MockEngine,
    limits: WorkerLimits,
    drop_counter: Option<Arc<AtomicUsize>>,
) -> (AppState, JoinHandle<()>) {
    let (worker, join) = EngineWorker::spawn_with_limits(engine, limits);
    (
        AppState {
            backend: AppBackend::Mock(worker),
            drop_counter,
        },
        join,
    )
}

/// Build app state from a backend name and optional Nemotron model directory.
///
/// `engine` is [`ENGINE_MOCK`] (default path) or [`ENGINE_ORT`] (requires
/// `--features ort` and a model directory). Unknown names error.
///
/// Also returns the engine worker's join handle. It resolves only when the
/// worker stops, so a process that is still serving should treat that as
/// fatal.
pub fn app_state_from_config(
    engine: &str,
    model_dir: Option<&Path>,
) -> Result<(AppState, JoinHandle<()>)> {
    match engine {
        ENGINE_MOCK => Ok(mock_app_state_with_worker(
            MockEngine::new(),
            WorkerLimits::default(),
            None,
        )),
        ENGINE_ORT => {
            #[cfg(feature = "ort")]
            {
                let dir = model_dir.ok_or_else(|| {
                    anyhow::anyhow!(
                        "engine `{ENGINE_ORT}` requires --model-dir / RMLK_NEMOTRON_MODEL_DIR"
                    )
                })?;
                let ort = OrtParakeetEngine::load(dir)
                    .map_err(|err| anyhow::anyhow!("failed to load OrtParakeetEngine: {err}"))?;
                println!(
                    "nemotron loaded from {} (onnx runtime execution provider: {})",
                    dir.display(),
                    ort.execution_provider()
                );
                let (worker, join) = EngineWorker::spawn(ort);
                Ok((
                    AppState {
                        backend: AppBackend::Ort(worker),
                        drop_counter: None,
                    },
                    join,
                ))
            }
            #[cfg(not(feature = "ort"))]
            {
                let _ = model_dir;
                bail!("engine `{ENGINE_ORT}` requires building with `--features ort`");
            }
        }
        ENGINE_NEMO => {
            #[cfg(feature = "nemo")]
            bail!(
                "engine `nemo` requires app_state_from_nemo_config with a library and GGUF model"
            );
            #[cfg(not(feature = "nemo"))]
            bail!("engine `nemo` requires building with `--features nemo`");
        }
        other => bail!("unknown engine `{other}` (expected `{ENGINE_MOCK}` or `{ENGINE_ORT}`)"),
    }
}

/// Build a native NeMo backend with one shared model and bounded worker slots.
///
/// # Safety
/// The selected SDK must satisfy [`crate::engine::NemoEngine::load`]'s contract.
#[cfg(feature = "nemo")]
pub unsafe fn app_state_from_nemo_config(
    config: &crate::engine::NemoConfig,
) -> Result<(AppState, JoinHandle<()>)> {
    // SAFETY: caller guarantees the selected native SDK's contract.
    let engine = unsafe { crate::engine::NemoEngine::load(config) }?;
    let engines = (0..config.max_sessions).map(|_| engine.clone()).collect();
    let limits = WorkerLimits {
        channel_capacity: WorkerLimits::default().in_flight_per_session + 2,
        ..WorkerLimits::default()
    };
    let (worker, join) = EngineWorker::spawn_pool(engines, limits);
    Ok((
        AppState {
            backend: AppBackend::Nemo(worker),
            drop_counter: None,
        },
        join,
    ))
}

async fn health() -> &'static str {
    "ok"
}

async fn ws_upgrade(State(state): State<AppState>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    let Some(first) = recv_binary_frame(&mut socket).await else {
        return;
    };

    match ClientFrame::decode(&first) {
        Ok(ClientFrame::Open) => match &state.backend {
            AppBackend::Mock(worker) => {
                run_connection(socket, &state, worker).await;
            }
            #[cfg(feature = "nemo")]
            AppBackend::Nemo(worker) => {
                run_connection(socket, &state, worker).await;
            }
            #[cfg(feature = "ort")]
            AppBackend::Ort(worker) => {
                run_connection(socket, &state, worker).await;
            }
        },
        Ok(_) => {
            send_error_and_close(
                &mut socket,
                error_code::UNEXPECTED_FRAME,
                error_message::EXPECTED_OPEN,
            )
            .await;
        }
        Err(_) => {
            send_error_and_close(
                &mut socket,
                error_code::MALFORMED_FRAME,
                error_message::MALFORMED_FRAME,
            )
            .await;
        }
    }
}

/// How the session ended, as seen from the worker's side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Exit {
    /// The worker has already dropped the session, or has stopped entirely.
    Settled,
    /// The worker may still hold the session and must be told to drop it.
    NeedsCancel,
}

/// Drive one connection that has sent `Open`.
///
/// This frame owns the session id and the [`SessionHandle`]. The session
/// loop runs under `catch_unwind` so that even a panic inside it ends here,
/// where the single cleanup point can still tell the worker to drop the
/// stream.
async fn run_connection<E: Engine + 'static>(
    mut socket: WebSocket,
    state: &AppState,
    worker: &EngineWorker<E>,
) {
    let id = worker.next_session_id();
    let mut stream = StreamState::new(id);
    if let Some(counter) = &state.drop_counter {
        stream.track_drops(Arc::clone(counter));
    }

    let mut session = match worker.open(id, stream).await {
        Ok(session) => session,
        Err(err) => {
            send_session_error_and_close(&mut socket, err).await;
            return;
        }
    };

    let exit = match catch_session_panic(id, drive_session(&mut socket, &mut session)).await {
        Some(exit) => exit,
        None => {
            send_error_and_close(&mut socket, error_code::INTERNAL, error_message::INTERNAL).await;
            Exit::NeedsCancel
        }
    };

    if let Exit::NeedsCancel = exit {
        // Waits for channel space with no timeout. If this never completes the
        // worker has stopped draining work, which is a process-level failure.
        let _ = session.cancel().await;
    }
}

/// Run the session future, isolating a panic to this connection.
///
/// Returns `None` when the future panicked; the panic has already been
/// logged at error level and the caller must treat the session as needing a
/// cancel.
async fn catch_session_panic(id: SessionId, session: impl Future<Output = Exit>) -> Option<Exit> {
    match AssertUnwindSafe(session).catch_unwind().await {
        Ok(exit) => Some(exit),
        Err(payload) => {
            log::error!(
                "connection task for session {} panicked: {}",
                id.as_u64(),
                crate::panic_message(payload.as_ref())
            );
            None
        }
    }
}

/// Wait for the worker to confirm the stream, acknowledge to the client, then
/// run the session loop.
async fn drive_session(socket: &mut WebSocket, session: &mut SessionHandle) -> Exit {
    match session.recv().await {
        Some(ConnEvent::Opened) => {}
        Some(ConnEvent::Failed(err)) => {
            send_session_error_and_close(socket, err).await;
            return Exit::Settled;
        }
        Some(_) => {
            send_error_and_close(socket, error_code::INTERNAL, error_message::INTERNAL).await;
            return Exit::NeedsCancel;
        }
        None => {
            send_error_and_close(socket, error_code::INTERNAL, error_message::INTERNAL).await;
            return Exit::Settled;
        }
    }

    let ack = ServerFrame::OpenAck {
        session_id: session.id().as_u64(),
    };
    if send_frame(socket, &ack).await.is_err() {
        return Exit::NeedsCancel;
    }

    session_loop(socket, session).await
}

/// Pipelined session loop.
///
/// Client frames become work items as they arrive, and worker events become
/// server frames as they arrive; neither side waits for the other. Once
/// `Finalize` or `Cancel` has been forwarded to the worker the socket is no
/// longer read, and the loop only waits for the worker's terminal event.
async fn session_loop(socket: &mut WebSocket, session: &mut SessionHandle) -> Exit {
    let mut closing = false;
    loop {
        tokio::select! {
            event = session.recv() => match event {
                Some(ConnEvent::Partial(partial)) => {
                    if let Some(event) = partial {
                        if send_frame(socket, &event.into_server_frame()).await.is_err() {
                            return Exit::NeedsCancel;
                        }
                    }
                    if send_frame(socket, &ServerFrame::AudioProcessed).await.is_err() {
                        return Exit::NeedsCancel;
                    }
                }
                Some(ConnEvent::Opened) => {}
                Some(ConnEvent::Finalized(event)) => {
                    let _ = send_frame(socket, &event.into_server_frame()).await;
                    let _ = socket.send(Message::Close(None)).await;
                    return Exit::Settled;
                }
                Some(ConnEvent::Cancelled) => {
                    let _ = socket.send(Message::Close(None)).await;
                    return Exit::Settled;
                }
                Some(ConnEvent::Failed(err)) => {
                    send_session_error_and_close(socket, err).await;
                    return Exit::Settled;
                }
                None => {
                    send_error_and_close(socket, error_code::INTERNAL, error_message::INTERNAL).await;
                    return Exit::Settled;
                }
            },
            msg = socket.recv(), if !closing => match msg {
                Some(Ok(Message::Binary(bytes))) => match ClientFrame::decode(&bytes) {
                    Ok(ClientFrame::Audio { pcm16 }) => match session.push_audio(pcm16).await {
                        Ok(()) => {}
                        Err(SessionError::Busy) => {
                            send_error_and_close(socket, error_code::BUSY, error_message::BUSY).await;
                            return Exit::NeedsCancel;
                        }
                        Err(SessionError::Engine(_)) => {
                            send_error_and_close(socket, error_code::INTERNAL, error_message::INTERNAL).await;
                            return Exit::Settled;
                        }
                    },
                    Ok(ClientFrame::Finalize) => {
                        if session.finalize().await.is_err() {
                            send_error_and_close(socket, error_code::INTERNAL, error_message::INTERNAL).await;
                            return Exit::Settled;
                        }
                        closing = true;
                    }
                    Ok(ClientFrame::Cancel) => {
                        if session.cancel().await.is_err() {
                            send_error_and_close(socket, error_code::INTERNAL, error_message::INTERNAL).await;
                            return Exit::Settled;
                        }
                        closing = true;
                    }
                    Ok(ClientFrame::Open) => {
                        send_error_and_close(socket, error_code::UNEXPECTED_FRAME, error_message::EXPECTED_OPEN).await;
                        return Exit::NeedsCancel;
                    }
                    Err(_) => {
                        send_error_and_close(socket, error_code::MALFORMED_FRAME, error_message::MALFORMED_FRAME).await;
                        return Exit::NeedsCancel;
                    }
                },
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return Exit::NeedsCancel,
                Some(Ok(_)) => {}
            },
        }
    }
}

async fn send_session_error_and_close(socket: &mut WebSocket, err: SessionError) {
    match err {
        SessionError::Busy => {
            send_error_and_close(socket, error_code::BUSY, error_message::BUSY).await;
        }
        SessionError::Engine(_) => {
            send_error_and_close(socket, error_code::INTERNAL, error_message::INTERNAL).await;
        }
    }
}

async fn recv_binary_frame(socket: &mut WebSocket) -> Option<axum::body::Bytes> {
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Binary(bytes) => return Some(bytes),
            Message::Close(_) => return None,
            _ => {}
        }
    }
    None
}

async fn send_frame(socket: &mut WebSocket, frame: &ServerFrame) -> Result<(), ()> {
    let bytes = frame.encode().map_err(|_| ())?;
    socket
        .send(Message::Binary(bytes.into()))
        .await
        .map_err(|_| ())
}

async fn send_error_and_close(socket: &mut WebSocket, code: u16, message: &str) {
    let _ = send_frame(
        socket,
        &ServerFrame::Error {
            code,
            message: message.into(),
        },
    )
    .await;
    let _ = socket.send(Message::Close(None)).await;
}

/// Serve `router` until `shutdown` completes (mock engine).
pub async fn serve(
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    serve_with_state(listener, shutdown, new_app_state()).await
}

/// Serve with an explicit shared [`AppState`] (tests / custom wiring).
///
/// Accepted sockets run with `TCP_NODELAY`. The server writes small frames
/// back to back (a `Partial` followed by `AudioProcessed`), and with Nagle on
/// the second one waits for the client's delayed ACK, which adds around 40 ms
/// to every step.
pub async fn serve_with_state(
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
    state: AppState,
) -> Result<()> {
    let listener = listener.tap_io(|tcp| {
        if let Err(err) = tcp.set_nodelay(true) {
            log::warn!("failed to set TCP_NODELAY on accepted connection: {err}");
        }
    });
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
        .context("http serve")?;
    Ok(())
}

/// Completes when the process receives Ctrl-C.
pub async fn shutdown_on_ctrl_c() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl-C handler");
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tower::ServiceExt;

    use std::time::Duration;

    use super::{
        app_state_from_config, catch_session_panic, new_app_state, router, serve_with_state,
        AppBackend, ConnEvent, Exit, SessionId, StreamState, ENGINE_MOCK, ENGINE_ORT,
    };

    #[tokio::test]
    async fn health_returns_ok() {
        let response = router(new_app_state())
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn graceful_shutdown_stops_serving() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            let _ = serve_with_state(
                listener,
                async {
                    let _ = shutdown_rx.await;
                },
                new_app_state(),
            )
            .await;
        });

        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut buf = vec![0u8; 1024];
        let n = stream.read(&mut buf).await.unwrap();
        let response = std::str::from_utf8(&buf[..n]).unwrap();
        assert!(response.contains("200"), "{response}");
        assert!(response.contains("ok"), "{response}");

        shutdown_tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn app_state_clones_share_worker() {
        let state = new_app_state();
        let clone = state.clone();
        // Irrefutable without the `ort` feature, refutable with it.
        #[allow(irrefutable_let_patterns)]
        let AppBackend::Mock(worker) = &state.backend
        else {
            panic!("default app state uses the mock backend");
        };

        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id)).await.unwrap();
        assert_eq!(session.recv().await, Some(ConnEvent::Opened));

        assert_eq!(clone.live_session_count(), 1);
        assert_eq!(state.live_session_count(), 1);
    }

    #[tokio::test]
    async fn app_state_from_config_defaults_to_mock() {
        let (state, _worker) = app_state_from_config(ENGINE_MOCK, None).unwrap();
        assert_eq!(state.engine_name(), ENGINE_MOCK);
    }

    #[tokio::test]
    async fn worker_join_resolves_only_after_last_state_drops() {
        let (state, mut worker) = app_state_from_config(ENGINE_MOCK, None).unwrap();

        let still_running = tokio::time::timeout(Duration::from_millis(50), &mut worker).await;
        assert!(
            still_running.is_err(),
            "worker must keep running while app state is alive"
        );

        drop(state);
        tokio::time::timeout(Duration::from_secs(2), worker)
            .await
            .expect("worker exits once the last app state drops")
            .unwrap();
    }

    #[tokio::test]
    async fn session_panic_is_caught_and_reported_as_needing_cancel() {
        let id = SessionId::from_raw(7);
        assert_eq!(
            catch_session_panic(id, async { Exit::Settled }).await,
            Some(Exit::Settled)
        );
        assert_eq!(
            catch_session_panic(id, async { panic!("boom") }).await,
            None
        );
    }

    #[test]
    fn app_state_from_config_rejects_unknown_engine() {
        let Err(err) = app_state_from_config("triton", None) else {
            panic!("expected unknown engine error");
        };
        assert!(err.to_string().contains("unknown engine"), "{err}");
    }

    #[test]
    fn app_state_from_config_ort_without_feature_or_dir_fails_clearly() {
        #[cfg(not(feature = "ort"))]
        {
            let Err(err) = app_state_from_config(ENGINE_ORT, Some(std::path::Path::new("/tmp")))
            else {
                panic!("expected feature error");
            };
            assert!(err.to_string().contains("--features ort"), "{err}");
        }
        #[cfg(feature = "ort")]
        {
            let Err(err) = app_state_from_config(ENGINE_ORT, None) else {
                panic!("expected model-dir error");
            };
            assert!(
                err.to_string().contains("model-dir")
                    || err.to_string().contains("RMLK_NEMOTRON_MODEL_DIR"),
                "{err}"
            );
        }
    }
}
