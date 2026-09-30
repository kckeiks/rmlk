//! HTTP routes served by the binary.
//!
//! [`AppState`] is cloned into each WebSocket upgrade task. Connection tasks
//! own the socket and apply [`crate::worker::ConnEvent`]s; they never touch
//! the engine. The engine worker owns inference and all stream state.

use std::future::Future;
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::{routing::get, Router};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

#[cfg(feature = "ort")]
use crate::engine::OrtParakeetEngine;
use crate::engine::{Engine, EngineEvent, MockEngine};
use crate::protocol::{error_code, error_message, ClientFrame, ServerFrame};
use crate::session::{SessionError, SessionRegistry};
use crate::worker::{ConnEvent, EngineWorker, SessionHandle};

/// Backend name for the deterministic mock engine.
pub const ENGINE_MOCK: &str = "mock";
/// Backend name for the Nemotron engine running on ONNX Runtime.
pub const ENGINE_ORT: &str = "ort";

/// Shared inference backend selected at process start.
#[derive(Clone)]
enum AppBackend {
    Mock(EngineWorker<MockEngine>),
    #[cfg(feature = "ort")]
    Ort(EngineWorker<OrtParakeetEngine>),
}

#[derive(Clone)]
pub struct AppState {
    /// Live session ids + allocator (shared across connection tasks).
    registry: Arc<Mutex<SessionRegistry>>,
    /// Engine worker handle (connection tasks send work items only).
    backend: AppBackend,
    /// When set, each opened stream increments this on drop (tests).
    drop_counter: Option<Arc<AtomicUsize>>,
}

impl AppState {
    /// Number of live sessions in the registry.
    pub async fn live_session_count(&self) -> usize {
        self.registry.lock().await.len()
    }

    /// Human-readable backend name ([`ENGINE_MOCK`] or [`ENGINE_ORT`]).
    pub fn engine_name(&self) -> &'static str {
        match &self.backend {
            AppBackend::Mock(_) => ENGINE_MOCK,
            #[cfg(feature = "ort")]
            AppBackend::Ort(_) => ENGINE_ORT,
        }
    }
}

/// Application router with shared registry + engine service.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ws", get(ws_upgrade))
        .with_state(state)
}

/// Default in-process app state (mock engine).
pub fn new_app_state() -> AppState {
    let (worker, _join) = EngineWorker::spawn(MockEngine::new());
    AppState {
        registry: Arc::new(Mutex::new(SessionRegistry::default())),
        backend: AppBackend::Mock(worker),
        drop_counter: None,
    }
}

/// App state that counts `StreamState` drops (integration tests).
pub fn new_app_state_with_drop_counter(drop_counter: Arc<AtomicUsize>) -> AppState {
    let (worker, _join) = EngineWorker::spawn(MockEngine::new());
    AppState {
        registry: Arc::new(Mutex::new(SessionRegistry::default())),
        backend: AppBackend::Mock(worker),
        drop_counter: Some(drop_counter),
    }
}

/// Build app state from a backend name and optional Nemotron model directory.
///
/// `engine` is [`ENGINE_MOCK`] (default path) or [`ENGINE_ORT`] (requires
/// `--features ort` and a model directory). Unknown names error.
pub fn app_state_from_config(engine: &str, model_dir: Option<&Path>) -> Result<AppState> {
    match engine {
        ENGINE_MOCK => Ok(new_app_state()),
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
                let (worker, _join) = EngineWorker::spawn(ort);
                Ok(AppState {
                    registry: Arc::new(Mutex::new(SessionRegistry::default())),
                    backend: AppBackend::Ort(worker),
                    drop_counter: None,
                })
            }
            #[cfg(not(feature = "ort"))]
            {
                let _ = model_dir;
                bail!("engine `{ENGINE_ORT}` requires building with `--features ort`");
            }
        }
        other => bail!("unknown engine `{other}` (expected `{ENGINE_MOCK}` or `{ENGINE_ORT}`)"),
    }
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
                handle_open(&mut socket, &state, worker).await;
            }
            #[cfg(feature = "ort")]
            AppBackend::Ort(worker) => {
                handle_open(&mut socket, &state, worker).await;
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

/// Handle an Open frame.
async fn handle_open<E: Engine + 'static>(
    socket: &mut WebSocket,
    state: &AppState,
    worker: &EngineWorker<E>,
) {
    let (id, mut stream) = state.registry.lock().await.open::<E::CallState>();
    if let Some(counter) = &state.drop_counter {
        stream.track_drops(Arc::clone(counter));
    }

    let mut session = match worker.open(id, stream).await {
        Ok(session) => session,
        Err(err) => {
            let _ = state.registry.lock().await.unregister(id);
            send_session_error_and_close(socket, err).await;
            return;
        }
    };
    if let Err(err) = wait_opened(&mut session).await {
        let _ = state.registry.lock().await.unregister(id);
        send_session_error_and_close(socket, err).await;
        return;
    }

    if send_frame(
        socket,
        &ServerFrame::OpenAck {
            session_id: id.as_u64(),
        },
    )
    .await
    .is_err()
    {
        cancel_and_wait(&mut session).await;
        let _ = state.registry.lock().await.unregister(id);
        return;
    }

    run_session(socket, &mut session).await;
    let _ = state.registry.lock().await.unregister(id);
}

/// Connection loop: send work items, apply worker events to the socket.
async fn run_session<C>(socket: &mut WebSocket, session: &mut SessionHandle<C>) {
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Close(_) => {
                cancel_and_wait(session).await;
                return;
            }
            Message::Binary(bytes) => match ClientFrame::decode(&bytes) {
                Ok(ClientFrame::Audio { pcm16 }) => match session.push_audio(pcm16).await {
                    Ok(()) => match wait_partial(session).await {
                        Ok(Some(event)) => {
                            if send_frame(socket, &event.into_server_frame())
                                .await
                                .is_err()
                            {
                                cancel_and_wait(session).await;
                                return;
                            }
                        }
                        Ok(None) => {}
                        Err(err) => {
                            send_session_error_and_close(socket, err).await;
                            return;
                        }
                    },
                    Err(err) => {
                        cancel_and_wait(session).await;
                        send_session_error_and_close(socket, err).await;
                        return;
                    }
                },
                Ok(ClientFrame::Finalize) => {
                    if let Err(err) = session.finalize().await {
                        send_session_error_and_close(socket, err).await;
                        return;
                    }
                    match wait_finalized(session).await {
                        Ok(event) => {
                            let _ = send_frame(socket, &event.into_server_frame()).await;
                            let _ = socket.send(Message::Close(None)).await;
                        }
                        Err(err) => {
                            send_session_error_and_close(socket, err).await;
                        }
                    }
                    return;
                }
                Ok(ClientFrame::Cancel) => {
                    cancel_and_wait(session).await;
                    let _ = socket.send(Message::Close(None)).await;
                    return;
                }
                Ok(ClientFrame::Open) => {
                    cancel_and_wait(session).await;
                    send_error_and_close(
                        socket,
                        error_code::UNEXPECTED_FRAME,
                        error_message::EXPECTED_OPEN,
                    )
                    .await;
                    return;
                }
                Err(_) => {
                    cancel_and_wait(session).await;
                    send_error_and_close(
                        socket,
                        error_code::MALFORMED_FRAME,
                        error_message::MALFORMED_FRAME,
                    )
                    .await;
                    return;
                }
            },
            _ => {}
        }
    }
    cancel_and_wait(session).await;
}

fn worker_gone(context: &str) -> SessionError {
    SessionError::Engine(crate::engine::EngineError::Failed(format!(
        "engine worker stopped while waiting for {context}"
    )))
}

fn unexpected_event(expected: &str, got: ConnEvent) -> SessionError {
    SessionError::Engine(crate::engine::EngineError::Failed(format!(
        "expected {expected}, got {got:?}"
    )))
}

async fn wait_opened<C>(session: &mut SessionHandle<C>) -> Result<(), SessionError> {
    match session.recv().await {
        Some(ConnEvent::Opened) => Ok(()),
        Some(ConnEvent::Failed(err)) => Err(err),
        Some(other) => Err(unexpected_event("Opened", other)),
        None => Err(worker_gone("Opened")),
    }
}

async fn wait_partial<C>(
    session: &mut SessionHandle<C>,
) -> Result<Option<EngineEvent>, SessionError> {
    match session.recv().await {
        Some(ConnEvent::Partial(event)) => Ok(event),
        Some(ConnEvent::Failed(err)) => Err(err),
        Some(other) => Err(unexpected_event("Partial", other)),
        None => Err(worker_gone("Partial")),
    }
}

async fn wait_finalized<C>(session: &mut SessionHandle<C>) -> Result<EngineEvent, SessionError> {
    loop {
        match session.recv().await {
            Some(ConnEvent::Partial(_)) => {}
            Some(ConnEvent::Finalized(event)) => return Ok(event),
            Some(ConnEvent::Failed(err)) => return Err(err),
            Some(other) => return Err(unexpected_event("Finalized", other)),
            None => return Err(worker_gone("Finalized")),
        }
    }
}

/// Send `CancelStream` and drain events until the worker confirms the session is gone.
async fn cancel_and_wait<C>(session: &mut SessionHandle<C>) {
    if session.cancel().await.is_err() {
        return;
    }
    loop {
        match session.recv().await {
            Some(ConnEvent::Cancelled) | Some(ConnEvent::Finalized(_)) => return,
            Some(ConnEvent::Failed(_)) => return,
            Some(ConnEvent::Partial(_)) | Some(ConnEvent::Opened) => {}
            None => return,
        }
    }
}

async fn send_session_error_and_close(socket: &mut WebSocket, err: SessionError) {
    match err {
        SessionError::Busy => {
            send_error_and_close(socket, error_code::BUSY, error_message::BUSY).await;
        }
        SessionError::Unknown(_) | SessionError::Engine(_) => {
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
pub async fn serve_with_state(
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
    state: AppState,
) -> Result<()> {
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

    use super::{
        app_state_from_config, new_app_state, router, serve_with_state, ENGINE_MOCK, ENGINE_ORT,
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
    async fn app_state_clones_share_registry() {
        let state = new_app_state();
        let clone = state.clone();
        {
            let mut reg = state.registry.lock().await;
            let _ = reg.open::<()>();
            assert_eq!(reg.len(), 1);
        }
        assert_eq!(clone.live_session_count().await, 1);
        assert_eq!(state.live_session_count().await, 1);
    }

    #[tokio::test]
    async fn app_state_from_config_defaults_to_mock() {
        let state = app_state_from_config(ENGINE_MOCK, None).unwrap();
        assert_eq!(state.engine_name(), ENGINE_MOCK);
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
