//! Engine worker: one blocking task owns the [`Engine`] and every live
//! [`StreamState`]. Connection tasks talk to it only through channels.
//!
//! Work flows through a single bounded FIFO channel of [`WorkItem`]s. Because
//! one channel carries both control and audio for every session, per-session
//! ordering is preserved and control items stay ordered relative to audio.
//! Replies go back on one unbounded channel per session as [`ConnEvent`]s.
//!
//! The worker runs exactly one engine call per work item, so the channel
//! order is the scheduling policy. No mutex sits on the engine or the queue.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::engine::{Engine, EngineError, EngineEvent};
use crate::session::{SessionError, SessionId, StreamState};

/// Work channel depth until Phase 8 provides `max_sessions`.
///
/// Sized as a generous `sessions × chunks_in_flight` product so that, with
/// per-session backpressure in place, the channel itself never fills.
pub const DEFAULT_WORK_CHANNEL_CAPACITY: usize = 64 * 16;

/// One unit of work for the engine worker.
pub enum WorkItem<C> {
    /// Register `stream` under `id` and run [`Engine::open_stream`].
    OpenStream {
        id: SessionId,
        stream: StreamState<C>,
        reply_tx: mpsc::UnboundedSender<ConnEvent>,
        cancelled: Arc<AtomicBool>,
    },
    /// Run one [`Engine::push_audio`] on `id`.
    PushAudio { id: SessionId, pcm16: Vec<i16> },
    /// Run [`Engine::finalize`] on `id` and drop its state.
    FinalizeStream { id: SessionId },
    /// Run [`Engine::cancel`] on `id` and drop its state.
    CancelStream { id: SessionId },
}

/// Events delivered from the worker to one connection task.
///
/// `Finalized`, `Cancelled`, and `Failed` are terminal: the worker has already
/// removed the session when it sends them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnEvent {
    /// [`Engine::open_stream`] succeeded; the session is live.
    Opened,
    /// One audio step produced these transcript events (may be empty).
    Partials(Vec<EngineEvent>),
    /// Stream finalized; includes the final transcript event(s).
    Finalized(Vec<EngineEvent>),
    /// Stream cancelled without a final transcript.
    Cancelled,
    /// Engine or session failure; the session has been dropped.
    Failed(SessionError),
}

struct Slot<C> {
    stream: StreamState<C>,
    reply_tx: mpsc::UnboundedSender<ConnEvent>,
    cancelled: Arc<AtomicBool>,
}

/// Handle to the engine worker. Cheap to clone; dropping the last clone lets
/// the worker drain and exit.
pub struct EngineWorker<E: Engine> {
    work_tx: mpsc::Sender<WorkItem<E::CallState>>,
    live: Arc<AtomicUsize>,
    next_id: Arc<AtomicU64>,
}

impl<E: Engine> Clone for EngineWorker<E> {
    fn clone(&self) -> Self {
        Self {
            work_tx: self.work_tx.clone(),
            live: Arc::clone(&self.live),
            next_id: Arc::clone(&self.next_id),
        }
    }
}

impl<E: Engine + 'static> EngineWorker<E> {
    /// Spawn the worker on Tokio's blocking pool with the default channel depth.
    pub fn spawn(engine: E) -> (Self, JoinHandle<()>) {
        Self::spawn_with_capacity(engine, DEFAULT_WORK_CHANNEL_CAPACITY)
    }

    /// Spawn the worker with an explicit work channel depth.
    pub fn spawn_with_capacity(engine: E, capacity: usize) -> (Self, JoinHandle<()>) {
        let (work_tx, work_rx) = mpsc::channel(capacity);
        let live = Arc::new(AtomicUsize::new(0));
        let live_worker = Arc::clone(&live);
        let join = tokio::task::spawn_blocking(move || worker_main(engine, work_rx, live_worker));
        (
            Self {
                work_tx,
                live,
                next_id: Arc::new(AtomicU64::new(0)),
            },
            join,
        )
    }

    /// Number of sessions the worker currently holds state for.
    pub fn live_sessions(&self) -> usize {
        self.live.load(Ordering::Acquire)
    }

    /// Allocate a fresh session id.
    pub fn next_session_id(&self) -> SessionId {
        SessionId::from_raw(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Send `stream` to the worker under `id` and return the session handle.
    ///
    /// The caller must wait for [`ConnEvent::Opened`] before treating the
    /// session as live.
    pub async fn open(
        &self,
        id: SessionId,
        stream: StreamState<E::CallState>,
    ) -> Result<SessionHandle<E::CallState>, SessionError> {
        let (reply_tx, reply_rx) = mpsc::unbounded_channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        self.send(WorkItem::OpenStream {
            id,
            stream,
            reply_tx,
            cancelled: Arc::clone(&cancelled),
        })
        .await?;
        Ok(SessionHandle {
            id,
            work_tx: self.work_tx.clone(),
            reply_rx,
            cancelled,
        })
    }

    async fn send(&self, item: WorkItem<E::CallState>) -> Result<(), SessionError> {
        send_item(&self.work_tx, item).await
    }
}

/// Connection-side handle for one session: sends work, receives events.
pub struct SessionHandle<C> {
    id: SessionId,
    work_tx: mpsc::Sender<WorkItem<C>>,
    reply_rx: mpsc::UnboundedReceiver<ConnEvent>,
    cancelled: Arc<AtomicBool>,
}

impl<C> SessionHandle<C> {
    /// Session id this handle controls.
    pub fn id(&self) -> SessionId {
        self.id
    }

    /// Wait for the next worker event. `None` means the worker is gone.
    pub async fn recv(&mut self) -> Option<ConnEvent> {
        self.reply_rx.recv().await
    }

    /// Queue one chunk of PCM16 for the engine.
    pub async fn push_audio(&self, pcm16: Vec<i16>) -> Result<(), SessionError> {
        send_item(&self.work_tx, WorkItem::PushAudio { id: self.id, pcm16 }).await
    }

    /// Ask the worker to finalize after any queued audio.
    pub async fn finalize(&self) -> Result<(), SessionError> {
        send_item(&self.work_tx, WorkItem::FinalizeStream { id: self.id }).await
    }

    /// Mark the session cancelled (queued audio is skipped) and ask the worker
    /// to drop it. Waits for channel space; never times out.
    pub async fn cancel(&self) -> Result<(), SessionError> {
        self.cancelled.store(true, Ordering::Release);
        send_item(&self.work_tx, WorkItem::CancelStream { id: self.id }).await
    }
}

async fn send_item<C>(tx: &mpsc::Sender<WorkItem<C>>, item: WorkItem<C>) -> Result<(), SessionError> {
    tx.send(item)
        .await
        .map_err(|_| SessionError::Engine(EngineError::Failed("engine worker stopped".into())))
}

fn worker_main<E: Engine>(
    mut engine: E,
    mut work_rx: mpsc::Receiver<WorkItem<E::CallState>>,
    live: Arc<AtomicUsize>,
) {
    let mut slots: HashMap<SessionId, Slot<E::CallState>> = HashMap::new();

    while let Some(item) = work_rx.blocking_recv() {
        match item {
            WorkItem::OpenStream {
                id,
                mut stream,
                reply_tx,
                cancelled,
            } => match engine.open_stream(&mut stream) {
                Ok(()) => {
                    slots.insert(
                        id,
                        Slot {
                            stream,
                            reply_tx,
                            cancelled,
                        },
                    );
                    live.fetch_add(1, Ordering::AcqRel);
                    deliver(&mut engine, &mut slots, &live, id, ConnEvent::Opened);
                }
                Err(err) => {
                    let _ = reply_tx.send(ConnEvent::Failed(SessionError::Engine(err)));
                }
            },
            WorkItem::PushAudio { id, pcm16 } => {
                let Some(slot) = slots.get_mut(&id) else { continue };
                if slot.cancelled.load(Ordering::Acquire) {
                    continue;
                }
                match engine.push_audio(&mut slot.stream, &pcm16) {
                    Ok(events) => {
                        deliver(&mut engine, &mut slots, &live, id, ConnEvent::Partials(events));
                    }
                    Err(err) => {
                        fail(&mut engine, &mut slots, &live, id, SessionError::Engine(err));
                    }
                }
            }
            WorkItem::FinalizeStream { id } => {
                let Some(slot) = remove(&mut slots, &live, id) else { continue };
                let event = match slot.stream.finalize(&mut engine) {
                    Ok(events) => ConnEvent::Finalized(events),
                    Err(err) => ConnEvent::Failed(err),
                };
                let _ = slot.reply_tx.send(event);
            }
            WorkItem::CancelStream { id } => {
                let Some(slot) = remove(&mut slots, &live, id) else { continue };
                let _ = slot.stream.cancel(&mut engine);
                let _ = slot.reply_tx.send(ConnEvent::Cancelled);
            }
        }
    }

    // Last sender dropped: release every remaining stream through the engine.
    for (_, slot) in slots.drain() {
        live.fetch_sub(1, Ordering::AcqRel);
        let _ = slot.stream.cancel(&mut engine);
    }
}

fn remove<C>(
    slots: &mut HashMap<SessionId, Slot<C>>,
    live: &AtomicUsize,
    id: SessionId,
) -> Option<Slot<C>> {
    let slot = slots.remove(&id)?;
    live.fetch_sub(1, Ordering::AcqRel);
    Some(slot)
}

/// Send `event` to the session's connection. If the connection is gone, the
/// session is cancelled and dropped as a backstop.
fn deliver<E: Engine>(
    engine: &mut E,
    slots: &mut HashMap<SessionId, Slot<E::CallState>>,
    live: &AtomicUsize,
    id: SessionId,
    event: ConnEvent,
) {
    let Some(slot) = slots.get(&id) else { return };
    if slot.reply_tx.send(event).is_err() {
        if let Some(slot) = remove(slots, live, id) {
            let _ = slot.stream.cancel(engine);
        }
    }
}

/// Terminal failure: cancel through the engine, drop the session, then report.
fn fail<E: Engine>(
    engine: &mut E,
    slots: &mut HashMap<SessionId, Slot<E::CallState>>,
    live: &AtomicUsize,
    id: SessionId,
    err: SessionError,
) {
    let Some(slot) = remove(slots, live, id) else { return };
    let reply_tx = slot.reply_tx.clone();
    let _ = slot.stream.cancel(engine);
    let _ = reply_tx.send(ConnEvent::Failed(err));
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use super::{ConnEvent, EngineWorker};
    use crate::engine::{Engine, EngineError, EngineEvent, MockEngine};
    use crate::session::{SessionError, SessionId, StreamState};

    /// Mock that counts engine calls so tests can observe the worker.
    #[derive(Clone, Default)]
    struct CountingEngine {
        pushes: Arc<AtomicUsize>,
        cancels: Arc<AtomicUsize>,
        fail_open: bool,
    }

    impl Engine for CountingEngine {
        type CallState = ();

        fn open_stream(&mut self, _s: &mut StreamState<()>) -> Result<(), EngineError> {
            if self.fail_open {
                return Err(EngineError::Failed("open refused".into()));
            }
            Ok(())
        }

        fn push_audio(
            &mut self,
            state: &mut StreamState<()>,
            pcm16: &[i16],
        ) -> Result<Vec<EngineEvent>, EngineError> {
            self.pushes.fetch_add(1, Ordering::SeqCst);
            MockEngine.push_audio(state, pcm16)
        }

        fn finalize(&mut self, state: &mut StreamState<()>) -> Result<Vec<EngineEvent>, EngineError> {
            MockEngine.finalize(state)
        }

        fn cancel(&mut self, _s: &mut StreamState<()>) -> Result<(), EngineError> {
            self.cancels.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    async fn wait_until(mut cond: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while !cond() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("condition should hold within timeout");
    }

    #[tokio::test]
    async fn open_push_finalize_round_trip() {
        let (worker, _join) = EngineWorker::spawn(MockEngine::new());
        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id, 16)).await.unwrap();

        assert_eq!(session.recv().await, Some(ConnEvent::Opened));
        assert_eq!(worker.live_sessions(), 1);

        session.push_audio(vec![0]).await.unwrap();
        assert_eq!(
            session.recv().await,
            Some(ConnEvent::Partials(vec![EngineEvent::Partial {
                text: "partial-1".into()
            }]))
        );

        session.finalize().await.unwrap();
        assert_eq!(
            session.recv().await,
            Some(ConnEvent::Finalized(vec![EngineEvent::Final {
                text: "final-1".into()
            }]))
        );
        assert_eq!(worker.live_sessions(), 0);
    }

    #[tokio::test]
    async fn cancelled_session_skips_queued_audio() {
        let engine = CountingEngine::default();
        let pushes = Arc::clone(&engine.pushes);
        let (worker, _join) = EngineWorker::spawn(engine);
        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id, 16)).await.unwrap();
        assert_eq!(session.recv().await, Some(ConnEvent::Opened));

        // Set the flag before any audio is queued so every chunk must be skipped.
        session.cancelled.store(true, Ordering::Release);
        session.push_audio(vec![0]).await.unwrap();
        session.push_audio(vec![1]).await.unwrap();
        session.cancel().await.unwrap();

        assert_eq!(session.recv().await, Some(ConnEvent::Cancelled));
        assert_eq!(pushes.load(Ordering::SeqCst), 0);
        assert_eq!(worker.live_sessions(), 0);
    }

    #[tokio::test]
    async fn unknown_id_is_ignored_and_worker_keeps_serving() {
        let (worker, _join) = EngineWorker::spawn(MockEngine::new());
        let ghost = SessionId::from_raw(999);
        let mut ghost_session = worker.open(ghost, StreamState::new(ghost, 16)).await.unwrap();
        assert_eq!(ghost_session.recv().await, Some(ConnEvent::Opened));
        ghost_session.cancel().await.unwrap();
        assert_eq!(ghost_session.recv().await, Some(ConnEvent::Cancelled));

        // Second cancel and a push for the same id must be silently dropped.
        ghost_session.cancel().await.unwrap();
        ghost_session.push_audio(vec![0]).await.unwrap();

        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id, 16)).await.unwrap();
        assert_eq!(session.recv().await, Some(ConnEvent::Opened));
        assert_eq!(worker.live_sessions(), 1);
    }

    #[tokio::test]
    async fn open_failure_reports_failed_and_holds_no_state() {
        let engine = CountingEngine {
            fail_open: true,
            ..CountingEngine::default()
        };
        let (worker, _join) = EngineWorker::spawn(engine);
        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id, 16)).await.unwrap();

        match session.recv().await {
            Some(ConnEvent::Failed(SessionError::Engine(_))) => {}
            other => panic!("expected Failed, got {other:?}"),
        }
        assert_eq!(worker.live_sessions(), 0);
    }

    #[tokio::test]
    async fn worker_exits_and_cancels_remaining_when_last_sender_drops() {
        let engine = CountingEngine::default();
        let cancels = Arc::clone(&engine.cancels);
        let (worker, join) = EngineWorker::spawn(engine);
        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id, 16)).await.unwrap();
        assert_eq!(session.recv().await, Some(ConnEvent::Opened));

        drop(session);
        drop(worker);

        tokio::time::timeout(Duration::from_secs(2), join)
            .await
            .expect("worker should exit once every sender is gone")
            .unwrap();
        assert_eq!(cancels.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dropped_reply_receiver_drops_session_on_next_event() {
        let engine = CountingEngine::default();
        let cancels = Arc::clone(&engine.cancels);
        let (worker, _join) = EngineWorker::spawn(engine);
        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id, 16)).await.unwrap();
        assert_eq!(session.recv().await, Some(ConnEvent::Opened));

        // Simulate a connection that vanished without sending CancelStream.
        let work_tx = session.work_tx.clone();
        drop(session);
        work_tx
            .send(super::WorkItem::PushAudio { id, pcm16: vec![0] })
            .await
            .unwrap();

        wait_until(|| worker.live_sessions() == 0).await;
        assert_eq!(cancels.load(Ordering::SeqCst), 1);
    }
}
