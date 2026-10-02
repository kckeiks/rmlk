//! Engine worker: each blocking task owns its [`Engine`] and its live
//! [`StreamState`]. Connection tasks talk to it only through channels.
//!
//! Work flows through a bounded FIFO channel per worker lane. Because
//! one channel carries both control and audio for each assigned session, per-session
//! ordering is preserved and control items stay ordered relative to audio.
//! Replies go back on one unbounded channel per session as [`ConnEvent`]s.
//!
//! # Scheduling and fairness
//!
//! The worker runs exactly one engine call per work item and takes items in
//! the order they entered the channel, so the channel order is the scheduling
//! policy. No mutex sits on the engine or the queue.
//!
//! Because the channel is FIFO, a session's item waits only behind items that
//! were already queued when it arrived. Per-session backpressure limits how
//! many items one session can have in the channel at once, call it `cap`. If
//! `R` sessions have work queued, an item therefore waits behind at most
//! `cap * (R - 1)` items from other sessions before the engine reaches it. A
//! session that sends audio faster than the engine can consume it fills only
//! its own allowance and cannot starve the others.

use futures_util::stream::{FuturesUnordered, StreamExt};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;

use crate::engine::{Engine, EngineError, EngineEvent};
use crate::session::{SessionError, SessionId, StreamState};

/// Audio items one session may have queued or running at once.
pub const DEFAULT_IN_FLIGHT_PER_SESSION: usize = 16;

/// Sessions the work channel is sized for until Phase 8 provides
/// `max_sessions`.
pub const DEFAULT_CHANNEL_SESSIONS: usize = 64;

/// How long a connection waits for room before its session is reported busy.
///
/// A realtime client produces one chunk every few hundred milliseconds, so
/// waiting this long means the engine is well behind on this session.
pub const DEFAULT_PUSH_TIMEOUT: Duration = Duration::from_secs(5);

/// Queue depths and timeouts for the worker and its sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerLimits {
    /// Depth of the shared work channel.
    pub channel_capacity: usize,
    /// Audio items one session may have queued or running at once.
    pub in_flight_per_session: usize,
    /// How long `push_audio` waits for room before returning `Busy`.
    pub push_timeout: Duration,
}

impl Default for WorkerLimits {
    fn default() -> Self {
        Self {
            channel_capacity: DEFAULT_CHANNEL_SESSIONS * DEFAULT_IN_FLIGHT_PER_SESSION,
            in_flight_per_session: DEFAULT_IN_FLIGHT_PER_SESSION,
            push_timeout: DEFAULT_PUSH_TIMEOUT,
        }
    }
}

/// One unit of work for the engine worker.
pub enum WorkItem {
    /// Register `stream` under `id` and run [`Engine::open_stream`].
    OpenStream {
        id: SessionId,
        stream: StreamState<()>,
        admission: Option<OwnedSemaphorePermit>,
        reply_tx: mpsc::UnboundedSender<ConnEvent>,
        cancelled: Arc<AtomicBool>,
    },
    /// Run one [`Engine::push_audio`] on `id`.
    ///
    /// `permit` is one unit of the session's in-flight allowance. It is
    /// released when this item is dropped, whether the step ran or was
    /// skipped.
    PushAudio {
        id: SessionId,
        pcm16: Vec<i16>,
        permit: OwnedSemaphorePermit,
    },
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
    /// One audio step ran; carries the partial transcript if one was produced.
    Partial(Option<EngineEvent>),
    /// Stream finalized; carries the final transcript event.
    Finalized(EngineEvent),
    /// Stream cancelled without a final transcript.
    Cancelled,
    /// Engine or session failure; the session has been dropped.
    Failed(SessionError),
}

struct Slot<C> {
    stream: StreamState<C>,
    reply_tx: mpsc::UnboundedSender<ConnEvent>,
    cancelled: Arc<AtomicBool>,
    _admission: Option<OwnedSemaphorePermit>,
}

struct Lane {
    tx: mpsc::Sender<WorkItem>,
    admission: Option<Arc<Semaphore>>,
}

/// Handle to the engine worker. Cheap to clone; dropping the last clone lets
/// the worker drain and exit.
pub struct EngineWorker<E: Engine> {
    lanes: Arc<Vec<Lane>>,
    _engine: PhantomData<fn() -> E>,
    live: Arc<AtomicUsize>,
    next_id: Arc<AtomicU64>,
    limits: WorkerLimits,
}

impl<E: Engine> Clone for EngineWorker<E> {
    fn clone(&self) -> Self {
        Self {
            lanes: Arc::clone(&self.lanes),
            _engine: PhantomData,
            live: Arc::clone(&self.live),
            next_id: Arc::clone(&self.next_id),
            limits: self.limits,
        }
    }
}

impl<E: Engine + 'static> EngineWorker<E> {
    /// Spawn the worker on Tokio's blocking pool with default limits.
    pub fn spawn(engine: E) -> (Self, JoinHandle<()>) {
        Self::spawn_with_limits(engine, WorkerLimits::default())
    }

    /// Spawn the worker with explicit limits.
    pub fn spawn_with_limits(engine: E, limits: WorkerLimits) -> (Self, JoinHandle<()>) {
        Self::spawn_lanes(vec![engine], limits, false)
    }

    /// One dedicated worker per admitted session. Engines must share their
    /// model when native batching is desired. Each lane admits one session,
    /// pins its state to one blocking thread, and preserves its FIFO order.
    /// Admission is fail-fast; excess sessions receive Busy before native open.
    pub fn spawn_pool(engines: Vec<E>, limits: WorkerLimits) -> (Self, JoinHandle<()>) {
        assert!(!engines.is_empty(), "worker pool must not be empty");
        Self::spawn_lanes(engines, limits, true)
    }

    fn spawn_lanes(engines: Vec<E>, limits: WorkerLimits, bounded: bool) -> (Self, JoinHandle<()>) {
        let live = Arc::new(AtomicUsize::new(0));
        let mut joins = FuturesUnordered::new();
        let mut lanes = Vec::with_capacity(engines.len());
        for engine in engines {
            let (tx, rx) = mpsc::channel(limits.channel_capacity);
            let live_worker = Arc::clone(&live);
            joins.push(tokio::task::spawn_blocking(move || {
                worker_main(engine, rx, live_worker)
            }));
            lanes.push(Lane {
                tx,
                admission: bounded.then(|| Arc::new(Semaphore::new(1))),
            });
        }
        let join = tokio::spawn(async move {
            while let Some(result) = joins.next().await {
                match result {
                    Ok(()) => {}
                    Err(err) if err.is_panic() => std::panic::resume_unwind(err.into_panic()),
                    Err(err) => panic!("engine worker task failed: {err}"),
                }
            }
        });
        (
            Self {
                lanes: Arc::new(lanes),
                _engine: PhantomData,
                live,
                next_id: Arc::new(AtomicU64::new(0)),
                limits,
            },
            join,
        )
    }

    /// Limits this worker and its sessions run under.
    pub fn limits(&self) -> WorkerLimits {
        self.limits
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
        stream: StreamState<()>,
    ) -> Result<SessionHandle, SessionError> {
        let (reply_tx, reply_rx) = mpsc::unbounded_channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut selected = None;
        let start = (id.as_u64() % self.lanes.len() as u64) as usize;
        for offset in 0..self.lanes.len() {
            let lane = &self.lanes[(start + offset) % self.lanes.len()];
            match &lane.admission {
                None => {
                    selected = Some((lane, None));
                    break;
                }
                Some(gate) => {
                    if let Ok(permit) = Arc::clone(gate).try_acquire_owned() {
                        selected = Some((lane, Some(permit)));
                        break;
                    }
                }
            }
        }
        let (lane, admission) = selected.ok_or(SessionError::Busy)?;
        send_item(
            &lane.tx,
            WorkItem::OpenStream {
                id,
                stream,
                admission,
                reply_tx,
                cancelled: Arc::clone(&cancelled),
            },
        )
        .await?;
        Ok(SessionHandle {
            id,
            work_tx: lane.tx.clone(),
            reply_rx,
            cancelled,
            permits: Arc::new(Semaphore::new(self.limits.in_flight_per_session)),
            push_timeout: self.limits.push_timeout,
        })
    }
}

/// Connection-side handle for one session: sends work, receives events.
pub struct SessionHandle {
    id: SessionId,
    work_tx: mpsc::Sender<WorkItem>,
    reply_rx: mpsc::UnboundedReceiver<ConnEvent>,
    cancelled: Arc<AtomicBool>,
    permits: Arc<Semaphore>,
    push_timeout: Duration,
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        // Normal connection cleanup awaits cancel. This covers an aborted
        // connection future: a full lane already has work which will sweep
        // closed reply channels before processing its next item.
        let _ = self
            .work_tx
            .try_send(WorkItem::CancelStream { id: self.id });
    }
}

impl SessionHandle {
    /// Session id this handle controls.
    pub fn id(&self) -> SessionId {
        self.id
    }

    /// Wait for the next worker event. `None` means the worker is gone.
    pub async fn recv(&mut self) -> Option<ConnEvent> {
        self.reply_rx.recv().await
    }

    /// Queue one chunk of PCM16 for the engine.
    ///
    /// Waits for one unit of this session's in-flight allowance and for room
    /// in the work channel. If both are not obtained within the push timeout
    /// the chunk is dropped and [`SessionError::Busy`] is returned.
    pub async fn push_audio(&self, pcm16: Vec<i16>) -> Result<(), SessionError> {
        let queued = tokio::time::timeout(self.push_timeout, async {
            let permit = Arc::clone(&self.permits)
                .acquire_owned()
                .await
                .map_err(|_| worker_stopped())?;
            send_item(
                &self.work_tx,
                WorkItem::PushAudio {
                    id: self.id,
                    pcm16,
                    permit,
                },
            )
            .await
        })
        .await;
        match queued {
            Ok(result) => result,
            Err(_) => Err(SessionError::Busy),
        }
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

async fn send_item(tx: &mpsc::Sender<WorkItem>, item: WorkItem) -> Result<(), SessionError> {
    tx.send(item).await.map_err(|_| worker_stopped())
}

fn worker_stopped() -> SessionError {
    SessionError::Engine(EngineError::Failed("engine worker stopped".into()))
}

fn worker_main<E: Engine>(
    mut engine: E,
    mut work_rx: mpsc::Receiver<WorkItem>,
    live: Arc<AtomicUsize>,
) {
    let mut slots: HashMap<SessionId, Slot<E::CallState>> = HashMap::new();

    while let Some(item) = work_rx.blocking_recv() {
        let abandoned: Vec<_> = slots
            .iter()
            .filter(|(_, slot)| slot.reply_tx.is_closed())
            .map(|(&id, _)| id)
            .collect();
        for id in abandoned {
            if let Some(slot) = remove(&mut slots, &live, id) {
                let _ = slot.stream.cancel(&mut engine);
            }
        }
        match item {
            WorkItem::OpenStream {
                id,
                stream,
                admission,
                reply_tx,
                cancelled,
            } => {
                let mut stream = stream.into_worker();
                if cancelled.load(Ordering::Acquire) || reply_tx.is_closed() {
                    continue;
                }
                match engine.open_stream(&mut stream) {
                    Ok(()) => {
                        slots.insert(
                            id,
                            Slot {
                                stream,
                                reply_tx,
                                cancelled,
                                _admission: admission,
                            },
                        );
                        live.fetch_add(1, Ordering::AcqRel);
                        deliver(&mut engine, &mut slots, &live, id, ConnEvent::Opened);
                    }
                    Err(err) => {
                        let _ = reply_tx.send(ConnEvent::Failed(SessionError::Engine(err)));
                    }
                }
            }
            WorkItem::PushAudio {
                id,
                pcm16,
                permit: _permit,
            } => {
                let Some(slot) = slots.get_mut(&id) else {
                    continue;
                };
                if slot.cancelled.load(Ordering::Acquire) {
                    continue;
                }
                match engine.push_audio(&mut slot.stream, &pcm16) {
                    Ok(event) => {
                        deliver(
                            &mut engine,
                            &mut slots,
                            &live,
                            id,
                            ConnEvent::Partial(event),
                        );
                    }
                    Err(err) => {
                        fail(
                            &mut engine,
                            &mut slots,
                            &live,
                            id,
                            SessionError::Engine(err),
                        );
                    }
                }
            }
            WorkItem::FinalizeStream { id } => {
                let Some(slot) = remove(&mut slots, &live, id) else {
                    continue;
                };
                let event = match slot.stream.finalize(&mut engine) {
                    Ok(event) => ConnEvent::Finalized(event),
                    Err(err) => ConnEvent::Failed(err),
                };
                let _ = slot.reply_tx.send(event);
            }
            WorkItem::CancelStream { id } => {
                let Some(slot) = remove(&mut slots, &live, id) else {
                    continue;
                };
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
    let Some(slot) = remove(slots, live, id) else {
        return;
    };
    let reply_tx = slot.reply_tx.clone();
    let _ = slot.stream.cancel(engine);
    let _ = reply_tx.send(ConnEvent::Failed(err));
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::{ConnEvent, EngineWorker, WorkerLimits};
    use crate::engine::{Engine, EngineError, EngineEvent, MockEngine};
    use crate::session::{SessionError, SessionId, StreamState};

    /// Mock that records which session each audio step belonged to.
    #[derive(Clone, Default)]
    struct RecordingEngine {
        steps: Arc<Mutex<Vec<SessionId>>>,
    }

    impl Engine for RecordingEngine {
        type CallState = ();

        fn open_stream(&mut self, _s: &mut StreamState<()>) -> Result<(), EngineError> {
            Ok(())
        }

        fn push_audio(
            &mut self,
            state: &mut StreamState<()>,
            pcm16: &[i16],
        ) -> Result<Option<EngineEvent>, EngineError> {
            self.steps.lock().unwrap().push(state.session_id());
            MockEngine::new().push_audio(state, pcm16)
        }

        fn finalize(&mut self, state: &mut StreamState<()>) -> Result<EngineEvent, EngineError> {
            MockEngine::new().finalize(state)
        }

        fn cancel(&mut self, _s: &mut StreamState<()>) -> Result<(), EngineError> {
            Ok(())
        }
    }

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
        ) -> Result<Option<EngineEvent>, EngineError> {
            self.pushes.fetch_add(1, Ordering::SeqCst);
            MockEngine::new().push_audio(state, pcm16)
        }

        fn finalize(&mut self, state: &mut StreamState<()>) -> Result<EngineEvent, EngineError> {
            MockEngine::new().finalize(state)
        }

        fn cancel(&mut self, _s: &mut StreamState<()>) -> Result<(), EngineError> {
            self.cancels.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// Skip partials and return on `Finalized`; anything else fails the test.
    async fn recv_finalized(session: &mut super::SessionHandle) {
        loop {
            match session.recv().await {
                Some(ConnEvent::Partial(_)) => {}
                Some(ConnEvent::Finalized(_)) => return,
                other => panic!("expected Finalized, got {other:?}"),
            }
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
        let mut session = worker.open(id, StreamState::new(id)).await.unwrap();

        assert_eq!(session.recv().await, Some(ConnEvent::Opened));
        assert_eq!(worker.live_sessions(), 1);

        session.push_audio(vec![0]).await.unwrap();
        assert_eq!(
            session.recv().await,
            Some(ConnEvent::Partial(Some(EngineEvent::Partial {
                text: "partial-1".into()
            })))
        );

        session.finalize().await.unwrap();
        assert_eq!(
            session.recv().await,
            Some(ConnEvent::Finalized(EngineEvent::Final {
                text: "final-1".into()
            }))
        );
        assert_eq!(worker.live_sessions(), 0);
    }

    #[tokio::test]
    async fn queued_chunks_from_two_sessions_interleave_in_arrival_order() {
        let engine = RecordingEngine::default();
        let steps = Arc::clone(&engine.steps);
        let (worker, _join) = EngineWorker::spawn(engine);

        let a_id = worker.next_session_id();
        let b_id = worker.next_session_id();
        let mut a = worker.open(a_id, StreamState::new(a_id)).await.unwrap();
        let mut b = worker.open(b_id, StreamState::new(b_id)).await.unwrap();
        assert_eq!(a.recv().await, Some(ConnEvent::Opened));
        assert_eq!(b.recv().await, Some(ConnEvent::Opened));

        // Alternate chunks, then let A queue a burst before B's last chunk.
        for _ in 0..3 {
            a.push_audio(vec![0]).await.unwrap();
            b.push_audio(vec![0]).await.unwrap();
        }
        for _ in 0..4 {
            a.push_audio(vec![0]).await.unwrap();
        }
        b.push_audio(vec![0]).await.unwrap();

        a.finalize().await.unwrap();
        b.finalize().await.unwrap();
        recv_finalized(&mut a).await;
        recv_finalized(&mut b).await;

        let expected = [
            a_id, b_id, a_id, b_id, a_id, b_id, a_id, a_id, a_id, a_id, b_id,
        ];
        assert_eq!(steps.lock().unwrap().as_slice(), &expected);
    }

    #[tokio::test]
    async fn cancelled_session_skips_queued_audio() {
        let engine = CountingEngine::default();
        let pushes = Arc::clone(&engine.pushes);
        let (worker, _join) = EngineWorker::spawn(engine);
        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id)).await.unwrap();
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
        let mut ghost_session = worker.open(ghost, StreamState::new(ghost)).await.unwrap();
        assert_eq!(ghost_session.recv().await, Some(ConnEvent::Opened));
        ghost_session.cancel().await.unwrap();
        assert_eq!(ghost_session.recv().await, Some(ConnEvent::Cancelled));

        // Second cancel and a push for the same id must be silently dropped.
        ghost_session.cancel().await.unwrap();
        ghost_session.push_audio(vec![0]).await.unwrap();

        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id)).await.unwrap();
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
        let mut session = worker.open(id, StreamState::new(id)).await.unwrap();

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
        let mut session = worker.open(id, StreamState::new(id)).await.unwrap();
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
        let mut session = worker.open(id, StreamState::new(id)).await.unwrap();
        assert_eq!(session.recv().await, Some(ConnEvent::Opened));

        // Simulate a connection that vanished without sending CancelStream.
        let work_tx = session.work_tx.clone();
        let permit = Arc::clone(&session.permits).acquire_owned().await.unwrap();
        drop(session);
        work_tx
            .send(super::WorkItem::PushAudio {
                id,
                pcm16: vec![0],
                permit,
            })
            .await
            .unwrap();

        wait_until(|| worker.live_sessions() == 0).await;
        assert_eq!(cancels.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn engine_panic_surfaces_on_join_handle_and_ends_sessions() {
        struct PanickingEngine;

        impl Engine for PanickingEngine {
            type CallState = ();

            fn open_stream(&mut self, _s: &mut StreamState<()>) -> Result<(), EngineError> {
                Ok(())
            }

            fn push_audio(
                &mut self,
                _s: &mut StreamState<()>,
                _pcm16: &[i16],
            ) -> Result<Option<EngineEvent>, EngineError> {
                panic!("engine step exploded");
            }

            fn finalize(&mut self, _s: &mut StreamState<()>) -> Result<EngineEvent, EngineError> {
                unreachable!()
            }

            fn cancel(&mut self, _s: &mut StreamState<()>) -> Result<(), EngineError> {
                Ok(())
            }
        }

        let (worker, join) = EngineWorker::spawn(PanickingEngine);
        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id)).await.unwrap();
        assert_eq!(session.recv().await, Some(ConnEvent::Opened));

        session.push_audio(vec![0]).await.unwrap();

        let joined = tokio::time::timeout(Duration::from_secs(2), join)
            .await
            .expect("worker task must end after the engine panics");
        let err = joined.expect_err("join must report the panic");
        assert!(err.is_panic());
        assert_eq!(
            crate::panic_message(err.into_panic().as_ref()),
            "engine step exploded"
        );

        // The reply channel closed with the worker, so the connection sees
        // the worker as gone.
        assert_eq!(session.recv().await, None);
    }

    #[tokio::test]
    async fn push_audio_reports_busy_when_in_flight_allowance_is_exhausted() {
        let step = Duration::from_millis(200);
        let limits = WorkerLimits {
            channel_capacity: 64,
            in_flight_per_session: 2,
            push_timeout: Duration::from_millis(20),
        };
        let (worker, _join) =
            EngineWorker::spawn_with_limits(MockEngine::with_step_delay(step), limits);
        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id)).await.unwrap();
        assert_eq!(session.recv().await, Some(ConnEvent::Opened));

        // Two chunks fill the allowance: one is running, one is queued.
        session.push_audio(vec![0]).await.unwrap();
        session.push_audio(vec![1]).await.unwrap();

        // The third cannot get a permit within the push timeout because the
        // engine step holding the first permit takes far longer than that.
        assert_eq!(session.push_audio(vec![2]).await, Err(SessionError::Busy));

        // Once the engine finishes a step its permit is released and the
        // session accepts audio again.
        assert!(matches!(session.recv().await, Some(ConnEvent::Partial(_))));
        session.push_audio(vec![3]).await.unwrap();
    }

    #[tokio::test]
    async fn skipped_audio_after_cancel_still_releases_permits() {
        let limits = WorkerLimits {
            channel_capacity: 64,
            in_flight_per_session: 2,
            push_timeout: Duration::from_millis(20),
        };
        let (worker, _join) = EngineWorker::spawn_with_limits(MockEngine::new(), limits);
        let id = worker.next_session_id();
        let mut session = worker.open(id, StreamState::new(id)).await.unwrap();
        assert_eq!(session.recv().await, Some(ConnEvent::Opened));

        session.cancelled.store(true, Ordering::Release);
        session.push_audio(vec![0]).await.unwrap();
        session.push_audio(vec![1]).await.unwrap();
        session.cancel().await.unwrap();
        assert_eq!(session.recv().await, Some(ConnEvent::Cancelled));

        // Both skipped items have been dropped by the worker, so the full
        // allowance is available again.
        assert_eq!(session.permits.available_permits(), 2);
    }
    #[tokio::test]
    async fn pool_admission_is_bounded_and_reuses_released_lane() {
        let (worker, join) = EngineWorker::spawn_pool(
            vec![MockEngine::new(), MockEngine::new()],
            WorkerLimits::default(),
        );
        let mut sessions = Vec::new();
        for _ in 0..2 {
            let id = worker.next_session_id();
            let mut session = worker.open(id, StreamState::new(id)).await.unwrap();
            assert_eq!(session.recv().await, Some(ConnEvent::Opened));
            sessions.push(session);
        }
        let id = worker.next_session_id();
        assert!(matches!(
            worker.open(id, StreamState::new(id)).await,
            Err(SessionError::Busy)
        ));
        sessions[0].cancel().await.unwrap();
        assert_eq!(sessions[0].recv().await, Some(ConnEvent::Cancelled));
        // The terminal event precedes release by at most the end of this worker
        // iteration. Wait on the admission permit, rather than live count.
        wait_until(|| {
            worker
                .lanes
                .iter()
                .any(|lane| lane.admission.as_ref().unwrap().available_permits() == 1)
        })
        .await;
        let id = worker.next_session_id();
        let mut replacement = worker.open(id, StreamState::new(id)).await.unwrap();
        assert_eq!(replacement.recv().await, Some(ConnEvent::Opened));
        assert_eq!(worker.live_sessions(), 2);
        drop(replacement);
        drop(sessions);
        wait_until(|| worker.live_sessions() == 0).await;
        drop(worker);
        tokio::time::timeout(Duration::from_secs(2), join)
            .await
            .unwrap()
            .unwrap();
    }

    struct ThreadLocalCall {
        owner: std::thread::ThreadId,
        _not_send: std::rc::Rc<()>,
    }
    impl Drop for ThreadLocalCall {
        fn drop(&mut self) {
            assert_eq!(self.owner, std::thread::current().id());
        }
    }
    struct LocalEngine {
        rendezvous: Arc<(Mutex<usize>, std::sync::Condvar)>,
    }
    impl Engine for LocalEngine {
        type CallState = ThreadLocalCall;
        fn open_stream(
            &mut self,
            state: &mut StreamState<Self::CallState>,
        ) -> Result<(), EngineError> {
            state.set_engine_call(ThreadLocalCall {
                owner: std::thread::current().id(),
                _not_send: std::rc::Rc::new(()),
            });
            Ok(())
        }
        fn push_audio(
            &mut self,
            state: &mut StreamState<Self::CallState>,
            _: &[i16],
        ) -> Result<Option<EngineEvent>, EngineError> {
            assert_eq!(
                state.engine_call().unwrap().owner,
                std::thread::current().id()
            );
            // Fail within a bounded time if the pool accidentally serializes
            // calls, rather than deadlocking the test's blocking threads.
            let (count, ready) = &*self.rendezvous;
            let mut count = count.lock().unwrap();
            *count += 1;
            ready.notify_all();
            let (count, _) = ready
                .wait_timeout_while(count, Duration::from_secs(2), |count| *count < 2)
                .unwrap();
            assert_eq!(*count, 2, "engine calls must overlap");
            Ok(None)
        }
        fn finalize(
            &mut self,
            state: &mut StreamState<Self::CallState>,
        ) -> Result<EngineEvent, EngineError> {
            state.clear_engine_call();
            Ok(EngineEvent::Final {
                text: String::new(),
            })
        }
        fn cancel(&mut self, state: &mut StreamState<Self::CallState>) -> Result<(), EngineError> {
            state.clear_engine_call();
            Ok(())
        }
    }

    #[tokio::test]
    async fn pool_keeps_non_send_state_on_one_thread_and_overlaps_calls() {
        let rendezvous = Arc::new((Mutex::new(0), std::sync::Condvar::new()));
        let engines = (0..2)
            .map(|_| LocalEngine {
                rendezvous: Arc::clone(&rendezvous),
            })
            .collect();
        let (worker, join) = EngineWorker::spawn_pool(engines, WorkerLimits::default());
        let mut sessions = Vec::new();
        for _ in 0..2 {
            let id = worker.next_session_id();
            let mut session = worker.open(id, StreamState::new(id)).await.unwrap();
            assert_eq!(session.recv().await, Some(ConnEvent::Opened));
            session.push_audio(vec![0]).await.unwrap();
            sessions.push(session);
        }
        for session in &mut sessions {
            assert_eq!(session.recv().await, Some(ConnEvent::Partial(None)));
            session.finalize().await.unwrap();
            recv_finalized(session).await;
        }
        drop(sessions);
        drop(worker);
        tokio::time::timeout(Duration::from_secs(2), join)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn failed_open_returns_admission_slot() {
        let (worker, join) = EngineWorker::spawn_pool(
            vec![CountingEngine {
                fail_open: true,
                ..CountingEngine::default()
            }],
            WorkerLimits::default(),
        );
        for _ in 0..3 {
            let id = worker.next_session_id();
            let mut session = worker.open(id, StreamState::new(id)).await.unwrap();
            assert!(matches!(session.recv().await, Some(ConnEvent::Failed(_))));
            wait_until(|| {
                worker.lanes[0]
                    .admission
                    .as_ref()
                    .unwrap()
                    .available_permits()
                    == 1
            })
            .await;
        }
        assert_eq!(worker.live_sessions(), 0);
        drop(worker);
        join.await.unwrap();
    }
}
