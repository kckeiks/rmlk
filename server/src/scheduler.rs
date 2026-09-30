//! Ready queue and B=1 worker for engine steps.
//!
//! When a connection task has engine work for a stream (new audio, later
//! finalize, and so on), it enqueues that session's id. A worker pops the next
//! id and runs one step on that session's stream state (one mailbox chunk per
//! pop). HTTP wiring lands in a later checklist item.
//!
//! # Fairness
//!
//! Policy: FIFO ready queue; one engine push per [`run_one_step`] pop; if that
//! session still has mailbox audio, re-enqueue it at the back.
//!
//! Starvation bound: while `R` sessions have pending work, a ready session waits
//! at most `R - 1` other sessions' steps before its next turn. A busy stream
//! cannot monopolize the worker by draining its whole mailbox in one go.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::engine::{Engine, EngineEvent};
use crate::session::{SessionError, SessionId, StreamState};

/// FIFO of session ids that currently have pending engine work.
#[derive(Debug, Default)]
pub struct Scheduler {
    queue: VecDeque<SessionId>,
    /// Ids currently in [`Self::queue`] (rejects duplicate enqueue).
    pending: HashSet<SessionId>,
}

impl Scheduler {
    /// Empty ready queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `id` has pending engine work. Returns `false` if already queued.
    pub fn enqueue(&mut self, id: SessionId) -> bool {
        if !self.pending.insert(id) {
            return false;
        }
        self.queue.push_back(id);
        true
    }

    /// Pop the oldest session id with pending work, if any.
    pub fn pop_ready(&mut self) -> Option<SessionId> {
        let id = self.queue.pop_front()?;
        self.pending.remove(&id);
        Some(id)
    }

    /// Number of session ids with pending work in the queue.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the ready queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// One B=1 step: which session ran, and the events it produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepOutcome {
    /// Session whose stream was stepped.
    pub session_id: SessionId,
    /// Transcript events from this engine push (may be empty).
    pub events: Vec<EngineEvent>,
}

/// Pop one ready session and run a single engine push on one mailbox chunk.
///
/// If the mailbox still has audio after the push, `id` is enqueued again so
/// other ready sessions can interleave. Returns `Ok(None)` when the queue is
/// empty.
pub fn run_one_step<E: Engine>(
    sched: &mut Scheduler,
    streams: &mut HashMap<SessionId, StreamState<E::CallState>>,
    engine: &mut E,
) -> Result<Option<StepOutcome>, SessionError> {
    let Some(id) = sched.pop_ready() else {
        return Ok(None);
    };
    let stream = streams
        .get_mut(&id)
        .ok_or(SessionError::Unknown(id))?;
    let events = stream.process_one_inbound(engine)?.unwrap_or_default();
    if stream.mailbox_len() > 0 {
        sched.enqueue(id);
    }
    Ok(Some(StepOutcome {
        session_id: id,
        events,
    }))
}

/// Drain the ready queue by running [`run_one_step`] until empty.
pub fn run_until_idle<E: Engine>(
    sched: &mut Scheduler,
    streams: &mut HashMap<SessionId, StreamState<E::CallState>>,
    engine: &mut E,
) -> Result<Vec<StepOutcome>, SessionError> {
    let mut outcomes = Vec::new();
    while let Some(step) = run_one_step(sched, streams, engine)? {
        outcomes.push(step);
    }
    Ok(outcomes)
}

#[cfg(test)]
mod tests {
    use super::{run_one_step, run_until_idle, Scheduler, StepOutcome};
    use crate::engine::{Engine, EngineEvent, MockEngine};
    use crate::session::{SessionId, StreamState};
    use std::collections::HashMap;

    #[test]
    fn ready_queue_is_fifo() {
        let mut sched = Scheduler::new();
        assert!(sched.is_empty());

        assert!(sched.enqueue(SessionId::from_raw(10)));
        assert!(sched.enqueue(SessionId::from_raw(20)));
        assert!(sched.enqueue(SessionId::from_raw(30)));
        assert_eq!(sched.len(), 3);

        assert_eq!(sched.pop_ready(), Some(SessionId::from_raw(10)));
        assert_eq!(sched.pop_ready(), Some(SessionId::from_raw(20)));
        assert_eq!(sched.pop_ready(), Some(SessionId::from_raw(30)));
        assert!(sched.is_empty());
        assert_eq!(sched.pop_ready(), None);
    }

    #[test]
    fn duplicate_enqueue_is_ignored() {
        let mut sched = Scheduler::new();
        let id = SessionId::from_raw(1);
        assert!(sched.enqueue(id));
        assert!(!sched.enqueue(id));
        assert_eq!(sched.len(), 1);
        assert_eq!(sched.pop_ready(), Some(id));
        assert!(sched.enqueue(id));
    }

    #[test]
    fn worker_runs_one_mock_step_at_a_time_fifo() {
        let mut engine = MockEngine::new();
        let mut sched = Scheduler::new();
        let mut streams = HashMap::new();

        let id_a = SessionId::from_raw(1);
        let id_b = SessionId::from_raw(2);
        let mut stream_a = StreamState::new(id_a, 16);
        let mut stream_b = StreamState::new(id_b, 16);
        engine.open_stream(&mut stream_a).unwrap();
        engine.open_stream(&mut stream_b).unwrap();

        stream_a.enqueue_audio(&[0]).unwrap();
        stream_a.enqueue_audio(&[1]).unwrap();
        stream_b.enqueue_audio(&[0]).unwrap();
        streams.insert(id_a, stream_a);
        streams.insert(id_b, stream_b);

        // A enqueued first (two mailbox chunks), then B (one chunk).
        assert!(sched.enqueue(id_a));
        assert!(sched.enqueue(id_b));

        let step1 = run_one_step(&mut sched, &mut streams, &mut engine)
            .unwrap()
            .expect("step 1");
        assert_eq!(
            step1,
            StepOutcome {
                session_id: id_a,
                events: vec![EngineEvent::Partial {
                    text: "partial-1".into()
                }],
            }
        );

        let step2 = run_one_step(&mut sched, &mut streams, &mut engine)
            .unwrap()
            .expect("step 2");
        assert_eq!(
            step2,
            StepOutcome {
                session_id: id_b,
                events: vec![EngineEvent::Partial {
                    text: "partial-1".into()
                }],
            }
        );

        let step3 = run_one_step(&mut sched, &mut streams, &mut engine)
            .unwrap()
            .expect("step 3");
        assert_eq!(
            step3,
            StepOutcome {
                session_id: id_a,
                events: vec![EngineEvent::Partial {
                    text: "partial-2".into()
                }],
            }
        );

        assert!(sched.is_empty());
        assert!(run_one_step(&mut sched, &mut streams, &mut engine)
            .unwrap()
            .is_none());
    }

    #[test]
    fn run_until_idle_drains_all_ready_work() {
        let mut engine = MockEngine::new();
        let mut sched = Scheduler::new();
        let mut streams = HashMap::new();

        let id = SessionId::from_raw(7);
        let mut stream = StreamState::new(id, 16);
        engine.open_stream(&mut stream).unwrap();
        stream.enqueue_audio(&[0]).unwrap();
        stream.enqueue_audio(&[0]).unwrap();
        streams.insert(id, stream);
        assert!(sched.enqueue(id));
        assert_eq!(streams.get(&id).unwrap().mailbox_len(), 2);

        let outcomes = run_until_idle(&mut sched, &mut streams, &mut engine).unwrap();
        assert_eq!(outcomes.len(), 2);
        assert!(sched.is_empty());
        assert_eq!(streams.get(&id).unwrap().mailbox_len(), 0);
        assert_eq!(streams.get(&id).unwrap().chunks_pushed(), 2);
    }

    /// While R sessions stay ready, gaps between one session's consecutive steps
    /// are at most R - 1 other steps (FIFO + re-enqueue at back).
    #[test]
    fn ready_session_not_starved_beyond_other_ready_count() {
        const N: usize = 4;
        const CHUNKS_PER_SESSION: usize = 8;

        let mut engine = MockEngine::new();
        let mut sched = Scheduler::new();
        let mut streams = HashMap::new();

        for i in 0..N {
            let id = SessionId::from_raw(i as u64);
            let mut stream = StreamState::new(id, CHUNKS_PER_SESSION);
            engine.open_stream(&mut stream).unwrap();
            for _ in 0..CHUNKS_PER_SESSION {
                stream.enqueue_audio(&[0]).unwrap();
            }
            streams.insert(id, stream);
            assert!(sched.enqueue(id));
        }

        let outcomes = run_until_idle(&mut sched, &mut streams, &mut engine).unwrap();
        assert_eq!(outcomes.len(), N * CHUNKS_PER_SESSION);

        for i in 0..N {
            let id = SessionId::from_raw(i as u64);
            let indices: Vec<usize> = outcomes
                .iter()
                .enumerate()
                .filter_map(|(idx, step)| (step.session_id == id).then_some(idx))
                .collect();
            assert_eq!(indices.len(), CHUNKS_PER_SESSION);
            for window in indices.windows(2) {
                let gap = window[1] - window[0] - 1;
                assert!(
                    gap <= N - 1,
                    "session {id:?}: {gap} other steps between consecutive turns (bound {})",
                    N - 1
                );
            }
        }
    }

    /// A light session stuck behind a heavy one still runs within one other step.
    #[test]
    fn light_session_not_blocked_by_heavy_mailbox() {
        let mut engine = MockEngine::new();
        let mut sched = Scheduler::new();
        let mut streams = HashMap::new();

        let heavy = SessionId::from_raw(1);
        let light = SessionId::from_raw(2);
        let mut heavy_stream = StreamState::new(heavy, 32);
        let mut light_stream = StreamState::new(light, 4);
        engine.open_stream(&mut heavy_stream).unwrap();
        engine.open_stream(&mut light_stream).unwrap();
        for _ in 0..16 {
            heavy_stream.enqueue_audio(&[0]).unwrap();
        }
        light_stream.enqueue_audio(&[0]).unwrap();
        streams.insert(heavy, heavy_stream);
        streams.insert(light, light_stream);

        assert!(sched.enqueue(heavy));
        assert!(sched.enqueue(light));

        let outcomes = run_until_idle(&mut sched, &mut streams, &mut engine).unwrap();
        let light_at = outcomes
            .iter()
            .position(|step| step.session_id == light)
            .expect("light session should run");
        assert!(
            light_at <= 1,
            "light session waited {light_at} steps; bound is 1 when one other session is ahead"
        );
    }
}
