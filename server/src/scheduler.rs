//! Ready queue for concurrent B=1 engine steps.
//!
//! When a connection task has engine work for a stream (new audio, later
//! finalize, and so on), it enqueues that session's id. A worker pops the next
//! id and runs one step on that session's stream state. This module is the
//! queue only — the worker loop lands in a later checklist item.

use std::collections::{HashSet, VecDeque};

use crate::session::SessionId;

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

#[cfg(test)]
mod tests {
    use super::Scheduler;
    use crate::session::SessionId;

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
}
