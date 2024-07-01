use std::sync::Arc;
use rmlk_graph::Graph;


// API: public. loads model from file or memory.
// An inference session.
// This contains session options.
// Think of this as data for running sessions or for session that ran
// vs SessionState that is local to the session.
pub struct Session<A> {
    alloc: A,
    state: SessionState<A>,
    // Options for sessionstate.
}



// API: private. Does not offer services for users to load/save model.
// Does Session map to a graph?
// REad-only state passed to each executor.
// Could contain metrics as well.
// This will be passed to executors as read only information about session
struct SessionState<A> {
    graph: Arc<Graph<A>>,
    plan: (),
}

impl<A> SessionState<A> {
}