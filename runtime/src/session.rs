// This thing has general information about session
pub struct Session {
    executor: (),
    state: SessionState,
}

// This has more specific information about how to run the computation
pub struct SessionState {

}

impl SessionState {
    pub fn execute_graph(&self)
}