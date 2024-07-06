#[derive(Debug)]
pub enum Error {
    Unknown,
    Executor,
    Overflow,
    MissingTensor,
    MissingNodeInGraph,
    CudnnInternal,
    AllocationFailed,
}
