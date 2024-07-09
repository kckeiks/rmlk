#[derive(Debug)]
pub enum Error {
    Unknown,
    InvalidInputDimensions,
    Executor,
    Overflow,
    MissingTensor,
    MissingNodeInGraph,
    CudnnInternal,
    AllocationFailed,
}
