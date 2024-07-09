#[derive(Debug)]
pub enum Error {
    Unknown,
    InvalidInputDimensions,
    InvalidAttribute,
    InvalidAttributeFormat,
    Executor,
    Overflow,
    MissingTensor,
    MissingNodeInGraph,
    CudnnInternal,
    AllocationFailed,
    MissingAttributes,
}
