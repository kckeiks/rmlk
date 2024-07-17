#[derive(Debug)]
pub enum Error {
    Unknown,
    InvalidTensorDimensions,
    InvalidAttribute,
    InvalidAttributeFormat,
    Executor,
    Overflow,
    MissingTensor,
    MissingNodeInGraph,
    CudnnInternal,
    AllocationFailed,
    MissingAttributes,
    UnsupportedShape,
    OutputShapeMismatch,
    ComputationError,
}
