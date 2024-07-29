#[derive(Debug)]
pub enum Error {
    Unknown,
    InvalidBufferSize,
    InvalidInputShapes,
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
    UnsupportedDataType,
    UnsupportedAttribute,
    OutputShapeMismatch,
    ComputationError,
    NoTensorFound,
}

pub type Result<T> = std::result::Result<T, Error>;
