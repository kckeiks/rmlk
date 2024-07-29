pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    FailedToLoadKernel,
    ModelDeserializationFailed,
    ComputingPlanFailed,
    MissingNode,
    NotSupported,
    Unknown,
    ContextError,
    MissingData,
    UnsupportedDataType,
    ComputationFailed,
    AllocationFailed,
    InvalidAttributeFormat,
    MissingAttributes,
    InvalidAttribute,
    UnsupportedAttribute,
    InvalidTensorDimensions,
}
