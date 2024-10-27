use cudarc::driver::DriverError;

pub type Result<T> = std::result::Result<T, Error>;

// Todo: Fix.
#[derive(Debug)]
pub enum Error {
    FailedToLoadKernel,
    ModelDeserializationFailed,
    ComputingPlanFailed,
    MissingNode,
    NotSupportedDD,
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
    Cuda(u32),
}

impl From<DriverError> for Error {
    fn from(value: DriverError) -> Self {
        Self::Cuda(value.0 as u32)
    }
}
