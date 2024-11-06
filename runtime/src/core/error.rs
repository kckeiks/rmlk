use cudarc::driver::DriverError;

pub type Result<T> = std::result::Result<T, Error>;

// Todo: Fix.
#[derive(Debug)]
pub enum Error {
    ComputationFailed,
    Cuda(u32),
    Device(String),
    Input(String),
    Internal(String),
    InvalidAttribute(String),
    InvalidTensor(String),
    InvalidTensorDimensions(Vec<usize>),
    ModelDeserializationFailed,
    MissingAttributes,
    MissingNodeInfo(String),
    NoSupport(String),
    UnknownNode(String),
    UnknownTensor(String),
}

impl From<DriverError> for Error {
    fn from(value: DriverError) -> Self {
        Self::Cuda(value.0 as u32)
    }
}
