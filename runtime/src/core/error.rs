use crate::core::allocators::ArenaId;
use crate::core::instance::BuilderError;
use crate::Value;
use rmlk_schema::{DataType, Op};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, InternalError>;

#[derive(Debug)]
pub enum Error {
    Computation {
        op: Op,
        name: String,
        error: Box<dyn std::error::Error>,
    },
    Internal {
        error: Box<dyn std::error::Error>,
    },
    ModelDeserializationFailed,
    NodeNotFound {
        id: usize,
    },
    ExpectedName {
        node_id: usize,
    },
    InvalidUserInput {
        input: HashMap<String, Value>,
    },
    FailedToFindNodeId {
        name: String,
    },
    ModelBuildFailed {
        error: BuilderError,
    },
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for Error {}

impl From<InternalError> for Error {
    fn from(value: InternalError) -> Self {
        Self::Internal {
            error: Box::new(value),
        }
    }
}

impl From<BuilderError> for Error {
    fn from(value: BuilderError) -> Self {
        Error::ModelBuildFailed { error: value }
    }
}

#[derive(Debug)]
pub enum InternalError {
    Attribute {
        inner: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
    AxisOutOfBounds {
        axis: i64,
    },
    BufferSizeMismatch {
        expected: usize,
        actual: usize,
    },
    Device {
        error: rmlk_cuda::Error,
    },
    ExecutionState(String),
    InvalidByteLength,
    InvalidTensorIndex {
        index: usize,
    },
    InvalidAttribute {
        name: String,
    },
    InvalidAttributeDataType {
        name: String,
    },
    InvalidRange {
        start: i64,
        end: i64,
    },
    InvalidTensorShape {
        shape: Vec<usize>,
    },
    InvalidMemoryAllocation {
        message: String,
    },
    IncompatibleTensorShape {
        shapes: HashMap<usize, Vec<usize>>,
    },
    IncompatibleShapesForBroadcast {
        shapes: HashMap<usize, Vec<usize>>,
    },
    InvalidInput {
        input: usize,
        op: Op,
        message: String,
    },
    MissingDeviceData,
    MissingAttributes,
    MissingAttribute {
        name: String,
    },
    MissingNode {
        id: usize,
    },
    MissingOutputNode {
        op: Op,
    },
    ExpectedNodeInfo {
        info: String,
        node_id: usize,
    },
    ExpectedShapeInDef {
        node_id: usize,
    },
    ExpectedDataTypeInDef {
        node_id: usize,
    },
    UnsupportedRankSize {
        message: String,
    },
    TensorStore(String),
    TensorNotFound {
        node_id: usize,
    },
    TensorNotFoundFromIndex {
        id: usize,
    },
    TensorIndexNotFound {
        node_id: usize,
    },
    UnableToConvertValue,
    UnexpectedTensorDataType {
        expected: DataType,
    },
    UnknownShapeBuffer {
        index: ArenaId,
    },
    UnsupportedDataType {
        dtype: DataType,
    },
    UnsupportedOp {
        op: Op,
    },
    UnsupportedInputValues {
        message: String,
    },
    ScalarInputsAreNotAllowed,
    CudaBumpAllocatorFailed,
}

impl InternalError {
    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

impl Display for InternalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InternalError::Attribute { inner } => write!(f, "{}", inner),
            InternalError::AxisOutOfBounds { axis } => {
                write!(f, "invalid axis `{axis}`")
            }
            InternalError::TensorStore(msg) => {
                write!(f, "tensor store error `{msg}`")
            }
            InternalError::ExecutionState(msg) => {
                write!(f, "execution state error `{msg}`")
            }
            InternalError::ExpectedDataTypeInDef { node_id } => {
                write!(f, "expected data type in node `{node_id}`")
            }
            InternalError::ExpectedShapeInDef { node_id } => {
                write!(f, "expected shape in node `{node_id}`")
            }
            InternalError::ExpectedNodeInfo { info, node_id } => {
                write!(f, "expected node info `{info}` from node `{node_id}`")
            }
            InternalError::Device { error: msg } => {
                write!(f, "device error `{msg:?}`")
            }
            InternalError::TensorNotFound { node_id } => {
                write!(f, "failed to find tensor: {node_id:?}")
            }
            InternalError::TensorNotFoundFromIndex { id } => {
                write!(f, "failed to find tensor from index: {id:?}")
            }
            InternalError::TensorIndexNotFound { node_id } => {
                write!(f, "failed to find tensor index: {node_id:?}")
            }
            InternalError::IncompatibleTensorShape { shapes } => {
                write!(f, "incompatible shapes `{shapes:?}`")
            }
            InternalError::IncompatibleShapesForBroadcast { shapes } => {
                write!(f, "incompatible shapes {shapes:?} for broadcast")
            }
            InternalError::InvalidByteLength => {
                write!(f, "invalid byte length")
            }
            InternalError::InvalidRange { start, end } => {
                write!(f, "invalid range start={start}, end={end}")
            }
            InternalError::InvalidTensorIndex { index } => {
                write!(f, "invalid tensor index: {index:?}")
            }
            InternalError::InvalidInput { input, op, message } => {
                write!(f, "invalid input `{input:?}` for op `{op:?}`: {message:?}")
            }
            InternalError::UnableToConvertValue => {
                write!(f, "unable to convert value")
            }
            InternalError::MissingDeviceData => {
                write!(f, "missing device data")
            }
            InternalError::MissingAttributes => {
                write!(f, "missing attributes")
            }
            InternalError::MissingAttribute { name } => {
                write!(f, "missing `{name}` attribute")
            }
            InternalError::MissingNode { id } => {
                write!(f, "missing node `{id}`")
            }
            InternalError::MissingOutputNode { op } => {
                write!(f, "missing output node `{op:?}`")
            }
            InternalError::InvalidAttribute { name } => {
                write!(f, "invalid attribute `{name}`")
            }
            InternalError::InvalidAttributeDataType { name } => {
                write!(f, "invalid data type for `{name}` attribute")
            }
            InternalError::UnexpectedTensorDataType { expected } => {
                write!(
                    f,
                    "unexpected tensor data type when expected `{expected:?}`"
                )
            }
            InternalError::UnsupportedDataType { dtype } => {
                write!(f, "unsupported `{dtype:?}` data type")
            }
            InternalError::UnsupportedOp { op } => {
                write!(f, "unsupported `{op:?}` op")
            }
            InternalError::UnsupportedRankSize { message } => {
                write!(f, "unsupported rank size `{message}`")
            }
            InternalError::InvalidTensorShape { shape } => {
                write!(f, "invalid tensor shape `{shape:?}`")
            }
            InternalError::InvalidMemoryAllocation { message } => {
                write!(f, "invalid memory allocation `{message}`")
            }
            InternalError::BufferSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "expected buffer of size `{expected}` instead of `{actual}`"
                )
            }
            InternalError::UnknownShapeBuffer { index } => {
                write!(f, "unknown buffer given index `{:?}`", index)
            }
            InternalError::UnsupportedInputValues { message } => {
                write!(f, "unsupported input values `{message}`")
            }
            InternalError::ScalarInputsAreNotAllowed => {
                write!(f, "scalar inputs are not allowed")
            }
            InternalError::CudaBumpAllocatorFailed => {
                write!(f, "cuda-bump allocator failed")
            }
        }
    }
}

impl std::error::Error for InternalError {}
