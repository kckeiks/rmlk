use crate::Value;
use rmlk_schema::{DataType, Op};
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, InternalError>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("inference failed: {error}")]
    InferenceError {
        error: anyhow::Error,
    },
    #[error("invalid inputs:\nreceived: {received:?}\nexpected input names: {expected:?}")]
    InvalidInputs {
        received: HashMap<String, Value>,
        expected: Vec<String>,
    },
    #[error("unknown input name `{name}`")]
    UnknownInputName {
        name: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum InternalError {
    #[error("attribute error: {inner}")]
    Attribute {
        inner: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
    #[error("axis out of bounds: axis {axis}")]
    AxisOutOfBounds {
        axis: i64,
    },
    #[error("buffer size mismatch: expected {expected} bytes, got {actual} bytes")]
    BufferSizeMismatch {
        expected: usize,
        actual: usize,
    },
    #[error("device error: {error}")]
    Device {
        error: rmlk_cuda::Error,
    },
    #[error("invalid execution state: {0}")]
    ExecutionState(String),
    #[error("invalid byte length")]
    InvalidByteLength,
    #[error("invalid tensor index: {index}")]
    InvalidTensorIndex {
        index: usize,
    },
    #[error("invalid attribute: {name}")]
    InvalidAttribute {
        name: String,
    },
    #[error("invalid data type for attribute: {name}")]
    InvalidAttributeDataType {
        name: String,
    },
    #[error("invalid range: start {start}, end {end}")]
    InvalidRange {
        start: i64,
        end: i64,
    },
    #[error("invalid tensor shape: {shape:?}")]
    InvalidTensorShape {
        shape: Vec<usize>,
    },
    #[error("invalid memory allocation: {message}")]
    InvalidMemoryAllocation {
        message: String,
    },
    #[error("incompatible tensor shapes: {shapes:?}")]
    IncompatibleTensorShape {
        shapes: HashMap<usize, Vec<usize>>,
    },
    #[error("incompatible shapes for broadcast: {shapes:?}")]
    IncompatibleShapesForBroadcast {
        shapes: HashMap<usize, Vec<usize>>,
    },
    #[error("invalid input {input} for op {op:?}: {message}")]
    InvalidInput {
        input: usize,
        op: Op,
        message: String,
    },
    #[error("missing device data")]
    MissingDeviceData,
    #[error("missing attributes")]
    MissingAttributes,
    #[error("missing attribute: {name}")]
    MissingAttribute {
        name: String,
    },
    #[error("missing node with id {id}")]
    MissingNode {
        id: usize,
    },
    #[error("missing output node for op {op:?}")]
    MissingOutputNode {
        op: Op,
    },
    #[error("expected node info `{info}` for node {node_id}")]
    ExpectedNodeInfo {
        info: String,
        node_id: usize,
    },
    #[error("expected shape in definition for node {node_id}")]
    ExpectedShapeInDef {
        node_id: usize,
    },
    #[error("expected data type in definition for node {node_id}")]
    ExpectedDataTypeInDef {
        node_id: usize,
    },
    #[error("unsupported rank size: {message}")]
    UnsupportedRankSize {
        message: String,
    },
    #[error("tensor store error: {0}")]
    TensorStore(String),
    #[error("tensor not found for node {node_id}")]
    TensorNotFound {
        node_id: usize,
    },
    #[error("tensor not found from index {id}")]
    TensorNotFoundFromIndex {
        id: usize,
    },
    #[error("tensor index not found for node {node_id}")]
    TensorIndexNotFound {
        node_id: usize,
    },
    #[error("unable to convert value")]
    UnableToConvertValue,
    #[error("unexpected tensor data type: expected {expected:?}")]
    UnexpectedTensorDataType {
        expected: DataType,
    },
    #[error("unsupported data type: {dtype:?}")]
    UnsupportedDataType {
        dtype: DataType,
    },
    #[error("unsupported op: {op:?}")]
    UnsupportedOp {
        op: Op,
    },
    #[error("unsupported input values: {message}")]
    UnsupportedInputValues {
        message: String,
    },
    #[error("scalar inputs are not allowed")]
    ScalarInputsAreNotAllowed,
    #[error("CUDA bump allocator failed")]
    CudaBumpAllocatorFailed,
}

#[derive(Debug, thiserror::Error)]
pub enum BuilderError {
    #[error("input node `{id}` not found")]
    InputNodeNotFound { id: usize },
    #[error("output node `{id}` not found")]
    OutputNodeNotFound { id: usize },
    #[error("missing name for node `{id}`")]
    MissingNodeName { id: usize },
    #[error("Unexpected device failure: {error}")]
    UnexpectedDeviceFailure { error: rmlk_cuda::Error },
    #[error("failed to deserialized")]
    ModelDeserializationFailed,
    #[error("build failure: {0}")]
    Internal(#[from] anyhow::Error),
}