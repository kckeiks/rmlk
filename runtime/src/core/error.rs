use rmlk_schema::DataType;
use std::fmt;

/// Stable prefix for [`UnsupportedDataType`]'s display string.
/// Node-suite harness matching must use this rather than a duplicate literal.
pub const UNSUPPORTED_DATA_TYPE_PREFIX: &str = "unsupported data type";

/// Stable prefix for unsupported Cast errors.
/// Node-suite harness matching must use this rather than a duplicate literal.
pub const UNSUPPORTED_CAST_PREFIX: &str = "unsupported cast";

#[derive(Debug, thiserror::Error)]
#[error("inference failed: {0}")]
pub struct InferenceError(#[from] pub(crate) anyhow::Error);

// Todo: add src and dst information.
#[derive(Debug, thiserror::Error)]
#[error("unable to convert")]
pub struct ConversionError;

#[derive(Debug)]
pub struct UnsupportedDataType(pub(crate) DataType);

impl fmt::Display for UnsupportedDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:?}", UNSUPPORTED_DATA_TYPE_PREFIX, self.0)
    }
}

impl std::error::Error for UnsupportedDataType {}

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
    #[error("ONNX import failed: {0}")]
    OnnxImport(#[from] rmlk_schema::OnnxImportError),
    #[error("failed to read ONNX file `{path}`: {source}")]
    OnnxRead {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("build failure: {0}")]
    Internal(#[from] anyhow::Error),
}
