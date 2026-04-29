use rmlk_schema::DataType;

#[derive(Debug, thiserror::Error)]
#[error("inference failed: {0}")]
pub struct InferenceError(#[from] pub(crate) anyhow::Error);

// Todo: add src and dst information.
#[derive(Debug, thiserror::Error)]
#[error("unable to convert")]
pub struct ConversionError;

#[derive(Debug, thiserror::Error)]
#[error("unsupported data type: {0:?}")]
pub struct UnsupportedDataType(pub(crate) DataType);

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
