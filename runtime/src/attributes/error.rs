use rmlk_schema::{DataType, Op};

pub type Result<T> = std::result::Result<T, AttributeError>;

#[derive(Debug, thiserror::Error)]
pub enum AttributeError {
    #[error(
        "invalid data type `{actual:?}` expected `{expected:?}` for attribute {name} op `{op:?}`"
    )]
    InvalidDataType {
        name: &'static str,
        op: Op,
        expected: DataType,
        actual: DataType,
    },
    #[error("unknown data type for attribute {name} op `{op:?}`")]
    UnknownDataType { name: &'static str, op: Op },
    #[error("missing attributes")]
    MissingAttributes,
    #[error("missing attribute: {name}")]
    MissingAttribute { name: String },
}
