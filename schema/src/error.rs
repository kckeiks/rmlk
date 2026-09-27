#[derive(Debug, PartialEq)]
pub enum Error {
    Invalid,
    UnknownEntry {
        name: String,
    },
    MissingField {
        name: String,
    },
    InvalidValue {
        field: String,
        value: String,
    },
    /// `Tensor::raw_data` was missing when typed element access required it.
    MissingRawData,
    /// Byte length of `raw_data` is not a multiple of the element size, or does
    /// not match the product of `dims`.
    InvalidRawData {
        expected_bytes: usize,
        got_bytes: usize,
    },
    /// Requested element type does not match `Tensor::data_type`.
    DtypeMismatch {
        expected: crate::DataType,
        got: crate::DataType,
    },
    /// Element count does not match the product of `dims`.
    LenMismatch {
        expected: usize,
        got: usize,
    },
    Unknown,
    NotSupported,
}
