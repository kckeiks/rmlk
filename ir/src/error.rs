#[derive(Debug)]
pub enum Error {
    Invalid,
    UnknownEntry { name: String },
    MissingField { name: String },
    InvalidValue { field: String, value: String },
    Unknown,
    NotSupported,
}
