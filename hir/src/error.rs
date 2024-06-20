pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Invalid,
    MissingField { name: String },
    InvalidValue { field: String, value: String },
}
