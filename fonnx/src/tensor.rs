use crate::model::StringStringEntryProto;

pub struct Tensor {
    dims: Vec<i64>,
    data_type: Option<DataType>,
    segment: Segment,
    // For float and complex64 values
    // Complex64 tensors are encoded as a single array of floats,
    // with the real components appearing in odd numbered positions,
    // and the corresponding imaginary component appearing in the
    // subsequent even numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
    // is encoded as [1.0, 2.0 ,3.0 ,4.0]
    // When this field is present, the data_type field MUST be FLOAT or COMPLEX64.
    // repeated float float_data = 4 [packed = true];
    float_data: Vec<f32>,
    // For int32, uint8, int8, uint16, int16, uint4, int4, bool, float8 and float16 values
    // float16 and float8 values must be bit-wise converted to an uint16_t prior
    // to writing to the buffer.
    // uint4 and int4 values must be packed to 4bitx2 prior to writing to the buffer, the first element is stored in
    // the 4 LSB and the second element is stored in the 4 MSB.
    // When this field is present, the data_type field MUST be
    // INT32, INT16, INT8, INT4, UINT16, UINT8, UINT4, BOOL, FLOAT16, BFLOAT16, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ
    int32_data: Vec<i32>,
    string_data: Vec<u8>,
    int64_data: Vec<i64>,
    name: Option<String>,
    doc_string: Option<String>,
    raw_data: Option<Vec<u8>>,
    external_data: Vec<StringStringEntryProto>,
    data_location: DataLocation,
    double_data: Vec<f64>,
    uint64_data: Vec<u64>,
    metadata_props: Vec<StringStringEntryProto>,
}

enum DataLocation {
    Default,
    External,
}

pub enum DataType {
    Undefined,
    Float,
    Double,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    String,
    Bool,
    Float16,
    Bfloat16,
    Complex64,
    Complex128,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Float8E5M2,
    Float8E5M2FNUZ,
    Uint4,
    Int4,
}

enum Segment {
    Begin,
    End,
}
pub struct SparseTensor {
    values: Option<Tensor>,
    indices: Option<Tensor>,
    dims: Vec<i64>,
}

struct Dimension {
    value: DimensionValue,
    denotation: Option<String>,
}

enum DimensionValue {
    Value(i64),
    String(String),
}

pub struct TensorShape {
    dim: Vec<Dimension>,
}
