use crate::error::Error;
use crate::model::StringStringEntryProto;
use crate::onnx;
use crate::onnx::mod_TensorShapeProto::mod_Dimension::OneOfvalue;
use crate::onnx::{TensorProto, TensorShapeProto};

pub struct Tensor {
    pub dims: Vec<usize>,
    pub data_type: DataType,
    // For very large tensors, we may want to store them in chunks, in which
    // case the following fields will specify the segment that is stored in
    // the current TensorProto.
    pub segment: Option<Segment>,
    // For float and complex64 values
    // Complex64 tensors are encoded as a single array of floats,
    // with the real components appearing in odd numbered positions,
    // and the corresponding imaginary component appearing in the
    // subsequent even numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
    // is encoded as [1.0, 2.0 ,3.0 ,4.0]
    // When this field is present, the data_type field MUST be FLOAT or COMPLEX64.
    // repeated float float_data = 4 [packed = true];
    pub float_data: Vec<f32>,
    // For int32, uint8, int8, uint16, int16, uint4, int4, bool, float8 and float16 values
    // float16 and float8 values must be bit-wise converted to an uint16_t prior
    // to writing to the buffer.
    // uint4 and int4 values must be packed to 4bitx2 prior to writing to the buffer, the first element is stored in
    // the 4 LSB and the second element is stored in the 4 MSB.
    // When this field is present, the data_type field MUST be
    // INT32, INT16, INT8, INT4, UINT16, UINT8, UINT4, BOOL, FLOAT16, BFLOAT16, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ
    pub int32_data: Vec<i32>,
    pub string_data: Vec<Vec<u8>>,
    pub int64_data: Vec<i64>,
    pub name: Option<String>,
    pub doc_string: Option<String>,
    pub raw_data: Option<Vec<u8>>,
    // Todo: We need to point to some place in different file.
    // Data can be stored inside the protobuf file using type-specific fields or raw_data.
    // Alternatively, raw bytes data can be stored in an external file, using the external_data field.
    // external_data stores key-value pairs describing data location. Recognized keys are:
    // - "location" (required) - POSIX filesystem path relative to the directory where the ONNX
    //                           protobuf model was stored
    // - "offset" (optional) - position of byte at which stored data begins. Integer stored as string.
    //                         Offset values SHOULD be multiples 4096 (page size) to enable mmap support.
    // - "length" (optional) - number of bytes containing data. Integer stored as string.
    // - "checksum" (optional) - SHA1 digest of file specified in under 'location' key.
    pub external_data: Vec<StringStringEntryProto>,
    // Location of the data for this tensor. MUST be one of:
    // - DEFAULT - data stored inside the protobuf message. Data is stored in raw_data (if set) otherwise in type-specified field.
    // - EXTERNAL - data stored in an external location as described by external_data field.
    // If value not set, data is stored in raw_data (if set) otherwise in type-specified field.
    pub data_location: Option<DataLocation>,
    pub double_data: Vec<f64>,
    pub uint64_data: Vec<u64>,
    pub metadata_props: Vec<StringStringEntryProto>,
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            dims: vec![],
            data_type: DataType::Undefined,
            segment: None,
            float_data: vec![],
            int32_data: vec![],
            string_data: vec![],
            int64_data: vec![],
            name: None,
            doc_string: None,
            raw_data: None,
            external_data: vec![],
            data_location: Some(DataLocation::Default),
            double_data: vec![],
            uint64_data: vec![],
            metadata_props: vec![],
        }
    }
}

impl TryFrom<TensorProto<'_>> for Tensor {
    type Error = Error;

    fn try_from(value: TensorProto) -> Result<Self, Self::Error> {
        Ok(Self {
            dims: value.dims.into_iter().map(|d| d as usize).collect(),
            data_type: value
                .data_type
                .ok_or(Error::MissingField {
                    name: "Tensor::data_type".to_string(),
                })?
                .try_into()?,
            segment: None,
            float_data: value.float_data.to_vec(),
            int32_data: value.int32_data.to_vec(),
            string_data: value
                .string_data
                .into_iter()
                .map(|data| data.to_vec())
                .collect(),
            int64_data: value.int64_data.to_vec(),
            name: value.name.map(|name| name.to_string()),
            doc_string: value.doc_string.map(|doc| doc.to_string()),
            raw_data: value.raw_data.map(|data| data.to_vec()),
            external_data: value.external_data.into_iter().map(From::from).collect(),
            data_location: value.data_location.map(From::from),
            double_data: value.double_data.to_vec(),
            uint64_data: value.uint64_data,
            metadata_props: value.metadata_props.into_iter().map(From::from).collect(),
        })
    }
}

enum DataLocation {
    Default,
    External,
}

impl From<onnx::mod_TensorProto::DataLocation> for DataLocation {
    fn from(value: onnx::mod_TensorProto::DataLocation) -> Self {
        match value {
            onnx::mod_TensorProto::DataLocation::DEFAULT => Self::Default,
            onnx::mod_TensorProto::DataLocation::EXTERNAL => Self::External,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
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

impl TryFrom<i32> for DataType {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let v = match value {
            0 => Self::Undefined,
            1 => Self::Float,
            2 => Self::Uint8,
            3 => Self::Int8,
            4 => Self::Uint16,
            5 => Self::Int16,
            6 => Self::Int32,
            7 => Self::Int64,
            8 => Self::String,
            9 => Self::Bool,
            10 => Self::Float16,
            11 => Self::Double,
            12 => Self::Uint32,
            13 => Self::Uint64,
            14 => Self::Complex64,
            15 => Self::Complex128,
            16 => Self::Bfloat16,
            17 => Self::Float8E4M3FN,
            18 => Self::Float8E4M3FNUZ,
            19 => Self::Float8E5M2,
            20 => Self::Float8E5M2FNUZ,
            21 => Self::Uint4,
            22 => Self::Int4,
            v => {
                return Err(Error::InvalidValue {
                    field: "DataType".to_string(),
                    value: format!("unknown type {v}"),
                })
            }
        };

        Ok(v)
    }
}

impl From<onnx::mod_TensorProto::DataType> for DataType {
    fn from(value: onnx::mod_TensorProto::DataType) -> Self {
        match value {
            onnx::mod_TensorProto::DataType::UNDEFINED => DataType::Undefined,
            onnx::mod_TensorProto::DataType::FLOAT => DataType::Float,
            onnx::mod_TensorProto::DataType::UINT8 => DataType::Uint8,
            onnx::mod_TensorProto::DataType::INT8 => DataType::Int8,
            onnx::mod_TensorProto::DataType::UINT16 => DataType::Uint16,
            onnx::mod_TensorProto::DataType::INT16 => DataType::Int16,
            onnx::mod_TensorProto::DataType::INT32 => DataType::Int32,
            onnx::mod_TensorProto::DataType::INT64 => DataType::Int64,
            onnx::mod_TensorProto::DataType::STRING => DataType::String,
            onnx::mod_TensorProto::DataType::BOOL => DataType::Bool,
            onnx::mod_TensorProto::DataType::FLOAT16 => DataType::Float16,
            onnx::mod_TensorProto::DataType::DOUBLE => DataType::Double,
            onnx::mod_TensorProto::DataType::UINT32 => DataType::Uint32,
            onnx::mod_TensorProto::DataType::UINT64 => DataType::Uint64,
            onnx::mod_TensorProto::DataType::COMPLEX64 => DataType::Complex64,
            onnx::mod_TensorProto::DataType::COMPLEX128 => DataType::Complex128,
            onnx::mod_TensorProto::DataType::BFLOAT16 => DataType::Bfloat16,
            onnx::mod_TensorProto::DataType::FLOAT8E4M3FN => DataType::Float8E4M3FN,
            onnx::mod_TensorProto::DataType::FLOAT8E4M3FNUZ => DataType::Float8E4M3FNUZ,
            onnx::mod_TensorProto::DataType::FLOAT8E5M2 => DataType::Float8E5M2,
            onnx::mod_TensorProto::DataType::FLOAT8E5M2FNUZ => DataType::Float8E5M2FNUZ,
            onnx::mod_TensorProto::DataType::UINT4 => DataType::Uint4,
            onnx::mod_TensorProto::DataType::INT4 => DataType::Int4,
        }
    }
}

struct Segment {
    pub begin: Option<i64>,
    pub end: Option<i64>,
}

impl From<onnx::mod_TensorProto::Segment> for Segment {
    fn from(value: onnx::mod_TensorProto::Segment) -> Self {
        Self {
            begin: value.begin,
            end: value.end,
        }
    }
}

pub struct SparseTensor {
    values: Option<Tensor>,
    indices: Option<Tensor>,
    dims: Vec<i64>,
}

impl TryFrom<onnx::SparseTensorProto<'_>> for SparseTensor {
    type Error = Error;

    fn try_from(value: onnx::SparseTensorProto) -> Result<Self, Self::Error> {
        Ok(SparseTensor {
            values: value.values.map(|t| t.try_into()).transpose()?,
            indices: value.indices.map(|t| t.try_into()).transpose()?,
            dims: value.dims,
        })
    }
}

pub struct Dimension {
    pub value: Option<DimensionValue>,
    // Standard denotation can optionally be used to denote tensor
    // dimensions with standard semantic descriptions to ensure
    // that operations are applied to the correct axis of a tensor.
    // Refer to https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md#denotation-definition
    // for pre-defined dimension denotations.
    pub denotation: Option<String>,
}

impl TryFrom<onnx::mod_TensorShapeProto::Dimension<'_>> for Dimension {
    type Error = Error;

    fn try_from(value: onnx::mod_TensorShapeProto::Dimension) -> Result<Self, Self::Error> {
        let dim_val = match value.value {
            OneOfvalue::dim_value(value) => Some(DimensionValue::Value(value)),
            OneOfvalue::dim_param(param) => Some(DimensionValue::String(param.to_string())),
            OneOfvalue::None => None,
        };

        Ok(Self {
            value: dim_val,
            denotation: value.denotation.map(|d| d.to_string()),
        })
    }
}

enum DimensionValue {
    Value(i64),
    String(String),
}

pub struct TensorShape {
    pub dim: Vec<Dimension>,
}

impl TryFrom<&Dimension> for usize {
    type Error = Error;

    fn try_from(value: &Dimension) -> Result<Self, Self::Error> {
        match value.value.as_ref().ok_or(Error::MissingField { name: "Dimension::value".to_string() })? {
            DimensionValue::Value(v) => {
                usize::try_from(*v).map_err(|_| Error::InvalidValue { field: "Dimension::value".to_string(), value: v.to_string() })
            }
            DimensionValue::String(v) => {
                v.parse().map_err(|_| Error::InvalidValue { field: "Dimension::value".to_string(), value: v.to_string() })
            }
        }
    }
}

impl TryFrom<TensorShapeProto<'_>> for TensorShape {
    type Error = Error;

    fn try_from(value: TensorShapeProto) -> Result<Self, Self::Error> {
        let mut dim = Vec::new();
        for d in value.dim {
            dim.push(d.try_into()?);
        }

        Ok(Self { dim })
    }
}
