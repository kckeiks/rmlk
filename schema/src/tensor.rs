use crate::error::Error;
use crate::onnx;
use crate::onnx::dimension_proto::OneOfvalue;
use crate::onnx::tensor_proto::DataLocation;
use crate::onnx::{TensorProto, TensorShapeProto};
use half::f16;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize)]
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
    // pub external_data: Vec<StringStringEntryProto>,
    // Location of the data for this tensor. MUST be one of:
    // - DEFAULT - data stored inside the protobuf message. Data is stored in raw_data (if set) otherwise in type-specified field.
    // - EXTERNAL - data stored in an external location as described by external_data field.
    // If value not set, data is stored in raw_data (if set) otherwise in type-specified field.
    // pub data_location: Option<DataLocation>,
    pub double_data: Vec<f64>,
    pub uint64_data: Vec<u64>,
    // Todo: remove this and use int32_data.
    pub bool_data: Vec<bool>,
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
            double_data: vec![],
            uint64_data: vec![],
            bool_data: vec![],
        }
    }
}

impl Tensor {
    /// Build a tensor from a typed host buffer.
    ///
    /// Data is stored little-endian in [`Self::raw_data`]. Typed fields
    /// (`float_data`, `int64_data`, …) are left empty. The product of `dims`
    /// must equal `data.len()`; an empty `dims` is a scalar and requires
    /// exactly one element.
    pub fn from_vec<T: AsRawBytes>(
        dims: impl Into<Vec<usize>>,
        data: Vec<T>,
    ) -> Result<Self, Error> {
        let dims = dims.into();
        let expected = dims.iter().copied().product::<usize>();
        if data.len() != expected {
            return Err(Error::LenMismatch {
                expected,
                got: data.len(),
            });
        }

        let mut raw = Vec::with_capacity(data.len().saturating_mul(T::BYTE_LEN));
        for value in &data {
            value.write_le(&mut raw);
        }

        Ok(Self {
            dims,
            data_type: T::data_type(),
            raw_data: Some(raw),
            ..Default::default()
        })
    }

    /// Decode [`Self::raw_data`] into a typed host buffer.
    ///
    /// Only reads `raw_data`; typed fields are ignored. The tensor's
    /// `data_type` must match `T`, and the byte length must equal
    /// `product(dims) * T::BYTE_LEN`.
    pub fn to_vec<T: AsRawBytes>(&self) -> Result<Vec<T>, Error> {
        if self.data_type != T::data_type() {
            return Err(Error::DtypeMismatch {
                expected: T::data_type(),
                got: self.data_type,
            });
        }

        let raw = self.raw_data.as_deref().ok_or(Error::MissingRawData)?;
        let expected_elems = self.dims.iter().copied().product::<usize>();
        let expected_bytes = expected_elems.saturating_mul(T::BYTE_LEN);
        if raw.len() != expected_bytes {
            return Err(Error::InvalidRawData {
                expected_bytes,
                got_bytes: raw.len(),
            });
        }

        let mut out = Vec::with_capacity(expected_elems);
        for chunk in raw.chunks_exact(T::BYTE_LEN) {
            out.push(T::read_le(chunk)?);
        }
        Ok(out)
    }
}

pub fn tensor_from_onnx_tensor(
    value: TensorProto,
    base_url: Option<PathBuf>,
) -> Result<Tensor, Error> {
    let external_data = match value.data_location {
        None | Some(DataLocation::DEFAULT) => None,
        Some(DataLocation::EXTERNAL) => {
            let mut location = None;
            let mut offset = None;
            let mut length = None;
            // Todo: Handle checksum.
            let mut _checksum = None;
            for entry in value.external_data {
                let key = entry.key.as_ref().ok_or(Error::MissingField {
                    name: "Tensor::external_data::key".to_string(),
                })?;
                match key.as_ref() {
                    "location" if location.is_none() => {
                        location = Some(entry.value.ok_or(Error::InvalidValue {
                            field: "location".to_string(),
                            value: "None".to_string(),
                        })?);
                    }
                    "offset" => {
                        offset = Some(entry.value.ok_or(Error::InvalidValue {
                            field: "offset".to_string(),
                            value: "None".to_string(),
                        })?);
                    }
                    "length" => {
                        length = Some(entry.value.ok_or(Error::InvalidValue {
                            field: "length".to_string(),
                            value: "None".to_string(),
                        })?);
                    }
                    "checksum" => {
                        _checksum = Some(entry.value.ok_or(Error::InvalidValue {
                            field: "checksum".to_string(),
                            value: "None".to_string(),
                        })?);
                    }
                    _ => todo!(),
                }
            }

            let location = location.ok_or(Error::InvalidValue {
                field: "location".to_string(),
                value: "None".to_string(),
            })?;

            let mut file = match base_url {
                None => File::open(location.as_ref())
                    .map_err(|_| Error::Unknown)
                    .unwrap(),
                Some(mut base) => {
                    base.push(location.as_ref());

                    File::open(format!("{}", base.as_os_str().to_string_lossy(),))
                        .map_err(|_| Error::Unknown)
                        .unwrap()
                }
            };

            match (offset, length) {
                (Some(offset_str), Some(length_str)) => {
                    let offset = offset_str.parse().map_err(|_| Error::Unknown).unwrap();
                    let length = length_str.parse().map_err(|_| Error::Unknown).unwrap();
                    let mut buf = vec![0; length];
                    file.read_at(&mut buf, offset)
                        .map_err(|_| Error::Unknown)
                        .unwrap();
                    Some(buf)
                }
                (None, None) => {
                    let mut buf = Vec::new();
                    file.read_to_end(&mut buf)
                        .map_err(|_| Error::Unknown)
                        .unwrap();
                    Some(buf)
                }
                _ => return Err(Error::Invalid),
            }
        }
    };

    let data_type: DataType = value
        .data_type
        .ok_or(Error::MissingField {
            name: "Tensor::data_type".to_string(),
        })?
        .try_into()?;

    if !data_type.is_supported() {
        return Err(Error::NotSupported);
    }

    assert!(value.metadata_props.is_empty(), "this is not supported");

    let mut res = Tensor {
        dims: value.dims.into_iter().map(|d| d as usize).collect(),
        data_type,
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
        double_data: value.double_data.to_vec(),
        uint64_data: value.uint64_data,
        bool_data: Vec::new(),
    };

    if let Some(data) = external_data {
        match res.data_type {
            DataType::Undefined => {
                return Err(Error::Invalid);
            }
            DataType::Int8 => {
                res.int32_data = u8_to_i32_vec(data.as_slice())?;
            }
            DataType::Float => {
                res.float_data = u8_to_f32_vec(data.as_slice())?;
            }
            DataType::Double => {
                res.double_data = u8_to_f64_vec(data.as_slice())?;
            }
            DataType::Uint8 => {
                res.int32_data = u8_to_i32_vec(data.as_slice())?;
            }
            DataType::Uint16 => {
                res.int32_data = u8_to_i32_vec(data.as_slice())?;
            }
            DataType::Int16 => {
                res.int32_data = u8_to_i32_vec(data.as_slice())?;
            }
            DataType::Int32 => {
                res.int32_data = u8_to_i32_vec(data.as_slice())?;
            }
            DataType::Uint32 => {
                res.int32_data = u8_to_i32_vec(data.as_slice())?;
            }
            DataType::Int64 => {
                res.int64_data = u8_to_i64_vec(data.as_slice())?;
            }
            DataType::Uint64 => {
                res.uint64_data = u8_to_u64_vec(data.as_slice())?;
            }
            DataType::String => {}
            DataType::Bool => {
                println!("loading bool");
                res.int32_data = u8_to_i32_vec(data.as_slice())?;
            }
            DataType::Float16 => {
                return Err(Error::NotSupported);
            }
            DataType::Bfloat16 => {
                return Err(Error::NotSupported);
            }
            DataType::Complex64 => {
                return Err(Error::NotSupported);
            }
            DataType::Complex128 => {
                return Err(Error::NotSupported);
            }
            DataType::Float8E4M3FN => {
                return Err(Error::NotSupported);
            }
            DataType::Float8E4M3FNUZ => {
                return Err(Error::NotSupported);
            }
            DataType::Float8E5M2 => {
                return Err(Error::NotSupported);
            }
            DataType::Float8E5M2FNUZ => {
                return Err(Error::NotSupported);
            }
            DataType::Uint4 => {
                res.int32_data = u8_to_i32_vec(data.as_slice())?;
            }
            DataType::Int4 => {
                res.int32_data = u8_to_i32_vec(data.as_slice())?;
            }
            DataType::USize => {
                unreachable!("this is not supported in onnx");
            }
        }
    }

    Ok(res)
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum DataType {
    // Todo: Fix.
    // changing the order of these three might
    // mess up some tests in the tensor crate.
    Float16,
    Float,
    Double,
    Int32,
    Int64,
    Undefined,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Uint64,
    String,
    Bool,
    Bfloat16,
    Complex64,
    Complex128,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Float8E5M2,
    Float8E5M2FNUZ,
    Uint4,
    Int4,
    USize,
}

impl DataType {
    pub fn is_supported(&self) -> bool {
        match self {
            DataType::Undefined => false,
            DataType::Int8 => true,
            DataType::Float => true,
            DataType::Double => true,
            DataType::Uint8 => true,
            DataType::Uint16 => true,
            DataType::Int16 => true,
            DataType::Int32 => true,
            DataType::Uint32 => true,
            DataType::Int64 => true,
            DataType::Uint64 => true,
            DataType::String => true,
            DataType::Bool => true,
            DataType::Float16 => false,
            DataType::Bfloat16 => false,
            DataType::Complex64 => false,
            DataType::Complex128 => false,
            DataType::Float8E4M3FN => false,
            DataType::Float8E4M3FNUZ => false,
            DataType::Float8E5M2 => false,
            DataType::Float8E5M2FNUZ => false,
            DataType::Uint4 => true,
            DataType::Int4 => true,
            DataType::USize => false,
        }
    }
}

impl From<DataType> for i32 {
    fn from(dt: DataType) -> Self {
        match dt {
            DataType::Undefined => 0,
            DataType::Float => 1,
            DataType::Uint8 => 2,
            DataType::Int8 => 3,
            DataType::Uint16 => 4,
            DataType::Int16 => 5,
            DataType::Int32 => 6,
            DataType::Int64 => 7,
            DataType::String => 8,
            DataType::Bool => 9,
            DataType::Float16 => 10,
            DataType::Double => 11,
            DataType::Uint32 => 12,
            DataType::Uint64 => 13,
            DataType::Complex64 => 14,
            DataType::Complex128 => 15,
            DataType::Bfloat16 => 16,
            DataType::Float8E4M3FN => 17,
            DataType::Float8E4M3FNUZ => 18,
            DataType::Float8E5M2 => 19,
            DataType::Float8E5M2FNUZ => 20,
            DataType::Uint4 => 21,
            DataType::Int4 => 22,
            DataType::USize => -1,
        }
    }
}

impl TryFrom<i32> for DataType {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let v = match value {
            -1 => Self::USize,
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

impl From<onnx::tensor_proto::DataType> for DataType {
    fn from(value: onnx::tensor_proto::DataType) -> Self {
        match value {
            onnx::tensor_proto::DataType::UNDEFINED => DataType::Undefined,
            onnx::tensor_proto::DataType::FLOAT => DataType::Float,
            onnx::tensor_proto::DataType::UINT8 => DataType::Uint8,
            onnx::tensor_proto::DataType::INT8 => DataType::Int8,
            onnx::tensor_proto::DataType::UINT16 => DataType::Uint16,
            onnx::tensor_proto::DataType::INT16 => DataType::Int16,
            onnx::tensor_proto::DataType::INT32 => DataType::Int32,
            onnx::tensor_proto::DataType::INT64 => DataType::Int64,
            onnx::tensor_proto::DataType::STRING => DataType::String,
            onnx::tensor_proto::DataType::BOOL => DataType::Bool,
            onnx::tensor_proto::DataType::FLOAT16 => DataType::Float16,
            onnx::tensor_proto::DataType::DOUBLE => DataType::Double,
            onnx::tensor_proto::DataType::UINT32 => DataType::Uint32,
            onnx::tensor_proto::DataType::UINT64 => DataType::Uint64,
            onnx::tensor_proto::DataType::COMPLEX64 => DataType::Complex64,
            onnx::tensor_proto::DataType::COMPLEX128 => DataType::Complex128,
            onnx::tensor_proto::DataType::BFLOAT16 => DataType::Bfloat16,
            onnx::tensor_proto::DataType::FLOAT8E4M3FN => DataType::Float8E4M3FN,
            onnx::tensor_proto::DataType::FLOAT8E4M3FNUZ => DataType::Float8E4M3FNUZ,
            onnx::tensor_proto::DataType::FLOAT8E5M2 => DataType::Float8E5M2,
            onnx::tensor_proto::DataType::FLOAT8E5M2FNUZ => DataType::Float8E5M2FNUZ,
            onnx::tensor_proto::DataType::UINT4 => DataType::Uint4,
            onnx::tensor_proto::DataType::INT4 => DataType::Int4,
        }
    }
}

pub trait DataTypeMap {
    fn data_type() -> DataType;
}

/// Types that can be stored in [`Tensor::raw_data`] as little-endian bytes.
///
/// Bound used by [`Tensor::from_vec`] and [`Tensor::to_vec`]. `usize` implements
/// [`DataTypeMap`] but not this trait: its width is not portable.
pub trait AsRawBytes: DataTypeMap + Sized {
    const BYTE_LEN: usize;

    fn write_le(&self, buf: &mut Vec<u8>);

    fn read_le(bytes: &[u8]) -> Result<Self, Error>;
}

macro_rules! impl_as_raw_bytes_int {
    ($ty:ty) => {
        impl AsRawBytes for $ty {
            const BYTE_LEN: usize = std::mem::size_of::<$ty>();

            fn write_le(&self, buf: &mut Vec<u8>) {
                buf.extend_from_slice(&self.to_le_bytes());
            }

            fn read_le(bytes: &[u8]) -> Result<Self, Error> {
                let arr: [u8; Self::BYTE_LEN] =
                    bytes.try_into().map_err(|_| Error::InvalidRawData {
                        expected_bytes: Self::BYTE_LEN,
                        got_bytes: bytes.len(),
                    })?;
                Ok(Self::from_le_bytes(arr))
            }
        }
    };
}

impl DataTypeMap for u8 {
    fn data_type() -> DataType {
        DataType::Uint8
    }
}

impl_as_raw_bytes_int!(u8);

impl DataTypeMap for u16 {
    fn data_type() -> DataType {
        DataType::Uint16
    }
}

impl_as_raw_bytes_int!(u16);

impl DataTypeMap for f16 {
    fn data_type() -> DataType {
        DataType::Float16
    }
}

impl AsRawBytes for f16 {
    const BYTE_LEN: usize = 2;

    fn write_le(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le(bytes: &[u8]) -> Result<Self, Error> {
        let arr: [u8; 2] = bytes.try_into().map_err(|_| Error::InvalidRawData {
            expected_bytes: 2,
            got_bytes: bytes.len(),
        })?;
        Ok(Self::from_le_bytes(arr))
    }
}

impl DataTypeMap for f32 {
    fn data_type() -> DataType {
        DataType::Float
    }
}

impl_as_raw_bytes_int!(f32);

impl DataTypeMap for f64 {
    fn data_type() -> DataType {
        DataType::Double
    }
}

impl_as_raw_bytes_int!(f64);

impl DataTypeMap for i32 {
    fn data_type() -> DataType {
        DataType::Int32
    }
}

impl_as_raw_bytes_int!(i32);

impl DataTypeMap for u32 {
    fn data_type() -> DataType {
        DataType::Uint32
    }
}

impl_as_raw_bytes_int!(u32);

impl DataTypeMap for i64 {
    fn data_type() -> DataType {
        DataType::Int64
    }
}

impl_as_raw_bytes_int!(i64);

impl DataTypeMap for u64 {
    fn data_type() -> DataType {
        DataType::Uint64
    }
}

impl_as_raw_bytes_int!(u64);

impl DataTypeMap for bool {
    fn data_type() -> DataType {
        DataType::Bool
    }
}

impl AsRawBytes for bool {
    const BYTE_LEN: usize = 1;

    fn write_le(&self, buf: &mut Vec<u8>) {
        buf.push(u8::from(*self));
    }

    fn read_le(bytes: &[u8]) -> Result<Self, Error> {
        let b = *bytes.first().ok_or(Error::InvalidRawData {
            expected_bytes: 1,
            got_bytes: 0,
        })?;
        Ok(b != 0)
    }
}

impl DataTypeMap for usize {
    fn data_type() -> DataType {
        DataType::USize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn from_vec_round_trips_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_vec([2, 2], data.clone()).unwrap();
        assert_eq!(t.dims, vec![2, 2]);
        assert_eq!(t.data_type, DataType::Float);
        assert!(t.float_data.is_empty());
        assert_eq!(
            t.raw_data.as_ref().unwrap().len(),
            4 * std::mem::size_of::<f32>()
        );
        assert_eq!(t.to_vec::<f32>().unwrap(), data);
    }

    #[test]
    fn from_vec_round_trips_i64_scalar() {
        let t = Tensor::from_vec([], vec![42i64]).unwrap();
        assert!(t.dims.is_empty());
        assert_eq!(t.to_vec::<i64>().unwrap(), vec![42]);
    }

    #[test]
    fn from_vec_round_trips_bool_and_f16() {
        let t = Tensor::from_vec([3], vec![true, false, true]).unwrap();
        assert_eq!(t.raw_data.as_ref().unwrap(), &[1, 0, 1]);
        assert_eq!(t.to_vec::<bool>().unwrap(), vec![true, false, true]);

        let halfs = vec![f16::from_f32(1.5), f16::from_f32(-0.5)];
        let t = Tensor::from_vec([2], halfs.clone()).unwrap();
        assert_eq!(t.data_type, DataType::Float16);
        assert_eq!(t.to_vec::<f16>().unwrap(), halfs);
    }

    #[test]
    fn from_vec_rejects_len_mismatch() {
        let err = Tensor::from_vec::<f32>([2, 2], vec![1.0, 2.0]).unwrap_err();
        assert_eq!(
            err,
            Error::LenMismatch {
                expected: 4,
                got: 2
            }
        );
    }

    #[test]
    fn to_vec_rejects_dtype_mismatch() {
        let t = Tensor::from_vec([2], vec![1i32, 2]).unwrap();
        let err = t.to_vec::<f32>().unwrap_err();
        assert_eq!(
            err,
            Error::DtypeMismatch {
                expected: DataType::Float,
                got: DataType::Int32,
            }
        );
    }

    #[test]
    fn to_vec_rejects_missing_raw_data() {
        let t = Tensor {
            dims: vec![1],
            data_type: DataType::Float,
            float_data: vec![1.0],
            ..Default::default()
        };
        assert_eq!(t.to_vec::<f32>().unwrap_err(), Error::MissingRawData);
    }

    #[test]
    fn little_endian_layout_is_stable() {
        let t = Tensor::from_vec([1], vec![0x0102_0304u32]).unwrap();
        assert_eq!(t.raw_data.as_ref().unwrap(), &[0x04, 0x03, 0x02, 0x01]);
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Segment {
    pub begin: Option<i64>,
    pub end: Option<i64>,
}

impl From<onnx::tensor_proto::Segment> for Segment {
    fn from(value: onnx::tensor_proto::Segment) -> Self {
        Self {
            begin: value.begin,
            end: value.end,
        }
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct SparseTensor {
    values: Option<Tensor>,
    indices: Option<Tensor>,
    dims: Vec<i64>,
}

impl TryFrom<onnx::SparseTensorProto<'_>> for SparseTensor {
    type Error = Error;

    fn try_from(value: onnx::SparseTensorProto) -> Result<Self, Self::Error> {
        Ok(SparseTensor {
            values: value
                .values
                .map(|t| tensor_from_onnx_tensor(t, None))
                .transpose()?,
            indices: value
                .indices
                .map(|t| tensor_from_onnx_tensor(t, None))
                .transpose()?,
            dims: value.dims,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Dimension {
    pub value: Option<DimensionValue>,
    // Standard denotation can optionally be used to denote tensor
    // dimensions with standard semantic descriptions to ensure
    // that operations are applied to the correct axis of a tensor.
    // Refer to https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md#denotation-definition
    // for pre-defined dimension denotations.
    pub denotation: Option<String>,
}

impl TryFrom<onnx::tensor_shape_proto::Dimension<'_>> for Dimension {
    type Error = Error;

    fn try_from(value: onnx::tensor_shape_proto::Dimension) -> Result<Self, Self::Error> {
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

#[derive(Debug, Deserialize, Serialize)]
pub enum DimensionValue {
    Value(i64),
    String(String),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TensorShape {
    pub dim: Vec<Dimension>,
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

fn u8_to_i32_vec(v: &[u8]) -> Result<Vec<i32>, Error> {
    let mut res = Vec::with_capacity(v.len() / 4);
    for chunk in v.chunks_exact(4) {
        res.push(i32::from_le_bytes(
            chunk.try_into().map_err(|_| Error::Unknown).unwrap(),
        ));
    }

    Ok(res)
}

fn u8_to_i64_vec(v: &[u8]) -> Result<Vec<i64>, Error> {
    let mut res = Vec::with_capacity(v.len() / 8);
    for chunk in v.chunks_exact(8) {
        res.push(i64::from_le_bytes(
            chunk.try_into().map_err(|_| Error::Unknown).unwrap(),
        ));
    }

    Ok(res)
}

fn u8_to_u64_vec(v: &[u8]) -> Result<Vec<u64>, Error> {
    let mut res = Vec::with_capacity(v.len() / 8);
    for chunk in v.chunks_exact(8) {
        res.push(u64::from_le_bytes(
            chunk.try_into().map_err(|_| Error::Unknown).unwrap(),
        ));
    }

    Ok(res)
}

fn u8_to_f32_vec(v: &[u8]) -> Result<Vec<f32>, Error> {
    let mut res = Vec::with_capacity(v.len() / 4);
    for chunk in v.chunks_exact(4) {
        res.push(f32::from_le_bytes(
            chunk.try_into().map_err(|_| Error::Unknown).unwrap(),
        ));
    }

    Ok(res)
}

fn u8_to_f64_vec(v: &[u8]) -> Result<Vec<f64>, Error> {
    let mut res = Vec::with_capacity(v.len() / 8);
    for chunk in v.chunks_exact(8) {
        res.push(f64::from_le_bytes(
            chunk.try_into().map_err(|_| Error::Unknown).unwrap(),
        ));
    }

    Ok(res)
}
