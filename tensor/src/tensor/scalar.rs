use crate::tensor::dtype::DType;
use std::array::TryFromSliceError;
use std::mem;

// Todo: Hide this public trait;
pub trait Scalar: Sized {
    type Bytes: AsRef<[u8]>;

    fn dtype() -> DType;

    fn from_bytes(bytes: &[u8]) -> Result<Self, std::array::TryFromSliceError>;

    fn to_bytes(self) -> Self::Bytes;

    fn alignment(&self) -> usize {
        mem::align_of::<Self>()
    }
}

impl Scalar for u8 {
    type Bytes = [u8; 1];

    fn dtype() -> DType {
        DType::U8
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, TryFromSliceError> {
        Ok(u8::from_ne_bytes(bytes.try_into()?))
    }

    fn to_bytes(self) -> Self::Bytes {
        u8::to_ne_bytes(self)
    }
}

impl Scalar for f32 {
    type Bytes = [u8; 4];

    fn dtype() -> DType {
        DType::F32
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, TryFromSliceError> {
        Ok(f32::from_ne_bytes(bytes.try_into()?))
    }

    fn to_bytes(self) -> Self::Bytes {
        f32::to_ne_bytes(self)
    }
}

// impl Scalar for u16 {
//     fn dtype() -> DType {
//         DType::U16
//     }
// }
//
// impl Scalar for u32 {
//     fn dtype() -> DType {
//         DType::U32
//     }
// }
//
// impl Scalar for i8 {
//     fn dtype() -> DType {
//         DType::I8
//     }
// }
//
// impl Scalar for i16 {
//     fn dtype() -> DType {
//         DType::I16
//     }
// }
//
// impl Scalar for i32 {
//     fn dtype() -> DType {
//         DType::I32
//     }
// }
//
// impl Scalar for f32 {
//     fn dtype() -> DType {
//         DType::F32
//     }
// }
