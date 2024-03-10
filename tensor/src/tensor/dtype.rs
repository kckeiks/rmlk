use crate::tensor::Error;
use std::mem;

#[derive(Clone, Copy, Debug)]
pub enum DType {
    F16,
    F32,
    I8,
    I16,
    I32,
    U8,
    U16,
    U32,
}

impl DType {
    // Todo: Move this to scalar?
    pub fn size(&self) -> usize {
        match self {
            DType::F16 => 2,
            DType::F32 => 4,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::U8 => 1,
            DType::U16 => 2,
            DType::U32 => 4,
        }
    }

    pub fn alignment(&self) -> usize {
        match self {
            DType::F16 => unimplemented!(),
            DType::F32 => mem::align_of::<f32>(),
            DType::I8 => mem::align_of::<i8>(),
            DType::I16 => mem::align_of::<i16>(),
            DType::I32 => mem::align_of::<i32>(),
            DType::U8 => mem::align_of::<u8>(),
            DType::U16 => mem::align_of::<u16>(),
            DType::U32 => mem::align_of::<u32>(),
        }
    }
}

impl TryFrom<u32> for DType {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let dtype = match value {
            0 => DType::F16,
            1 => DType::F32,
            2 => DType::I8,
            3 => DType::I16,
            4 => DType::I32,
            5 => DType::U8,
            6 => DType::U16,
            7 => DType::U32,
            _ => return Err(Error::UnknownDType),
        };
        Ok(dtype)
    }
}
