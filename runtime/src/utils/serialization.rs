use anyhow::bail;
use half::f16;

impl FromBytes for bool {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Vec<Self>> {
        // Each byte is one boolean.
        let vec = bytes.iter().map(|&b| b != 0).collect();
        Ok(vec)
    }
}

pub trait FromF32 {
    fn from_f32(value: f32) -> Self;
}

impl FromF32 for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }
}

impl FromF32 for f16 {
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }
}

pub trait FromBytes: Sized {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Vec<Self>>;
}

impl FromBytes for f16 {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for f16", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for f32 {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for f32", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for f64 {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for f64", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for i32 {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for i32", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for u32 {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for u32", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for i64 {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for i64", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for u64 {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for u64", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}
