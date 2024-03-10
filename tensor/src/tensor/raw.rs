use crate::alloc::TensorAllocator;
use crate::tensor::dtype::DType;
use crate::tensor::op::Op;
use crate::tensor::scalar::Scalar;
use crate::tensor::{Error, Result};
use std::alloc::{Allocator, Layout};
use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::rc::{Rc, Weak};

pub const MAX_DIMS: usize = 4;
pub const MAX_SRC: usize = 10;

pub type RawTensorPtr<M, N> = Rc<RefCell<RawTensor<M>>, N>;
pub type WeakRawTensorPtr<M, N> = Weak<RefCell<RawTensor<M>>, N>;

#[derive(Debug)]
pub(crate) struct RawTensor<A: TensorAllocator> {
    pub dtype: DType,
    pub shape: [usize; MAX_DIMS],
    pub stride: [usize; MAX_DIMS],
    pub op: Op,
    pub src: [Option<RawTensorPtr<A, A::MetadataAlloc>>; MAX_SRC],
    pub alloc: A::ComputeAlloc,
    pub layout: Layout,
    pub data: Option<NonNull<[u8]>>,
}

impl<A: TensorAllocator> RawTensor<A> {
    pub fn new_with_dt(dtype: DType, shape: &[usize], alloc: A::ComputeAlloc) -> Result<Self> {
        if shape.is_empty() || shape.len() > MAX_DIMS {
            return Err(Error::InvalidShape);
        }

        let mut shape_arr = [1usize; MAX_DIMS];
        for (i, element_num) in shape.iter().enumerate() {
            shape_arr[i] = *element_num;
        }

        let mut stride = [0usize; MAX_DIMS];
        stride[0] = dtype.size();
        for i in 1..MAX_DIMS {
            stride[i] += stride[i - 1] * shape_arr[i - 1];
        }

        let src: [Option<RawTensorPtr<A, A::MetadataAlloc>>; MAX_SRC] = Default::default();

        let mut size = dtype.size();
        for i in 0..MAX_DIMS {
            size += (shape_arr[i] - 1) * stride[i];
        }

        let mut layout =
            Layout::from_size_align(size, dtype.alignment()).map_err(|_| Error::AlignmentError)?;

        Ok(Self {
            dtype,
            shape: shape_arr,
            stride,
            op: Op::NoOp,
            src,
            alloc,
            layout,
            data: None,
        })
    }

    /// Allocates memory for the tensor's data.
    ///
    /// Note: data may not be initialized.
    pub unsafe fn alloc<D: Scalar>(&mut self) -> Result<()> {
        if self.data.is_some() {
            return Err(Error::AlreadyInitialized);
        }
        let ptr = self.alloc.allocate(self.layout)?;
        self.data = Some(ptr);
        Ok(())
    }

    pub fn alloc_zeroed<D: Scalar>(&mut self) -> Result<()> {
        if !self.data.is_none() {
            return Err(Error::AlreadyInitialized);
        }
        let ptr = self.alloc.allocate_zeroed(self.layout)?;
        self.data = Some(ptr);
        Ok(())
    }

    unsafe fn data(&self) -> Option<&[MaybeUninit<u8>]> {
        self.data.map(|data| data.as_uninit_slice())
    }

    unsafe fn data_mut(&mut self) -> Option<&mut [MaybeUninit<u8>]> {
        self.data.map(|mut data| data.as_uninit_slice_mut())
    }

    unsafe fn assume_init_data(&self) -> Option<&[u8]> {
        self.data()
            .map(|data| MaybeUninit::slice_assume_init_ref(data))
    }

    unsafe fn assume_init_data_mut(&mut self) -> Option<&mut [u8]> {
        self.data_mut()
            .map(|data| MaybeUninit::slice_assume_init_mut(data))
    }

    fn len(&self) -> usize {
        unsafe { self.data().map(|d| d.len()).unwrap_or(0) }
    }

    /// Safety: assumes tensor has been initialized.
    pub unsafe fn get<D: Scalar>(&self, dtype: DType, coordinates: &[usize]) -> Result<D> {
        assert!(matches!(&self.dtype, dtype), "invalid `dtype` for read");
        assert!(self.data.is_some(), "needs allocation");
        assert!(!coordinates.is_empty(), "empty list of coordinates");
        assert!(
            coordinates.len() <= MAX_DIMS,
            "coordinates exceed dimension limit"
        );

        let mut index = 0;
        for i in 0..coordinates.len() {
            index += coordinates[i] * self.stride[i];
        }

        let data = self
            .assume_init_data()
            .expect("Already verified that it exists");
        if index >= data.len() || data[index..].len() < dtype.size() {
            return Err(Error::OutOfMemRange);
        }

        let bytes = &data[index..index + dtype.size()];
        D::from_bytes(bytes).map_err(|e| Error::ScalarReadFailed)
    }

    /// Safety: assumes tensor has been initialized.
    pub unsafe fn set<D: Scalar>(&mut self, coordinates: &[usize], value: D) -> Result<()> {
        let dtype = D::dtype();
        assert!(matches!(&self.dtype, dtype), "invalid `dtype` for read");
        assert!(self.data.is_some(), "raw tensor is uninitialized");
        assert!(!coordinates.is_empty(), "empty list of coordinates");
        assert!(
            coordinates.len() <= MAX_DIMS,
            "coordinates exceed dimension limit"
        );

        let mut index = 0;
        for i in 0..coordinates.len() {
            index += coordinates[i] * self.stride[i];
        }

        let data = self.data_mut().expect("Already verified that it exists");
        if index >= data.len() || data[index..].len() < dtype.size() {
            return Err(Error::OutOfMemRange);
        }

        let mut section = &mut data[index..index + dtype.size()];
        let value = value.to_bytes();
        MaybeUninit::write_slice(section, value.as_ref());

        Ok(())
    }

    pub fn binary_op(
        a: RawTensorPtr<A, A::MetadataAlloc>,
        b: RawTensorPtr<A, A::MetadataAlloc>,
        op: Op,
    ) -> Result<Self> {
        let mut tensor_c = {
            let tensor_a = a.try_borrow().map_err(|_| Error::AlreadyBorrowed)?;
            let tensor_b = b.try_borrow().map_err(|_| Error::AlreadyBorrowed)?;

            let dtype_a = &tensor_a.dtype;
            let dtype_b = &tensor_b.dtype;
            assert!(matches!(dtype_a, dtype_b));

            if !tensor_a.is_broadcast_compatible(&tensor_b) {
                return Err(Error::InvalidShape);
            }

            Self::new_with_dt(
                tensor_a.dtype,
                tensor_a.shape.as_slice(),
                tensor_a.alloc.clone(),
            )?
        };

        tensor_c.src[0] = Some(a);
        tensor_c.src[1] = Some(b);

        Ok(tensor_c)
    }

    pub fn dup(&self) -> Result<RawTensor<A>> {
        Self::new_with_dt(self.dtype, &self.shape, self.alloc.clone())
    }

    pub fn is_broadcast_compatible(&self, b: &RawTensor<A>) -> bool {
        self.shape[0] % b.shape[0] == 0
            && self.shape[1] % b.shape[1] == 0
            && self.shape[2] % b.shape[2] == 0
            && self.shape[3] % b.shape[3] == 0
    }

    pub unsafe fn light_tensor(&self) -> LightTensor {
        assert!(self.data.is_some());

        LightTensor {
            dtype: self.dtype,
            shape: self.shape,
            stride: self.stride,
            layout: self.layout,
            data: self.data.unwrap(),
        }
    }
}

impl<A: TensorAllocator> Drop for RawTensor<A> {
    fn drop(&mut self) {
        if let Some(ptr) = self.data {
            unsafe {
                self.alloc.deallocate(ptr.cast(), self.layout);
            }
        }
    }
}

pub struct LightTensor {
    pub dtype: DType,
    pub shape: [usize; MAX_DIMS],
    pub stride: [usize; MAX_DIMS],
    pub layout: Layout,
    pub data: NonNull<[u8]>,
}
