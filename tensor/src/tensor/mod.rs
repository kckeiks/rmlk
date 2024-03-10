pub mod dtype;
pub mod op;
pub mod raw;
mod scalar;

use crate::alloc::TensorAllocator;
use crate::tensor::op::Op;
use crate::tensor::raw::{RawTensor, RawTensorPtr, WeakRawTensorPtr};
use scalar::Scalar;
use std::alloc::{AllocError, Allocator, Global};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
/*
   shape = [3, 2, 2, [1]]
   stride = [4, 12, 24, [48]]

   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

 [
    [
     [0, 1, 2],
     [3, 4, 5]
    ],
    [
     [6, 7, 8],
     [9, 10, 11]
    ],
 ]


   elemen_n = [2, 2, 3, [1]]
   stride = [4, 8, 16, 48]

   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

   [
    [[0, 1],
     [2, 3]],
    [[4, 5],
     [6, 7]],
    [[8, 9],
     [10, 11]],
    ]

    Coordinates (1, 1, 0, 0)

    index = 4 * 1 + 8 * 1 + 16 * 0 + 48 * 0 = 12


    Coordinates (0, 0, 2)

    index = 4 * 0 + 8 * 0 + 16 * 2 = 32

   elemen_n = [12, 1, 1, 1]
   stride = [4, 4*12=48, 48, 48]

    coordinates = (5, 0, 0, 0)

    index = 4 * 5 + 48 * 0 + 48 * 0 + 48 * 0 = 20

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

*/

pub type Result<T> = std::result::Result<T, Error>;

/// Public tensor API.
pub struct Tensor<D, T: TensorAllocator> {
    inner: RawTensorPtr<T, T::MetadataAlloc>,
    init_done: bool,
    _phantom: PhantomData<D>,
}

impl<D, T> Tensor<D, T>
where
    D: Scalar,
    T: TensorAllocator,
{
    fn from_ptr(ptr: RawTensorPtr<T, T::MetadataAlloc>) -> Tensor<D, T> {
        Self {
            inner: ptr,
            init_done: false,
            _phantom: PhantomData,
        }
    }

    fn with_data_alloc(shape: &[usize], talloc: T) -> Result<Self> {
        let ptr = RawTensor::new_with_dt(D::dtype(), shape, talloc.compute_alloc())
            .map(|raw| Rc::new_in(RefCell::new(raw), talloc.metadata_alloc()))?;
        Ok(Self::from_ptr(ptr))
    }

    pub fn set(&mut self, coordinates: &[usize], value: D) -> Result<()> {
        let mut tensor = self.inner.borrow_mut();
        unsafe {
            if !self.init_done {
                tensor.alloc::<D>();
                self.init_done = true;
            }
            tensor.set::<D>(coordinates, value)
        }
    }

    pub fn get(&self, coordinates: &[usize]) -> Result<D> {
        let mut tensor = self.inner.borrow();
        if self.init_done {
            unsafe { tensor.get(D::dtype(), coordinates) }
        } else {
            Err(Error::Uninitialized)
        }
    }

    pub fn add(&mut self, b: Tensor<D, T>) -> Result<Tensor<D, T>> {
        self.binary_op(b, Op::Add)
    }

    fn binary_op(&mut self, b: Tensor<D, T>, op: Op) -> Result<Tensor<D, T>> {
        let alloc = Rc::allocator(&self.inner).clone();
        let a = self.inner.clone();
        let b = b.inner.clone();

        let raw_tensor = RawTensor::binary_op(a, b, op)?;

        let ptr = Rc::new_in(RefCell::new(raw_tensor), alloc);
        Ok(Self::from_ptr(ptr))
    }
}

pub struct Builder<A: TensorAllocator> {
    alloc: A,
}

impl Default for Builder<Global> {
    fn default() -> Self {
        Self { alloc: Global }
    }
}

impl<A: TensorAllocator> Builder<A> {
    pub fn new_with_alloc(alloc: A) -> Self {
        Self { alloc }
    }

    pub fn new_tensor<S: Scalar>(&self, shape: &[usize]) -> Result<Tensor<S, A>> {
        Tensor::with_data_alloc(shape, self.alloc.clone())
    }
}

#[derive(Debug)]
pub enum Error {
    AllocationFailed,
    AlreadyInitialized,
    DeallocatedTensor,
    OutOfMemRange,
    InvalidShape,
    InvalidDtypeForOp,
    OutOfMemory,
    AlignmentError,
    UnknownDType,
    Uninitialized,
    ScalarReadFailed,
    AlreadyBorrowed,
}

impl From<AllocError> for Error {
    fn from(value: AllocError) -> Self {
        Self::AllocationFailed
    }
}

#[cfg(test)]
mod test {
    use crate::tensor::{Builder, Error};
    #[test]
    fn test_tensor_create_1d() {
        let mut builder = Builder::default();
        let shape = [10];
        let mut tensor = builder.new_tensor::<f32>(&shape).unwrap();
        for i in 0..shape[0] {
            tensor.set(&[i], i as f32).unwrap();
        }

        let mut result = Vec::new();
        for i in 0..shape[0] {
            let val = tensor.get(&[i]).unwrap();
            result.push(val);
        }
        assert_eq!(
            result,
            vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );

        let result = tensor.set(&[10], 69.0);
        assert!(matches!(result, Err(Error::OutOfMemRange)));
        let result = tensor.get(&[10]);
        assert!(matches!(result, Err(Error::OutOfMemRange)));
    }
}
