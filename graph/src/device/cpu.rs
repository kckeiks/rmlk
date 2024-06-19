use crate::device::{Device, DeviceError, Tensor};
use crate::node::Node;
use crate::{device, Op};
use rmlk_hir::DataType;
use std::alloc::Global;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct CpuDevice;

impl Device for CpuDevice {
    type Tensor = CpuTensor;
    type Allocator = Global;
    fn allocator(&self) -> Self::Allocator {
        Global
    }

    fn tensor(&self) -> Self::Tensor {
        CpuTensor {
            dtype: DataType::Undefined,
            stride: Vec::new(),
            shape: Vec::new(),
            data: Vec::new(),
        }
    }

    fn tensor_from_hir(&self, input: rmlk_hir::Tensor) -> device::Result<Self::Tensor> {
        let shape = input.dims;
        let mut stride = vec![0; shape.len()];
        stride[shape.len() - 1] = 1;
        for dim in (0..shape.len() - 1).rev() {
            stride[dim] += stride[dim + 1] * shape[dim + 1];
        }

        Ok(CpuTensor {
            dtype: input.data_type.unwrap_or(DataType::Undefined),
            stride,
            shape,
            data: input.float_data,
        })
    }
}

#[derive(Debug)]
pub struct CpuTensor {
    dtype: DataType,
    stride: Vec<usize>,
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor<CpuDevice> for CpuTensor {
    fn compute<'a>(
        &mut self,
        op: Op,
        mut inputs: impl Iterator<Item = Arc<Node<CpuDevice>>>,
    ) -> device::Result<()> {
        let a = inputs.next().ok_or(DeviceError::MissingInput)?;
        let b = inputs.next().ok_or(DeviceError::MissingInput)?;

        debug_assert!(inputs.next().is_none());

        let tensor_a = a.tensor();
        let tensor_b = b.tensor();

        if tensor_a.dtype != tensor_b.dtype {
            return Err(DeviceError::InvalidInputDataType);
        }

        if tensor_a.shape != tensor_b.shape {
            return Err(DeviceError::InvalidShape);
        }

        if tensor_a.stride != tensor_b.stride {
            return Err(DeviceError::InvalidStride);
        }

        match op {
            Op::NoOp => {
                return Err(DeviceError::InvalidOp);
            }
            Op::Add => {
                self.data = Vec::with_capacity(tensor_a.data.len());
                for (i, (a, b)) in tensor_a.data.iter().zip(tensor_b.data.iter()).enumerate() {
                    self.data.insert(i, a + b);
                }
            }
            Op::Mul => {
                self.data = Vec::with_capacity(tensor_a.data.len());
                for (i, (a, b)) in tensor_a.data.iter().zip(tensor_b.data.iter()).enumerate() {
                    self.data.insert(i, a * b);
                }
            }
            Op::MatMul => {
                if tensor_a.shape.len() != 2 {
                    return Err(DeviceError::InvalidShape);
                }
                // i % dim
                // 0 % 2 = 0
                // 1 % 2 = 1
                // 2 % 2 = 0
                // 3 % 2 = 1
                // 0 = 0 + col0/row0 = (0 % 2) * 2 + 0 = (0 % 2) * 2 + 0/2 = (i % dim) * dim + (i/dim)
                // 2 = 1 + col0/row1 = (1 % 2) * 2 + 0 = (1 % 2) * 2 + 1/2 =
                // 1 = 2 + col1/row0 = (2 % 2) * 2 + 1 = (2 % 2) * 2 + 2/2 =
                // 3 = 3 + col1/row1 = (3 % 2) * 2 + 1 = (3 % 2) * 2 + 3/2 =
                /*
                    [[1, 2],   [1, 2, 3, 4]
                     [3, 4]]

                    [[1, 2],  [10, 20, 30, 40]
                     [3, 4]]

                     [
                      [ [1] * {10} + [2] * {30}, [1] * {20} + [2] * {40}],
                      [ [3] * {10} + [4] * {30}, [3] * {20} + [4] * {40}]
                     ]


                    [
                      [ [1] * {1} + [2] * {3} = 7, [1] * {2} + [2] * {4} = 10],
                      [ [3] * {1} + [4] * {3} = 15, [3] * {2} + [4] * {4} = 22]
                     ]

                     [
                      [(1) , [2] ,  (3), ..]
                */
                let m = tensor_a.shape[tensor_a.shape.len() - 2];
                let k = tensor_a.shape[tensor_a.shape.len() - 1];
                let n = tensor_b.shape[tensor_b.shape.len() - 1];

                self.data = vec![0.0; k * n];
                let step_a = tensor_a.stride[tensor_a.stride.len() - 2];
                let step_b = tensor_b.stride[tensor_b.stride.len() - 2];
                for i in 0..n * k {
                    for j in 0..m {
                        self.data[i] += tensor_a.data[j + (i / step_a) * step_a]
                            * tensor_b.data[i % step_b + j * step_b];
                    }
                }
            }
            Op::Sub => {
                self.data = Vec::with_capacity(tensor_a.data.len());
                for (i, (a, b)) in tensor_a.data.iter().zip(tensor_b.data.iter()).enumerate() {
                    self.data.insert(i, a - b);
                }
            }
        }

        // Todo: Move.
        self.dtype = tensor_a.dtype;
        Ok(())
    }
}
