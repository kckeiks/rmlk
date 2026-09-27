use crate::core::error::UnsupportedDataType;
use crate::core::Context;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use crate::providers::TENSOR_3D_RANK;
use crate::utils::FromBytes;
use crate::{attributes, utils};
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::trilu;
use rmlk_cuda::kernels::trilu::TriluKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub struct TriluBackend {
    stream: Arc<CudaStream>,
}

impl TriluBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => TriluKernel::FwdF16,
            DataType::Float => TriluKernel::FwdF32,
            DataType::Double => TriluKernel::FwdF64,
            DataType::Int32 => TriluKernel::FwdI32,
            DataType::Uint32 => TriluKernel::FwdU32,
            DataType::Int64 => TriluKernel::FwdI64,
            DataType::Uint64 => TriluKernel::FwdU64,
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        trilu::load_kernel(self.stream.context(), kernel_name).map_err(Into::into)
    }

    fn get_k(&self, ctx: &Context<Cuda>) -> Result<i64> {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        match ctx.get_input(1) {
            Ok(k_tensor) => {
                debug!(
                    "[k][scalar={:?}][dtype={:?}]",
                    k_tensor.is_scalar(),
                    k_tensor.dtype(),
                );

                if !k_tensor.is_scalar() {
                    return Err(TriluError::InvalidKTensor.into());
                }

                let k = scratch_alloc.allocate(1)?;
                k_tensor.payload_to_host(k)?;

                Ok(k[0])
            }
            Err(_) => Ok(0),
        }
    }

    fn compute_trilu<T>(&mut self, ctx: &Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + FromBytes,
    {
        let func = self.load_cuda_function(T::data_type())?;

        let upper = match ctx.get_attributes() {
            None => true,
            Some(attr) => attributes::trilu::get_upper(&attr),
        };

        let input_tensor = ctx.get_input(0)?;

        debug!(
            "[input][dtype={:?}][shape={:?}][stride=[{:?}]",
            input_tensor.dtype(),
            input_tensor.shape(),
            input_tensor.stride()
        );

        let output_tensor = ctx.get_output(0)?;
        output_tensor.copy_shape(input_tensor.shape_handle());
        output_tensor.init_payload::<T>()?;

        debug!(
            "[output][dtype={:?}][shape={:?}][stride=[{:?}]",
            output_tensor.dtype(),
            output_tensor.shape(),
            output_tensor.stride()
        );

        let k = self.get_k(ctx)?;

        let (batch_shape, batch_stride) = utils::create_3d_shape_and_stride(&input_tensor.shape());

        debug!(
            "[input][batch][shape={:?}][stride=[{:?}]",
            batch_shape, batch_stride
        );

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let info_on_host = scratch_alloc.allocate(2 * TENSOR_3D_RANK)?;
        info_on_host[..TENSOR_3D_RANK].copy_from_slice(&batch_shape);
        info_on_host[TENSOR_3D_RANK..].copy_from_slice(&batch_stride);

        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let info = cuda_bump.alloc_from_slice_with_fallback(info_on_host)?;
        let info_data = info.data::<usize>();

        let input_ptr = input_tensor.payload();
        let input_data = input_ptr.data::<T>();

        {
            let mut output_ptr = output_tensor.payload_mut();
            let mut output_data = output_ptr.data_mut::<T>();

            unsafe {
                trilu::compute(
                    self.stream.clone(),
                    func,
                    upper,
                    k,
                    TENSOR_3D_RANK,
                    &info_data,
                    input_data.as_ref(),
                    output_data.as_mut(),
                )?;
            }
        }

        #[cfg(feature = "dump")]
        debug::write_results_trilu::<T>("debugging/trilu", self.stream.clone(), ctx).unwrap();

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_trilu::<f16>(ctx),
            DataType::Float => self.compute_trilu::<f32>(ctx),
            DataType::Double => self.compute_trilu::<f64>(ctx),
            DataType::Int32 => self.compute_trilu::<i32>(ctx),
            DataType::Int64 => self.compute_trilu::<i64>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

#[derive(Debug)]
pub enum TriluError {
    InvalidKTensor,
}

impl Display for TriluError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for TriluError {}

#[cfg(test)]
mod tests {
    use crate::testing::OpTest;
    use rmlk_schema::{AttributeType, Op};

    #[test]
    fn k_zero_lower() {
        let out = OpTest::new(Op::Trilu)
            .input(
                [2, 3, 3],
                vec![
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 17.0, 18.0,
                ],
            )
            .input([], vec![0i64])
            .attr("upper", AttributeType::Int(0))
            .output([2, 3, 3])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![
                1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, 13.0, 14.0, 0.0, 16.0,
                17.0, 18.0,
            ]
        );
    }

    #[test]
    fn no_k_lower() {
        let out = OpTest::new(Op::Trilu)
            .input(
                [2, 3, 3],
                vec![
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 17.0, 18.0,
                ],
            )
            .attr("upper", AttributeType::Int(0))
            .output([2, 3, 3])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![
                1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, 13.0, 14.0, 0.0, 16.0,
                17.0, 18.0,
            ]
        );
    }

    #[test]
    fn one_k_upper() {
        let out = OpTest::new(Op::Trilu)
            .input(
                [1, 3, 4],
                vec![
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            )
            .input([], vec![1i64])
            .attr("upper", AttributeType::Int(1))
            .output([1, 3, 4])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 12.0]
        );
    }

    #[test]
    fn no_attribute() {
        let out = OpTest::new(Op::Trilu)
            .input(
                [1, 3, 4],
                vec![
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            )
            .input([], vec![1i64])
            .output([1, 3, 4])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 12.0]
        );
    }

    #[test]
    fn large_upper_k1() {
        let out = OpTest::new(Op::Trilu)
            .input([3, 3], vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .input([], vec![1i64])
            .attr("upper", AttributeType::Int(1))
            .output([3, 3])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![0.0, 1.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn rejects_bool() {
        let err = OpTest::new(Op::Trilu)
            .input([2, 2], vec![true, false, true, false])
            .output([2, 2])
            .run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
