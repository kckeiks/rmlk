use crate::context::ExecutionContext;
use crate::cuda;
use crate::cuda::CudaProvider;
use crate::provider::Provider;
use crate::Result;
use cudarc::cublas::CudaBlas;
use rmlk_ir::DataType;

pub fn gemm(ctx: &mut ExecutionContext<CudaProvider>) -> Result<()> {
    let lhs = ctx.get_input(0)?;
    let rhs = ctx.get_input(1)?;

    let lhs_shape = lhs.shape();
    let b = lhs_shape[..lhs_shape.len() - 2].iter().product::<usize>();
    let m = lhs_shape[lhs_shape.len() - 2];
    let k = lhs_shape[lhs_shape.len() - 1];

    let rhs_shape = rhs.shape();
    let n = rhs_shape[rhs_shape.len() - 2];

    match *lhs.dtype() {
        DataType::Float16 => {
            todo!()
        }
        DataType::Float => {
            let lhs_stride = lhs.stride();
            let rhs_stride = rhs.stride();
            let config = cuda::ops::matmul::gemm_config::<f32>(
                1.0,
                0.0,
                (b, m, n, k),
                (lhs_shape, lhs_stride),
                (rhs_shape, rhs_stride),
            )
            .unwrap();

            // Todo: Make this more generic.
            let out_tensor = ctx.allocate(DataType::Float, vec![b, m, n])?;
            // let mut out_slice = unsafe { device.alloc::<f32>(b * m * n).unwrap() };

            let out_slice = out_tensor.data_mut().unwrap().f32_mut().unwrap();

            let device = ctx.provider().device();
            let cublas = CudaBlas::new(device).unwrap();
            unsafe {
                cuda::ops::matmul::gemm_stride_batched_f32(
                    &cublas,
                    config,
                    &rhs.data().ok_or(()).unwrap().f32().unwrap().slice(..),
                    &lhs.data().ok_or(()).unwrap().f32().unwrap().slice(..),
                    out_slice,
                )
                .unwrap();
            };
            // let _ = out.init(Data::F32(out_slice));
        }
        DataType::Double => {
            todo!()
        }
        _ => todo!(),
    }
    Ok(())
}
