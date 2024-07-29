use crate::cuda::data::CudaData;
use crate::error::Error;
use crate::error::Result;
use crate::kernel::Context;
use cudarc::cudnn;
use cudarc::cudnn::{CudnnDataType, PoolingForward};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use rmlk_ir::DataType;
use std::ops::AddAssign;
use std::sync::Arc;

pub fn compute_output_shape<T: Num + Copy + AddAssign>(
    x_shape: &[T],
    y_shape: &mut [T],
) -> Result<()> {
    if x_shape.len() < 4 {
        return Err(Error::InvalidInputShapes);
    }

    if x_shape.len() != y_shape.len() {
        return Err(Error::InvalidBufferSize);
    }

    for i in 0..2 {
        y_shape[i] = x_shape[1];
    }

    // For reference, see https://github.com/onnx/onnx/blob/main/docs/Operators.md#outputs-59.
    for i in 2..y_shape.len() {
        y_shape[i] = T::one();
    }

    Ok(())
}

pub fn compute_v2<T>(
    device: Arc<CudaDevice>,
    (alpha, beta): (T, T),
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    y_data: &mut CudaSlice<T>,
    y_shape: &[i32],
    y_stride: &[i32],
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let cudnn = cudnn::Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;

    debug_assert!(x_shape.len() == 4 || x_shape.len() == 5);

    let kernel_shape = x_shape[2..].as_ref();
    let pads = x_shape[2..].iter().map(|_| 0).collect::<Box<[i32]>>();
    let strides = x_shape[2..].iter().map(|_| 1).collect::<Box<[i32]>>();

    debug!(
        "x_shape={x_shape:?},\
        kernel_shape={kernel_shape:?},\
        pads={pads:?},\
        strides={strides:?},\
        out_shape={y_shape:?},\
        out_stride={y_stride:?}"
    );

    let x_desc = cudnn
        .create_nd_tensor::<T>(x_shape, x_stride)
        .map_err(|_| Error::CudnnInternal)?;

    let pooling = cudnn
        .create_poolingnd::<T>(
            &kernel_shape,
            &pads,
            &strides,
            cudarc::cudnn::sys::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
            cudarc::cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
        )
        .map_err(|_| Error::CudnnInternal)
        .unwrap();

    let out_desc = cudnn
        .create_nd_tensor(y_shape, y_stride)
        .map_err(|_| Error::CudnnInternal)?;

    let forward_f = PoolingForward {
        pooling: &pooling,
        x: &x_desc,
        y: &out_desc,
    };

    forward_f
        .launch((alpha, beta), x_data, y_data)
        .map_err(|_| Error::CudnnInternal)
        .unwrap();

    Ok(())
}

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let cudnn = cudnn::Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;

    let x = ctx.get_input(0)?;

    debug_assert!(x.shape().len() == 4 || x.shape().len() == 5);

    // Todo: let's fix the casting here.
    let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
    let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

    let kernel_shape = x_shape[2..].as_ref();
    let pads = x_shape[2..].iter().map(|_| 0).collect::<Box<[i32]>>();
    let strides = x_shape[2..].iter().map(|_| 1).collect::<Box<[i32]>>();

    // Todo: if there is no shape, simply set it.
    // Although, we probably need some other type of way
    // to flag tensors when they could have any shape.

    let mut out_shape = x_shape.clone();
    // For reference, see https://github.com/onnx/onnx/blob/main/docs/Operators.md#outputs-59.
    for i in 2..out_shape.len() {
        out_shape[i] = 1;
    }

    // Todo: Make helper functions.
    let dims = out_shape.len();
    let mut out_stride = vec![0i32; dims];
    out_stride[dims - 1] = 1;
    for i in (0..dims - 1).rev() {
        out_stride[i] += out_stride[i + 1] * out_stride[i + 1];
    }
    let out_slice_size = out_shape.iter().product::<i32>();

    debug!(
        "x_shape={x_shape:?},\
        kernel_shape={kernel_shape:?},\
        pads={pads:?},\
        strides={strides:?},\
        out_shape={out_shape:?},\
        out_stride={out_stride:?}"
    );

    match x.dtype() {
        DataType::Float => {
            let x_desc = cudnn
                .create_nd_tensor::<f32>(&x_shape, &x_stride)
                .map_err(|_| Error::CudnnInternal)?;

            let pooling = cudnn
                .create_poolingnd::<f32>(
                    &kernel_shape,
                    &pads,
                    &strides,
                    cudarc::cudnn::sys::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                    cudarc::cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                )
                .map_err(|_| Error::CudnnInternal)
                .unwrap();

            let out_desc = cudnn
                .create_nd_tensor(out_shape.as_ref(), out_stride.as_ref())
                .map_err(|_| Error::CudnnInternal)?;
            let mut out_slice = device
                .alloc_zeros::<f32>(usize::try_from(out_slice_size).unwrap())
                .map_err(|_| Error::CudnnInternal)?;

            let forward_f = PoolingForward {
                pooling: &pooling,
                x: &x_desc,
                y: &out_desc,
            };

            forward_f
                .launch(
                    (1.0, 0.0),
                    x.data().ok_or(Error::MissingTensor)?.f32()?,
                    &mut out_slice,
                )
                .map_err(|_| Error::CudnnInternal)
                .unwrap();

            let out_tensor = ctx.get_output_mut(0)?;
            out_tensor.init(CudaData::F32(out_slice));
        }
        _ => {
            todo!()
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::cuda::data::CudaData;
    use crate::cuda::kernel::global_average_pool::{compute_output_shape, compute_v2};
    use crate::cuda::kernel::CudaKernel;
    use crate::kernel::{Context, Kernel};
    use crate::test_utils::{TestNode, TestParams};
    use crate::{test_utils, Tensor};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};

    #[test]
    fn test_global_average_pool_f32_2d_v2() {
        let device = CudaDevice::new(0).unwrap();

        let mut x = Tensor::<CudaData>::new_with_shape(DataType::Float, vec![1, 1, 3, 3]);
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_data = device
            .htod_copy(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();

        let mut y_shape = x.shape().clone();
        compute_output_shape(x.shape(), y_shape.as_mut_slice()).unwrap();

        let mut y = Tensor::<CudaData>::new_with_shape(DataType::Float, y_shape.to_vec());
        let y_stride = y.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let y_shape = y_shape.iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let mut y_data = device
            .alloc_zeros(y_shape.iter().map(|n| *n as usize).product())
            .unwrap();

        compute_v2::<f32>(
            device.clone(),
            (1.0, 0.0),
            &x_data,
            &x_shape,
            &x_stride,
            &mut y_data,
            &y_shape,
            &y_stride,
        )
        .unwrap();
        let result = device.dtoh_sync_copy(&y_data).unwrap();

        assert_eq!(result, vec![5.0])
    }

    #[test]
    fn test_global_average_pool_f32_2d() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 3, 3];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape,
            dtype,
            data: Some(CudaData::F32(
                device
                    .htod_copy(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                    .unwrap(),
            )),
        };

        let node_c = TestNode {
            shape: vec![1, 1, 1, 1],
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a],
            outputs: vec![node_c],
            attributes: Vec::new(),
            op: Op::GlobalAveragePool,
        };

        let (_, mut state) = test_utils::build_graph_and_state(params);
        let mut context = Context::new(&mut state, 1).unwrap();

        let cuda_kernel = CudaKernel::new(Op::GlobalAveragePool, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![5.0])
    }
}
