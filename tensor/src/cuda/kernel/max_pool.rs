use crate::attribute::pooling::MaxPoolAttributes;
use crate::cuda::data::CudaData;
use crate::{Context, Error, Result};
use cudarc::cudnn::{Cudnn, CudnnDataType, PoolingForward};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::{FromPrimitive, Num};
use rmlk_ir::DataType;
use std::ops::AddAssign;
use std::sync::Arc;

// Todo: figure out how to make this generic.
pub fn compute_output_shape<T>(
    x_shape: &[T],
    kernel_shape: &[T],
    pads: &[T],
    strides: &[T],
    y_shape: &mut [T],
    ceil_mode: bool,
) -> Result<()>
where
    T: Num + Copy + AddAssign + FromPrimitive,
    f64: From<T>,
{
    let two = T::one() + T::one();
    if kernel_shape.len() == 2 && y_shape.len() == 4 {
        let height =
            (f64::from(x_shape[2] + two * pads[0] - kernel_shape[0]) / f64::from(strides[0])) + 1.0;
        let height = match ceil_mode {
            true => height.ceil(),
            false => height.floor(),
        };

        let width =
            (f64::from(x_shape[3] + two * pads[1] - kernel_shape[1]) / f64::from(strides[1])) + 1.0;
        let width = match ceil_mode {
            true => width.ceil(),
            false => width.floor(),
        };

        y_shape[0] = x_shape[0];
        y_shape[1] = x_shape[1];
        y_shape[2] = T::from_f64(height).ok_or(Error::ComputationError)?;
        y_shape[3] = T::from_f64(width).ok_or(Error::ComputationError)?;
    } else if kernel_shape.len() == 3 && y_shape.len() == 5 {
        let depth =
            (f64::from(x_shape[1] + two * pads[0] - kernel_shape[0]) / f64::from(strides[0])) + 1.0;
        let depth = match ceil_mode {
            true => depth.ceil(),
            false => depth.floor(),
        };

        let height =
            (f64::from(x_shape[2] + two * pads[1] - kernel_shape[1]) / f64::from(strides[1])) + 1.0;
        let height = match ceil_mode {
            true => height.ceil(),
            false => height.floor(),
        };

        let width =
            (f64::from(x_shape[3] + two * pads[2] - kernel_shape[2]) / f64::from(strides[2])) + 1.0;
        let width = match ceil_mode {
            true => width.ceil(),
            false => width.floor(),
        };

        y_shape[0] = x_shape[0];
        y_shape[1] = x_shape[1];
        y_shape[2] = T::from_f64(depth).ok_or(Error::ComputationError)?;
        y_shape[3] = T::from_f64(height).ok_or(Error::ComputationError)?;
        y_shape[4] = T::from_f64(width).ok_or(Error::ComputationError)?;
    } else {
        return Err(Error::InvalidInputShapes);
    }

    Ok(())
}

pub fn compute_v2<T>(
    device: Arc<CudaDevice>,
    (alpha, beta): (T, T),
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    kernel_shape: &[i32],
    pads: &[i32],
    strides: &[i32],
    y_data: &mut CudaSlice<T>,
    y_shape: &[i32],
    y_stride: &[i32],
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let cudnn = Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;

    let x_desc = cudnn
        .create_nd_tensor::<T>(&x_shape, &x_stride)
        .map_err(|_| Error::CudnnInternal)?;

    let pooling = cudnn
        .create_poolingnd::<T>(
            kernel_shape,
            // Todo: Let's preprocess pads.
            pads,
            strides,
            cudarc::cudnn::sys::cudnnPoolingMode_t::CUDNN_POOLING_MAX,
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
}

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let cudnn = Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;
    // Input data tensor.
    let x = ctx.get_input(0)?;

    if x.shape().len() != 4 && x.shape().len() != 5 {
        return Err(Error::UnsupportedShape);
    }

    // Todo: let's fix the casting here.
    let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
    let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

    let attrs = MaxPoolAttributes::new(ctx.get_attributes().ok_or(Error::MissingAttributes)?)?;

    let out_shape = attrs.calculate_output_shape(x_shape.as_ref())?;

    let out = ctx.get_output(0)?;
    if out.shape().len() != out_shape.len() {
        return Err(Error::OutputShapeMismatch);
    }

    // Todo: if there is no shape, simply set it.
    // Although, we probably need some other type of way
    // to flag tensors when they could have any shape.
    if out
        .shape()
        .iter()
        .zip(out_shape.iter())
        .map(|(a, b)| (*a, *b as usize))
        .find(|(a, b)| a != b)
        .is_some()
    {
        return Err(Error::OutputShapeMismatch);
    }

    let out_stride = out
        .stride()
        .iter()
        .map(|v| *v as i32)
        .collect::<Box<[i32]>>();
    let out_slice_size = out_shape.iter().product::<i32>();

    debug!(
        "x_shape={x_shape:?}, \
        pads={:?}, \
        strides={:?}",
        attrs.pads(),
        attrs.strides()
    );

    match x.dtype() {
        DataType::Float => {
            let x_desc = cudnn
                .create_nd_tensor::<f32>(&x_shape, &x_stride)
                .map_err(|_| Error::CudnnInternal)?;

            let pooling = cudnn
                .create_poolingnd::<f32>(
                    attrs.kernel_shape(),
                    // Todo: Let's preprocess pads.
                    attrs.pads(),
                    attrs.strides(),
                    cudarc::cudnn::sys::cudnnPoolingMode_t::CUDNN_POOLING_MAX,
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
                .map_err(|_| Error::CudnnInternal)?;

            let out_tensor = ctx.get_output_mut(0)?;
            out_tensor.init(CudaData::F32(out_slice));
        }
        _ => return Err(Error::UnsupportedDataType),
    }

    // Todo: Finish.
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::attribute::pooling::MaxPoolAttributes;
    use crate::cuda::data::CudaData;
    use crate::cuda::kernel::max_pool::{compute_output_shape, compute_v2};
    use crate::cuda::kernel::CudaKernel;
    use crate::kernel::{Context, Kernel};
    use crate::test_utils::{TestMaxPoolAttributes, TestNode, TestParams};
    use crate::{test_utils, utils, Tensor};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};
    use std::collections::HashMap;

    #[test]
    fn test_max_pool_f32_2d_v2() {
        let device = CudaDevice::new(0).unwrap();

        let mut x = Tensor::<CudaData>::new_with_shape(DataType::Float, vec![1, 1, 4, 4]);
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_data = device
            .htod_copy(vec![
                1.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
            ])
            .unwrap();

        let attributes = test_utils::create_max_pool_attributes(TestMaxPoolAttributes {
            dilations: None,
            kernel_shape: Some(Box::new([2, 2])),
            strides: Some(Box::new([2, 2])),
            row_major_order: None,
            ceil_mode: None,
            pads: None,
        });
        let attributes = attributes
            .into_iter()
            .map(|attr| (attr.name.clone().into_boxed_str(), attr))
            .collect::<HashMap<_, _>>();

        let attrs = MaxPoolAttributes::new(&attributes).unwrap();

        let mut y_shape = vec![0; x_shape.len()].into_boxed_slice();
        compute_output_shape(
            &x_shape,
            attrs.kernel_shape(),
            attrs.pads(),
            attrs.strides(),
            &mut y_shape,
            false,
        )
        .unwrap();

        let mut y_stride = vec![0; x_shape.len()].into_boxed_slice();
        utils::calculate_stride(&y_shape, &mut y_stride);

        let mut y_data = device
            .alloc_zeros(y_shape.iter().map(|d| *d as usize).product())
            .unwrap();

        compute_v2::<f32>(
            device.clone(),
            (1.0, 0.0),
            &x_data,
            &x_shape,
            &x_stride,
            attrs.kernel_shape(),
            attrs.pads(),
            attrs.strides(),
            &mut y_data,
            &y_shape,
            &y_stride,
        )
        .unwrap();
        let result = device.dtoh_sync_copy(&y_data).unwrap();

        assert_eq!(result, vec![6.0, 8.0, 3.0, 4.0])
    }

    #[test]
    fn test_max_pool_f32_2d() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 4, 4];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape,
            dtype,
            data: Some(CudaData::F32(
                device
                    .htod_copy(vec![
                        1.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0,
                        4.0,
                    ])
                    .unwrap(),
            )),
        };

        let node_c = TestNode {
            shape: vec![1, 1, 2, 2],
            dtype,
            data: None,
        };

        let attributes = test_utils::create_max_pool_attributes(TestMaxPoolAttributes {
            dilations: None,
            kernel_shape: Some(Box::new([2, 2])),
            strides: Some(Box::new([2, 2])),
            row_major_order: None,
            ceil_mode: None,
            pads: None,
        });

        let params = TestParams {
            inputs: vec![node_a],
            outputs: vec![node_c],
            attributes,
            op: Op::MaxPool,
        };

        let (_, mut state) = test_utils::build_graph_and_state(params);
        let mut context = Context::new(&mut state, 1).unwrap();

        let cuda_kernel = CudaKernel::new(Op::MaxPool, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![6.0, 8.0, 3.0, 4.0])
    }
}
