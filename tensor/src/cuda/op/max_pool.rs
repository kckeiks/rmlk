use crate::attribute::pooling::MaxPoolAttributes;
use crate::cuda::data::CudaData;
use crate::{Context, Error, Result};
use cudarc::cudnn::{Cudnn, PoolingForward};
use cudarc::driver::CudaDevice;
use log::debug;
use rmlk_ir::DataType;
use std::sync::Arc;

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

    let kernel_dims = match x.shape().len() {
        4 => 2,
        5 => 3,
        _ => unreachable!("we already checked the dimensions of x for the supported dimensions"),
    };

    let attrs = MaxPoolAttributes::new(
        ctx.get_attributes().ok_or(Error::MissingAttributes)?,
        kernel_dims,
    )?;

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
    use crate::cuda::data::CudaData;
    use crate::cuda::kernel::CudaKernel;
    use crate::kernel::{Context, Kernel};
    use crate::test_utils;
    use crate::test_utils::{TestMaxPoolAttributes, TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};

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

        let (_, state) = test_utils::build_graph_and_state(params);
        let mut context = Context::new(state, 1).unwrap();

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
