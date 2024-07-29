use crate::cuda::data::CudaData;
use crate::{Context, Error};
use crate::{Result, Tensor};
use cudarc::cudnn::{sys, ActivationForward, Cudnn, CudnnDataType};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use rmlk_ir::DataType;
use std::sync::Arc;

// pub fn compute_v2() -> {}

pub fn compute_v2<T: CudnnDataType>(
    device: Arc<CudaDevice>,
    (alpha, beta): (T, T),
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    y_data: &mut CudaSlice<T>,
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let cudnn = Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;

    let x_desc = cudnn
        .create_nd_tensor::<T>(x_shape, x_stride)
        .map_err(|_| Error::CudnnInternal)?;

    let y_desc = cudnn
        .create_nd_tensor::<T>(x_shape, x_stride)
        .map_err(|_| Error::CudnnInternal)?;

    let activation_desc = cudnn
        .create_activation::<T>(
            sys::cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
            sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            f64::MAX,
        )
        .map_err(|_| Error::CudnnInternal)?;

    let op = ActivationForward {
        act: &activation_desc,
        x: &x_desc,
        y: &y_desc,
    };

    op.launch((alpha, beta), x_data, y_data)
        .map_err(|_| Error::CudnnInternal)?;

    Ok(())
}

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let cudnn = Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;
    let x = ctx.get_input(0)?;

    let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
    let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

    match x.dtype() {
        DataType::Float => {
            let x_data = x.data().unwrap().f32().unwrap();
            let x_desc = cudnn
                .create_nd_tensor::<f32>(&x_shape, &x_stride)
                .map_err(|_| Error::CudnnInternal)?;

            let output_size = x.shape().iter().product();
            let mut y_data = device
                .alloc_zeros::<f32>(output_size)
                .map_err(|_| Error::CudnnInternal)?;
            let y_desc = cudnn
                .create_nd_tensor::<f32>(&x_shape, &x_stride)
                .map_err(|_| Error::CudnnInternal)?;

            let activation_desc = cudnn
                .create_activation::<f32>(
                    sys::cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
                    sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                    f64::MAX,
                )
                .map_err(|_| Error::CudnnInternal)?;

            let op = ActivationForward {
                act: &activation_desc,
                x: &x_desc,
                y: &y_desc,
            };

            op.launch((1.0, 0.0), x_data, &mut y_data)
                .map_err(|_| Error::CudnnInternal)?;

            let out = ctx.get_output_mut(0)?;
            out.init(CudaData::F32(y_data));
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
    use crate::cuda::kernel::CudaKernel;
    use crate::cuda::kernel::activation::compute_v2;
    use crate::kernel::{Context, Kernel};
    use crate::test_utils::{TestNode, TestParams};
    use crate::{test_utils, Error, Tensor};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};

    #[test]
    fn test_relu_f32_v2() {
        let device = CudaDevice::new(0).unwrap();
        let mut x = Tensor::<CudaData>::new_with_shape(DataType::Float, vec![1, 1, 2, 2]);
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_data = device.htod_copy(vec![-1.0, 2.0, -3.0, 100.0]).unwrap();

        let mut y_data = device.alloc_zeros(x.shape().iter().product()).unwrap();
        compute_v2::<f32>(
            device.clone(),
            (1.0, 0.0),
            &x_data,
            &x_shape,
            &x_stride,
            &mut y_data,
        )
        .unwrap();
        let result = device.dtoh_sync_copy(&y_data).unwrap();

        assert_eq!(result, vec![0.0, 2.0, 0.0, 100.0])
    }

    #[test]
    fn test_relu_f32() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 2, 2];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape,
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![-1.0, 2.0, -3.0, 100.0]).unwrap(),
            )),
        };

        let node_c = TestNode {
            shape: vec![1, 1, 2, 2],
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a],
            outputs: vec![node_c],
            attributes: Vec::new(),
            op: Op::Relu,
        };

        let (_, mut state) = test_utils::build_graph_and_state(params);
        let mut context = Context::new(&mut state, 1).unwrap();

        let cuda_kernel = CudaKernel::new(Op::Relu, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![0.0, 2.0, 0.0, 100.0])
    }
}
