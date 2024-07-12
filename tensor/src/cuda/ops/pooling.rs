use crate::cuda::data::CudaData;
use crate::{Context, Error, Result};
use cudarc::cudnn::{sys, Cudnn, CudnnDataType, CudnnError, TensorDescriptor};
use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
use std::marker::PhantomData;
use std::sync::Arc;

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let cudnn = Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;
    // Input data tensor.
    let x = ctx.get_input(0)?;

    // Todo: Let's define an attributes object.

    // Todo: Finish.
    Ok(())
}

pub struct CudnnExtended(Arc<Cudnn>);

impl CudnnExtended {
    pub fn create_poolingnd<T: CudnnDataType>(
        self,
        input: &[std::ffi::c_int],
        pads: &[std::ffi::c_int],
        strides: &[std::ffi::c_int],
        mode: sys::cudnnPoolingMode_t,
        nan_propagation: sys::cudnnNanPropagation_t,
    ) -> std::result::Result<PoolingDescriptor<T>, CudnnError> {
        // Todo: Create pooling descriptor.
        // Todo: call cudnn to create descriptor.
        todo!()
    }
}

pub struct PoolingDescriptor<T> {
    desc: sys::cudnnPoolingDescriptor_t,
    #[allow(unused)]
    handle: Arc<Cudnn>,
    marker: PhantomData<T>,
}

pub struct PoolingForward<'a, P, X, Y> {
    pooling: &'a PoolingDescriptor<P>,
    x: &'a TensorDescriptor<X>,
    y: &'a TensorDescriptor<Y>,
}

impl<'a, P, X, Y> PoolingForward<'a, P, X, Y>
where
    P: CudnnDataType,
    X: CudnnDataType,
    Y: CudnnDataType,
{
    pub fn launch<Input: DevicePtr<X>, Output: DevicePtrMut<Y>>(
        &self,
        (alpha, beta): (Y, Y),
        input: &X,
        y: &mut Y,
    ) -> Result<()> {
        todo!()
    }
}
