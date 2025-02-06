use crate::attributes;
use crate::core::allocators::ScratchAllocator;
use crate::core::error::InternalError::InvalidTensorShape;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits,
};
use num_traits::{Num, Signed, ToPrimitive};
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

/*
    [
        [
            [0, 1, 2],
            [3, 4, 5]
        ],
        [
            [6, 7, 8],
            [9, 10, 11]
        ],
        [
            [12, 13, 14],
            [15, 16, 17]
        ]
    ]

    // axis = 0
    // [[[2, 1], [1, 2]], [[1, 0], [1, 2]]]
    //

    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    shape = [3, 2, 3]
    stride = [6 ,3 , 1]



    for (int d = num_dims - 1; d >= 0; d--) { \
        unsigned int i_dim = tmp_i % dims[d]; \
        a_i += i_dim * a_strides[d]; \
        b_i += i_dim * b_strides[d]; \
        tmp_i /= dims[d]; \
    } \

    for

    [
        [
            [0, 1],
            [2, 3]
        ],
        [
            [4, 5],
            [6, 7]
        ],
        [
            [8, 9],
            [10, 11]
        ]
    ]

    // axis = 0
    // [[[2, 1], [1, 2]], [[1, 0], [1, 2]]]
    //

    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    shape = [3, 2, 3]
    stride = [4 ,2 , 1]

    axis = 0
    indices = [0, 1]
    idx = [0, 4]
    out_shape = [2, 2, 2]

    axis = 0
    indices = [0, 2]
    idx = [0, 8]
    out_shape = [2, 2, 2]

    axis = 0
    indices = [1, 2]
    idx = [0, 8]
    out_shape = [2, 2, 2]

    tmp_i=self.current
    i = 0

    d_i = tmp_i % shape[d_i]
    i += d_i * stride[d_i]
    tmp_i = tmp_i / shape[d_i]

    tmp_i=self.current
    i = 0

    d_i = tmp_i % shape[d_i]
    i += d_i * stride[d_i]
    tmp_i = tmp_i / shape[d_i]

    0*4 + 0*2 + 0*1 = 0
    0*4 + 0*2 + 1*1 = 1

    0*4 + 1*2 + 0*1 = 2
    0*4 + 1*2 + 1*1 = 3

    1*4 + 0*2 + 0*1 = 4
    1*4 + 0*2 + 1*1 = 5

    1*4 + 1*2 + 0*1 = 6
    1*4 + 1*2 + 1*1 = 7

    2*4 + 0*2 + 0*1 = 8
    2*4 + 0*2 + 1*1 = 9

    2*4 + 1*2 + 0*1 = 10
    2*4 + 1*2 + 1*1 = 11
*/

pub struct GatherBackend {
    device: Arc<CudaDevice>,
    f: CudaFunction,
}

impl GatherBackend {
    pub fn new(device: Arc<CudaDevice>, f: CudaFunction) -> Self {
        Self { device, f }
    }
}

impl GatherBackend {
    fn compute_output_shape(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()> {
        let data = ctx.get_input(0)?;
        let indices = ctx.get_input(1)?;

        let data_rank = data.shape().len();
        let indices_rank = indices.shape().len();

        if data_rank < axis {
            return Err(InternalError::InvalidAxis { axis: axis as i64 });
        }

        let alloc = ctx.execution_state().scratch_alloc().clone();

        let output_shape_buf = alloc.allocate::<usize>(data_rank - 1 + indices_rank)?;

        output_shape_buf[..axis].copy_from_slice(&data.shape()[..axis]);
        output_shape_buf[axis..axis + indices_rank].copy_from_slice(&indices.shape());

        if axis + 1 < data_rank {
            output_shape_buf[axis + indices_rank..].copy_from_slice(&data.shape()[axis + 1..]);
        }

        let output = ctx.get_output(0)?;
        let dst_id = output.dst_id();

        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape_buf, dst_id)?;

        Ok(())
    }

    fn _slice_data<D, S>(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        S: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num + Copy,
        i64: From<S>,
    {
        let data = ctx.get_input(0)?;
        let indices = ctx.get_input(1)?;
        let output = ctx.get_output(0)?;

        let data_ptr = data.try_dev_data_ptr()?;
        let indices_ptr = indices.try_dev_data_ptr()?;
        let mut output_ptr = output.try_dev_data_ptr_mut()?;

        let alloc = ctx.execution_state().scratch_alloc().clone();

        let view = indices_ptr.data::<S>();
        let indices_on_host = alloc.allocate_fill::<S>(view.len(), S::zero())?;
        self.device
            .dtoh_sync_copy_into(view.as_ref(), indices_on_host)
            .map_err(rmlk_cuda::Error::from)?;

        for (i, dim_idx) in
            DataIterator::new(indices.shape(), indices.stride(), indices_on_host).enumerate()
        {
            let norm_i = normalize_index(i64::from(*dim_idx), data.shape().len())?;
            let start = norm_i * data.stride()[axis];

            let rem = match axis == data.shape().len() - 1 {
                true => 1,
                false => data.shape()[axis + 1..].iter().product(),
            };

            let view = data_ptr.data::<D>();
            let subs = view.slice(start..start + rem);

            let mut out_data = output_ptr.data_mut::<D>();
            let mut out_slice = out_data.slice_mut(i * rem..i * rem + rem);

            self.device.dtod_copy(&subs, &mut out_slice)?;
        }

        Ok(())
    }

    fn slice_data<D>(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let dtype = ctx.get_input(0)?.try_dev_data_ptr()?.dtype();

        match dtype {
            DataType::Int32 => {
                self._slice_data::<D, i32>(axis, ctx)?;
            }
            DataType::Int64 => {
                self._slice_data::<D, i64>(axis, ctx)?;
            }
            _ => {
                return Err(InternalError::UnsupportedOpForDataType {
                    dtype,
                    op: Op::Gather,
                })
            }
        }

        Ok(())
    }

    fn compute_gather<D, T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let axis = ctx
            .get_attributes()
            .map(attributes::gather::get_axis)
            .unwrap_or(0);

        let data_rank = ctx.get_input(0)?.shape().len();
        let norm_axis = normalize_index(axis as i64, data_rank)?;

        self.compute_output_shape(norm_axis, ctx)?;

        let output = ctx.get_output(0)?;
        common::init_tensor_device_data::<D>(&self.device, output)?;

        self.slice_data::<D>(norm_axis, ctx)?;

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: GatherKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_gather::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType { op: Op::Add, dtype }),
        }
    }
}

pub struct DataIterator<'a, T> {
    shape: &'a [usize],
    stride: &'a [usize],
    data: &'a [T],
    current: usize,
    rank: usize,
}

impl<'a, T> DataIterator<'a, T> {
    pub fn new(shape: &'a [usize], stride: &'a [usize], data: &'a [T]) -> Self {
        debug_assert_eq!(shape.len(), stride.len());

        let rank = shape.iter().product::<usize>();
        Self {
            shape,
            stride,
            data,
            rank,
            current: 0,
        }
    }
}

impl<'a, T> Iterator for DataIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let old_current = self.current;
        self.current += 1;

        match self.shape.len() {
            0 => self.data.get(old_current),
            _ => {
                let mut i = 0;
                let mut tmp_i = self.current;
                for (d_i, dim) in self.shape.iter().enumerate().rev() {
                    let norm_i = tmp_i % dim;
                    i += norm_i * self.stride[d_i];
                    tmp_i /= dim;
                }

                self.data.get(i)
            }
        }
    }
}

pub fn normalize_index(index: i64, rank: usize) -> Result<usize> {
    let norm_index = match index < 0 {
        true => rank
            .checked_sub(
                index
                    .unsigned_abs()
                    .to_usize()
                    .expect("the runtime to be running in a `64-bit` system"),
            )
            .ok_or(InternalError::InvalidAxis { axis: index })?,
        false => {
            index
                .to_usize()
                .expect("the runtime to be running in a `64-bit` system")
                % rank
        }
    };

    Ok(norm_index)
}

trait GatherKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        alloc: &ScratchAllocator,
        a_dev_data: &CudaSlice<T>,
        a_shape: &[usize],
        a_stride: &[usize],
        b_dev_data: &CudaSlice<T>,
        b_shape: &[usize],
        b_stride: &[usize],
        c_shape: &[usize],
        c_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl GatherKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        alloc: &ScratchAllocator,
        a_dev_data: &CudaSlice<T>,
        _a_shape: &[usize],
        a_stride: &[usize],
        b_dev_data: &CudaSlice<T>,
        _b_shape: &[usize],
        b_stride: &[usize],
        c_shape: &[usize],
        c_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        todo!()
    }
}

pub struct NoOpKernel(());

impl GatherKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaDevice>,
        _: CudaFunction,
        _: &ScratchAllocator,
        _: &CudaSlice<T>,
        _: &[usize],
        _: &[usize],
        _: &CudaSlice<T>,
        _: &[usize],
        _: &[usize],
        _: &[usize],
        _: &mut CudaSlice<T>,
    ) -> Result<()> {
        Ok(())
    }
}
