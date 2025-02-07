use crate::core::allocators::ScratchAllocator;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use crate::utils::DataIterator;
use crate::{attributes, utils};
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits,
};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::fmt::Debug;
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
}

impl GatherBackend {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
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
        S: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num + Copy + Debug,
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

        // Given `axis` and the `data` shape S=[d_i, d_i2, ..., d_n-1, dn],
        // we split the shape at `axis` to create:
        //
        //              S_lhs = [d_1, d_2, ..., d_axis-1]
        //              S_rhs = [d_axis+1, d_axis+2, ..., d_n]
        //
        // We take sub-slices of size `d_axis+1 * d_axis+2 * ... * d_n` from `data`
        // using `indices`, `axis` and the `data` stride and
        // copy these sub-slices to the output tensor.
        // We must traverse the dimensions `d_axis+1, d_axis+2, ..., d_n`,
        // so we loop `d_1 * d_2 * ... * d_axis-1` times while advancing by
        // section of size `d_axis+1 * d_axis+2 * ... * d_n`.
        let (lhs, _) = data.shape().split_at(axis);
        // Element count on the dimensions derived from `axis`.
        let elem_count = data.shape()[axis];
        // Count of the slices we will write in the output tensor.
        let mut slice_count = 0;
        // Size of one slice.
        let slice_size = match axis == data.shape().len() - 1 {
            true => 1,
            false => data.shape()[axis + 1..].iter().product(),
        };
        // We stack these dimensions on the `lhs`.
        let stack_size = lhs.iter().product::<usize>();
        for stack_level in 0..stack_size {
            for dim_i in DataIterator::new(indices.shape(), indices.stride(), indices_on_host) {
                // Get the index.
                let norm_i = utils::normalize_index(i64::from(*dim_i), data.shape()[axis])?;
                let start = stack_level * elem_count + (norm_i * data.stride()[axis]);

                // Slice the input.
                let view = data_ptr.data::<D>();
                let subslice = view.slice(start..start + slice_size);

                // Create a writeable slice of the output.
                let mut out_data = output_ptr.data_mut::<D>();
                let mut out_slice = out_data.slice_mut(slice_count * slice_size..slice_count * slice_size + slice_size);

                // Write to the output slice.
                self.device.dtod_copy(&subslice, &mut out_slice)?;

                slice_count += 1;
            }
        }

        Ok(())
    }

    fn slice_data<D>(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let dtype = ctx.get_input(1)?.try_dev_data_ptr()?.dtype();

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
        let norm_axis = utils::normalize_index(axis as i64, data_rank)?;

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

fn split_shape(axis: usize, shape: &[usize]) -> (&[usize], usize, &[usize]) {
    debug_assert!(shape.is_empty());
    debug_assert!(shape.len() > axis);
    (&shape[..axis], shape[axis], &shape[axis..])
}
