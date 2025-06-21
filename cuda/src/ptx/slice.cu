#include "slice_op_macros.cuh"
#include "cuda_fp16.h"

SLICE_OP(__half, int32_t, slice_fwd_f16_with_index_i32)
SLICE_OP(__half, int64_t, slice_fwd_f16_with_index_i64)
SLICE_OP(float, int32_t, slice_fwd_f32_with_index_i32)
SLICE_OP(float, int64_t, slice_fwd_f32_with_index_i64)
SLICE_OP(double, int32_t, slice_fwd_f64_with_index_i32)
SLICE_OP(double, int64_t, slice_fwd_f64_with_index_i64)
SLICE_OP(int32_t, int32_t, slice_fwd_i32_with_index_i32)
SLICE_OP(int32_t, int64_t, slice_fwd_i32_with_index_i64)
SLICE_OP(int64_t, int32_t, slice_fwd_i64_with_index_i32)
SLICE_OP(int64_t, int64_t, slice_fwd_i64_with_index_i64)
