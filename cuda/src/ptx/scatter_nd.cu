#include "scatter_op_macros.cuh"
#include "cuda_fp16.h"

SCATTER_ND_OP(__half, scatter_nd_fwd_f16, __hadd(in, __hmul(__float2half(0.0f), out)))
SCATTER_ND_OP(__half, scatter_nd_add_fwd_f16, __hadd(in, out))
SCATTER_ND_OP(__half, scatter_nd_mul_fwd_f16, __hmul(in, out))
SCATTER_ND_OP(__half, scatter_nd_max_fwd_f16, in > out ? in : out)
SCATTER_ND_OP(__half, scatter_nd_min_fwd_f16, in < out ? in : out)

SCATTER_ND_OP(float, scatter_nd_fwd_f32, in + 0.0f * out)
SCATTER_ND_OP(float, scatter_nd_add_fwd_f32, in + out)
SCATTER_ND_OP(float, scatter_nd_mul_fwd_f32, in * out)
SCATTER_ND_OP(float, scatter_nd_max_fwd_f32, in > out ? in : out)
SCATTER_ND_OP(float, scatter_nd_min_fwd_f32, in < out ? in : out)

SCATTER_ND_OP(double, scatter_nd_fwd_f64, in + 0.0 * out)
SCATTER_ND_OP(double, scatter_nd_add_fwd_f64, in + out)
SCATTER_ND_OP(double, scatter_nd_mul_fwd_f64, in * out)
SCATTER_ND_OP(double, scatter_nd_max_fwd_f64, in > out ? in : out)
SCATTER_ND_OP(double, scatter_nd_min_fwd_f64, in < out ? in : out)

SCATTER_ND_OP(int32_t, scatter_nd_fwd_i32, in + 0 * out)
SCATTER_ND_OP(int32_t, scatter_nd_add_fwd_i32, in + out)
SCATTER_ND_OP(int32_t, scatter_nd_mul_fwd_i32, in * out)
SCATTER_ND_OP(int32_t, scatter_nd_max_fwd_i32, in > out ? in : out)
SCATTER_ND_OP(int32_t, scatter_nd_min_fwd_i32, in < out ? in : out)

SCATTER_ND_OP(int64_t, scatter_nd_fwd_i64, in + 0 * out)
SCATTER_ND_OP(int64_t, scatter_nd_add_fwd_i64, in + out)
SCATTER_ND_OP(int64_t, scatter_nd_mul_fwd_i64, in * out)
SCATTER_ND_OP(int64_t, scatter_nd_max_fwd_i64, in > out ? in : out)
SCATTER_ND_OP(int64_t, scatter_nd_min_fwd_i64, in < out ? in : out)