#include "transpose_op_macros.cuh"
#include "cuda_fp16.h"

TRANSPOSE_OP(__half, transpose_fwd_f16)
TRANSPOSE_OP(float, transpose_fwd_f32)
TRANSPOSE_OP(double, transpose_fwd_f64)
TRANSPOSE_OP(int32_t, transpose_fwd_i32)
TRANSPOSE_OP(int64_t, transpose_fwd_i64)