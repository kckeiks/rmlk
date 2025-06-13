#include "binary_op_macros.cuh"
#include "cuda_fp16.h"

BINARY_OP(__half, sub_fwd_f16, x - y)

BINARY_OP(float, sub_fwd_f32, x - y)

BINARY_OP(double, sub_fwd_f64, x - y)

BINARY_OP(int32_t, sub_fwd_i32, x - y)

BINARY_OP(int64_t, sub_fwd_i64, x - y)