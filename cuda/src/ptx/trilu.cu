#include "trilu_op_macros.cuh"
#include "cuda_fp16.h"

TRILU_OP(__half, trilu_fwd_f16)

TRILU_OP(float, trilu_fwd_f32)

TRILU_OP(double, trilu_fwd_f64)

TRILU_OP(int32_t, trilu_fwd_i32)

TRILU_OP(int64_t, trilu_fwd_i64)


