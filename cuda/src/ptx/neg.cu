#include "unary_op_macro.cuh"
#include "cuda_fp16.h"

UNARY_OP(__half, neg_fwd_f16,  __hmul(x, __float2half(-1.0f)))

UNARY_OP(float, neg_fwd_f32, x * -1.0f)

UNARY_OP(double, neg_fwd_f64, x * -1.0)

UNARY_OP(int32_t, neg_fwd_i32, x * -1)

UNARY_OP(int64_t, neg_fwd_i64, x * -1)