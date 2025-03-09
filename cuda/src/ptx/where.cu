#include "ternary_op_macro.cuh"
#include "cuda_fp16.h"

TERNARY_OP(__half, where_fwd_f16, z ? x : y)

TERNARY_OP(float, where_fwd_f32, z ? x : y)

TERNARY_OP(double, where_fwd_f64, z ? x : y)
