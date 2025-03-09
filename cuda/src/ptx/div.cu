#include "binary_op_macros.cuh"
#include "cuda_fp16.h"

BINARY_OP(__half, div_fwd_f16, x / y)

BINARY_OP(float, div_fwd_f32, x / y)

BINARY_OP(double, div_fwd_f64, x / y)
