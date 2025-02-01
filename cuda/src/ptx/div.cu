#include "binary_op_macros.cuh"

BINARY_OP(__half, div_fwd_f16, x / y)

BINARY_OP(float, div_fwd_f32, x / y)

BINARY_OP(double, div_fwd_f64, x / y)
