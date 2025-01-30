#include "ternary_op_macro.cuh"

TERNARY_OP(__half, bmul_fwd_f16, z ? x : y)

TERNARY_OP(float, bmul_fwd_f32, z ? x : y)

TERNARY_OP(double, bmul_fwd_f64, z ? x : y)
