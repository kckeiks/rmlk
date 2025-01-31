#include "unary_op_macro.cuh"
#include <math.h>


UNARY_OP(__half, sqrt_fwd_f16, hsqrt(x))

UNARY_OP(float, sqrt_fwd_f32, sqrtf(x))

UNARY_OP(double, sqrt_fwd_f64, sqrt(x))
