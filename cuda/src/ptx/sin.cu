#include "unary_op_macro.cuh"
#include "cuda_fp16.h"
#include <math.h>

UNARY_OP(__half, sin_fwd_f16, __float2half(sinf(__half2float(x))))

UNARY_OP(float, sin_fwd_f32, sinf((float)x))

UNARY_OP(double, sin_fwd_f64, sin(x))


