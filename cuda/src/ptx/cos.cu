#include "unary_op_macro.cuh"
#include "cuda_fp16.h"
#include <math.h>

UNARY_OP(__half, cos_fwd_f16, __float2half(cosf(__half2float(x))))

UNARY_OP(float, cos_fwd_f32, cosf((float)x))

UNARY_OP(double, cos_fwd_f64, cos(x))


