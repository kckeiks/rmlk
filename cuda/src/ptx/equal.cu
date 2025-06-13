#include "binary_op_macros.cuh"
#include "cuda_fp16.h"

BINARY_OP_WITH_OUTPUT_TYPE(__half, bool, equal_fwd_f16, x == y)

BINARY_OP_WITH_OUTPUT_TYPE(float, bool, equal_fwd_f32, x == y)

BINARY_OP_WITH_OUTPUT_TYPE(double, bool, equal_fwd_f64, x == y)

BINARY_OP_WITH_OUTPUT_TYPE(int32_t, bool, equal_fwd_i32, x == y)

BINARY_OP_WITH_OUTPUT_TYPE(int64_t, bool, equal_fwd_i64, x == y)
