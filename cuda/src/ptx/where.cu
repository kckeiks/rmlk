#include "ternary_op_macro.cuh"
#include "cuda_fp16.h"

TERNARY_OP_WITH_TYPENAME_Z(__half, bool, where_fwd_f16, z ? x : y)

TERNARY_OP_WITH_TYPENAME_Z(float, bool, where_fwd_f32, z ? x : y)

TERNARY_OP_WITH_TYPENAME_Z(double, bool, where_fwd_f64, z ? x : y)

TERNARY_OP_WITH_TYPENAME_Z(int32_t, bool, where_fwd_i32, z ? x : y)

TERNARY_OP_WITH_TYPENAME_Z(int64_t, bool, where_fwd_i64, z ? x : y)
