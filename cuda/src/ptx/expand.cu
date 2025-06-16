#include "expand_op_macros.cuh"
#include "cuda_fp16.h"

#include <stdint.h>

EXPAND_OP(__half, expand_fwd_f16)
EXPAND_OP(float, expand_fwd_f32)
EXPAND_OP(double, expand_fwd_f64)
EXPAND_OP(int32_t, expand_fwd_i32)
EXPAND_OP(uint32_t, expand_fwd_u32)
EXPAND_OP(int64_t, expand_fwd_i64)
EXPAND_OP(uint64_t, expand_fwd_u64)

