#include <cstdint>
#include "aggregation_op_macros.cuh"
#include "cuda_fp16.h"

REDUCE_MEAN_OP_THREAD_PER_OUTPUT(__half, reduce_mean_fwd_f16, double, __half2float(x), double, __float2half(static_cast<float>(div)))
REDUCE_MEAN_OP_THREAD_PER_OUTPUT(float, reduce_mean_fwd_f32, double, x, double, static_cast<float>(div))
REDUCE_MEAN_OP_THREAD_PER_OUTPUT(double, reduce_mean_fwd_f64, double, x, double, div)
REDUCE_MEAN_OP_THREAD_PER_OUTPUT(uint32_t,reduce_mean_fwd_u32, uint64_t, x, double, static_cast<uint32_t>(div))
REDUCE_MEAN_OP_THREAD_PER_OUTPUT(uint64_t,reduce_mean_fwd_u64, uint64_t, x, double, static_cast<uint64_t>(div))
