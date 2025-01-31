#include "cuda_fp16.h"

#define UNARY_OP(TYPENAME, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t numel, \
    const TYPENAME *data, \
    TYPENAME *out \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        TYPENAME x = data[i]; \
        out[i] = (FUNC); \
    } \
}