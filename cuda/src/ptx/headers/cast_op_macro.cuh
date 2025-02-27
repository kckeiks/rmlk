#define CAST_OP(IN_TYPE, OUT_TYPE, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t numel, \
    const IN_TYPE *data, \
    OUT_TYPE *out \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        IN_TYPE x = data[i]; \
        out[i] = (FUNC); \
    } \
}
