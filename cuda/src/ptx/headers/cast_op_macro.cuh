#define CAST_OP(IN_TYPE, OUT_TYPE, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,  /* The number of elements in the input.   */\
    const IN_TYPE *data,     /* The input tensor data.                 */\
    OUT_TYPE *out            /* The output data.                       */\
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        IN_TYPE x = data[i]; \
        out[i] = (FUNC); \
    } \
}
