#define UNARY_OP(TYPENAME, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,  /* The number of elements in the input.  */\
    const TYPENAME *data,    /* The input tensor data.                */\
    TYPENAME *out            /* The output data.                      */\
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        TYPENAME x = data[i]; \
        out[i] = (FUNC); \
    } \
}