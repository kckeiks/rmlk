#define LONG_TERNARY_OP(TYPENAME, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,        /* The number of elements in the output.     */\
    const size_t rank,             /* The rank of input tensor (must be > 0).   */\
    const size_t *info,            /* The shape and stride of the input.        */\
    const TYPENAME *x_data,        /* The input tensor data.                    */\
    const TYPENAME *y_data,        /* The input tensor data.                    */\
    const TYPENAME *z_data,        /* The input tensor data.                    */\
    TYPENAME *out                  /* The output data.                          */\
) { \
    const size_t *dims = info; \
    const size_t *x_strides = info + rank; \
    const size_t *y_strides = info + 2 * rank; \
    const size_t *z_strides = info + 3 * rank; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        unsigned int tmp_i = i; \
        unsigned int x_i = 0; \
        unsigned int y_i = 0; \
        unsigned int z_i = 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            unsigned int i_dim = tmp_i % dims[d]; \
            x_i += i_dim * x_strides[d]; \
            y_i += i_dim * y_strides[d]; \
            z_i += i_dim * z_strides[d]; \
            tmp_i /= dims[d]; \
        } \
        TYPENAME x = x_data ? x_data[x_i] : out[i]; \
        TYPENAME y = y_data ? y_data[y_i] : out[i]; \
        TYPENAME z = z_data ? z_data[z_i] : out[i]; \
        TYPENAME fx; \
        FUNC\
        out[i] = fx; \
    } \
} \

#define TERNARY_OP(TYPENAME, FORWARD, FUNC) \
    LONG_TERNARY_OP(TYPENAME, FORWARD, fx = (FUNC);)