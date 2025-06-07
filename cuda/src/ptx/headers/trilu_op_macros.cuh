#define LONG_TRILU_OP(TYPENAME, FORWARD) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,   /* The number of elements in the output.                                                                       */\
    const size_t rank,        /* The rank of input tensor (must be > 0).                                                                     */\
    const size_t upper,       /* Indicates whether upper or lower part of matrix is retained.                                                */\
    const int k,              /* The value corresponding to the number diagonals above or below the main diagonal to exclude or include.     */\
    const size_t *info,       /* The shape and stride of the input.                                                                          */\
    const TYPENAME *input,    /* The input tensor data.                                                                                      */\
    TYPENAME *output          /* The output tensor data.                                                                                     */\
) { \
    const size_t *dims = info; \
    const size_t *strides = info + rank; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        unsigned int tmp_i = i; \
        unsigned int offset = 0; \
        unsigned int col_idx = (tmp_i % dims[2]); \
        offset +=  col_idx * strides[2]; \
        tmp_i /= dims[2]; \
        \
        unsigned int row_idx = (tmp_i % dims[1]); \
        offset +=  row_idx * strides[1]; \
        tmp_i /= dims[1]; \
        \
        unsigned int batch_idx = (tmp_i % dims[0]); \
        offset +=  batch_idx * strides[0]; \
        tmp_i /= dims[0]; \
        \
        unsigned int start = 0; \
        unsigned int end = 0; \
        if (upper) { \
            start = max(k + (int)row_idx, 0); \
            end = dims[2]; \
        } else { \
            start = 0; \
            int col_dim = (int)dims[2]; \
            end = min(k + (int)row_idx + 1, col_dim); \
        }\
        \
        if (col_idx >= start && col_idx < end) { \
            output[offset] = input[offset]; \
        } \
    } \
} \

#define TRILU_OP(TYPENAME, FORWARD) \
    LONG_TRILU_OP(TYPENAME, FORWARD)