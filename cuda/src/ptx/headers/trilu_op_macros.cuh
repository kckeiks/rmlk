#define LONG_TRILU_OP(TYPENAME, FORWARD) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,   /* The number of elements in the output.                                                                       */\
    const size_t rank,        /* The rank of input tensor (must be > 0).                                                                     */\
    const bool upper,         /* Indicates whether upper or lower part of matrix is retained.                                                */\
    const int64_t k,          /* The value corresponding to the number diagonals above or below the main diagonal to exclude or include.     */\
    const size_t *info,       /* The shape and stride of the input.                                                                          */\
    const TYPENAME *input,    /* The input tensor data.                                                                                      */\
    TYPENAME *output          /* The output tensor data.                                                                                     */\
) { \
    const size_t *dims = info; \
    const size_t *strides = info + rank; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        size_t tmp_i = i; \
        size_t offset = 0; \
        size_t col_idx = (tmp_i % dims[2]); \
        offset +=  col_idx * strides[2]; \
        tmp_i /= dims[2]; \
        \
        size_t row_idx = (tmp_i % dims[1]); \
        offset +=  row_idx * strides[1]; \
        tmp_i /= dims[1]; \
        \
        size_t batch_idx = (tmp_i % dims[0]); \
        offset +=  batch_idx * strides[0]; \
        tmp_i /= dims[0]; \
        \
        size_t start = 0; \
        size_t end = 0; \
        if (upper) { \
            start = max(k + (int64_t)row_idx, (int64_t)0); \
            end = dims[2]; \
        } else { \
            start = 0; \
            int64_t col_dim = (int64_t)dims[2]; \
            end = min(k + (int64_t)row_idx + 1, col_dim); \
        }\
        \
        if (col_idx >= start && col_idx < end) { \
            output[offset] = input[offset]; \
        } \
    } \
} \

#define TRILU_OP(TYPENAME, FORWARD) \
    LONG_TRILU_OP(TYPENAME, FORWARD)