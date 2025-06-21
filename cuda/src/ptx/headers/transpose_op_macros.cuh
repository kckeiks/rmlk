#define LONG_TRANSPOSE_OP(TYPENAME, FORWARD) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,      /* The number of elements in the input.         */ \
    const size_t rank,           /* Rank of the input.                           */ \
    const size_t *info,          /* The shape of dst and stride of src and dst.  */ \
    const size_t *perm,          /* Permute the axes according to these values.  */ \
    const TYPENAME *src_data,    /* The input tensor data.                       */ \
    TYPENAME *dst_data           /* The output tensor data.                      */ \
) { \
    const size_t *dims = info; \
    const size_t *src_strides = info + rank; \
    const size_t *dst_strides = info + 2 * rank; \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        size_t tmp_i = i; \
        size_t src_i = 0; \
        size_t dst_i = 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            int idx = tmp_i % dims[d]; \
            src_i += idx * src_strides[perm[d]]; \
            dst_i += idx * dst_strides[d]; \
            tmp_i /= dims[d]; \
        } \
        dst_data[dst_i] = src_data[src_i]; \
    } \
} \

#define TRANSPOSE_OP(TYPENAME, FORWARD) \
    LONG_TRANSPOSE_OP(TYPENAME, FORWARD)