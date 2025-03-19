/**
    Input:
        data, indices, updates

     1. We derive the cuda thread index, thread_idx.
     2. We use mixed-radix decomposition to index into the tuple in indices, which we will call indices.idx_tuple
     3. We use mixed-radix decomposition to index into the updates input and read the value-slice to use in the output computation, lets call it updates.value_slice
     4. We use indices.idx_tuple to index in data and update it using value slice.
*/
#define SCATTER_ND_OP(TYPENAME, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,      /* The number of elements in the output.       */\
    const size_t rank,           /* The rank of input tensor (must be > 0).     */\
    const size_t *info,          /* The shape and stride of a and b.            */\
    const TYPENAME *data,        /* The input tensor data.                      */\
    const TYPENAME *indices,     /* The input tensor data.                      */\
    const TYPENAME *updates,     /* The input tensor data.                      */\
    TYPENAME *output             /* The output data.                            */\
) { \
    size_t data_shape = info;\
    size_t data_stride = info + rank;\
    size_t indices_shape = info + 2 * rank;\
    size_t indices_stride = info + 3 * rank;\
    size_t updates_shape = info + 4 * rank;\
    size_t updates_stride = info + 5 * rank;\
    for (unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < num_elems; thread_idx += blockDim.x * gridDim.x) {\
        int tmp = thread_idx;\
        size_t indices_i = 0;\
        for (int d = rank - 2; d >= 0; d--) {\
            size_t i_ = tmp % indices_shape[d];\
            indices_i += i_ * indices_stride[d];\
            tmp /= indices_shape[d];\
        }\
    }\
}

