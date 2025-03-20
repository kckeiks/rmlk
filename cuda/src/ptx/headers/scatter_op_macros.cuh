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
    const size_t num_elems,      /* The number of elements in the output.                                             */\
    const size_t data_rank,      /* The rank of data (must be > 0).                                                   */\
    const size_t indices_rank,   /* The rank of indices (must be > 0).                                                */\
    const size_t updates_rank,   /* The rank of updates (must be = indices.rank + data.rank - indices.shape[-1] - 1). */\
    const size_t *info,          /* The shape and stride of a and b.                                                  */\
    const TYPENAME *data,        /* The input tensor data.                                                            */\
    const TYPENAME *indices,     /* The input tensor data.                                                            */\
    const TYPENAME *updates,     /* The input tensor data.                                                            */\
    TYPENAME *output             /* The output data.                                                                  */\
) { \
    const size_t *data_shape = info;\
    const size_t *data_stride = info + data_rank;\
    const size_t *indices_shape = info + 2 * data_rank;\
    const size_t *indices_stride = info + 2 * data_rank + indices_rank;\
    const size_t *updates_shape = info + 2 * data_rank + 2 * indices_rank;\
    const size_t *updates_stride = info + 2 * data_rank + 2 * indices_rank + updates_rank;\
    for (unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < num_elems; thread_idx += blockDim.x * gridDim.x) {\
        size_t linear_idx = thread_idx;\
        size_t indices_offset = 0;\
        for (int d = indices_rank - 2; d >= 0; d--) {\
            size_t dim_idx = linear_idx % indices_shape[d];\
            indices_offset += dim_idx * indices_stride[d];\
            linear_idx /= indices_shape[d];\
        }\
        \
        /* This means each update entry specifies an update to a slice of the tensor. */\
        size_t data_offset = 0;\
        for(int i = 0; i < indices_shape[indices_rank - 1]; i++) {\
            data_offset += indices[indices_offset + i] * data_stride[i];\
        }\
        \
        linear_idx = thread_idx;\
        size_t updates_offset = 0;\
        for(int d = indices_rank - 2; d >= 0; d--) {\
            size_t dim_idx = linear_idx % updates_shape[d];\
            updates_offset += dim_idx * updates_stride[d];\
            linear_idx /= updates_shape[d];\
        }\
        \
        size_t slice_len = 1;\
        for (int d = indices_rank - 1; d < updates_rank; d++) {\
            slice_len *= updates_shape[d];\
        }\
        \
        for(int i = 0; i < slice_len; i++) {\
            output[data_offset + i] = updates[updates_offset + i];\
        }\
}
