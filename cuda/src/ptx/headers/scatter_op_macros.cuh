/**
    Assumptions
    ===========
    - Rank of data must be >= 1.
    - Rank of indices must be >= 1.
    - Rank of updates must be = indices.rank + data.rank - indices.shape[-1] - 1.
    - The indices.shape[-1] <= data.rank.
    - The (1, ..., indices.rank - 1) dimensions of indices = (1, ..., indices.rank - 1) dimensions of updates.
    - The update.shape = indices.shape[0: indices.rank - 1] ++ data.shape[indices.shape[-1] : data.rank].
    - If indices.rank > 2, the parameter num_elems must be the product of the dimensions in
      indices.shape[0:indices.rank - 1]. Otherwise, it must be 1.
    - No negative indices.
    - The output = data.

    Note: if multiple entries in indices refer to the same slice in data,
    the final contents of that slice are unspecified.
*/
#define LONG_SCATTER_ND_OP(TYPENAME, FORWARD, FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,      /* The product of the dimensions in indices.shape[0: indices_rank - 1].              */\
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
            TYPENAME in = updates[updates_offset + i];\
            TYPENAME out = output[data_offset + i];\
            TYPENAME f;\
            FUNC\
            output[data_offset + i] = f;\
        }\
}\

#define SCATTER_ND_OP(TYPENAME, FORWARD, FUNC) \
    LONG_SCATTER_ND_OP(TYPENAME, FORWARD, f = (FUNC);)