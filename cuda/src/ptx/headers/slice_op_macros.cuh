#define SLICE_OP(TYPENAME, TYPENAME_INDEX, FORWARD) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,          /* The number of elements in the output.                               */\
    const size_t axes_len,           /* The number of elements in the axes input.                           */\
    const size_t rank,               /* The rank of input tensor (must be > 0).                             */\
    const size_t *info,              /* The shape and stride of input and output.                           */\
    const TYPENAME *input,           /* The input tensor data.                                              */\
    const TYPENAME_INDEX *starts,    /* The starting indices.                                               */\
    const TYPENAME_INDEX *ends,      /* The ending tensor data.                                             */\
    const TYPENAME_INDEX *axes,      /* The axes that `starts` and `ends` apply to.                         */\
    const TYPENAME_INDEX *steps,     /* The slice steps corresponding `axes`. It must not contain 0 steps.  */\
    TYPENAME *output                 /* The output data.                                                    */\
) { \
    size_t thread_idx =  blockIdx.x * blockDim.x + threadIdx.x; \
    if (thread_idx >= num_elems) { \
        return; \
    } \
    \
    const size_t *input_shape = info; \
    const size_t *input_stride = info + rank; \
    const size_t *output_shape = info + 2 * rank; \
    const size_t *output_stride = info + 3 * rank; \
    \
    TYPENAME_INDEX starts_buf[8]; \
    TYPENAME_INDEX ends_buf[8]; \
    TYPENAME_INDEX steps_buf[8]; \
    for (int d = rank - 1; d >= 0; d--) { \
        starts_buf[d] = 0; \
        ends_buf[d] =  static_cast<TYPENAME_INDEX>(input_shape[d]); \
        steps_buf[d] = 1; \
    } \
    \
    size_t indices_len = axes_len; \
    if (indices_len == 0) { \
        indices_len = rank; \
    } \
    \
    for(int k = 0; k < indices_len; k++) { \
        int axis = k; \
        if (axes_len > 0 && axes[k] < 0) { \
            axis = axes[k]  + rank; \
        } else if (axes_len > 0) { \
            axis = axes[k]; \
        }\
        \
        steps_buf[axis] = steps ? steps[k]: steps_buf[axis]; \
        \
        TYPENAME_INDEX dim = static_cast<TYPENAME_INDEX>(input_shape[axis]); \
        TYPENAME_INDEX start = starts[k]; \
        if (start < 0) { \
            start = dim + start; \
        } \
        if (start < 0) { \
            start = 0; \
        } \
        if (steps_buf[axis] >= 0 && start > dim) { \
            start = dim; \
        } else if (steps_buf[axis] < 0 && start >= dim) { \
            start = dim - 1; \
        } \
        starts_buf[axis] = start; \
        TYPENAME_INDEX end = ends[k]; \
        if (end < 0) { \
            end += dim; \
        } \
        if (end < -1) { \
            end = -1; \
        } \
        if (steps_buf[axis] >= 0 && end > dim) { \
            end = dim; \
        } else if (steps_buf[axis] < 0 && end >= dim) { \
            end = dim - 1; \
        } \
        ends_buf[axis] = end; \
    }\
    \
    for (size_t i = thread_idx; i < num_elems; i += blockDim.x * gridDim.x) { \
        bool done = true; \
        size_t tmp_i = i; \
        size_t in_offset = 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            TYPENAME_INDEX out_idx = static_cast<TYPENAME_INDEX>(tmp_i % output_shape[d]); \
            TYPENAME_INDEX in_idx = (steps_buf[d] * out_idx) + starts_buf[d]; \
            /* Check if it's passed the end. */ \
            if ((steps_buf[d] > 0 && in_idx >= ends_buf[d]) || (steps_buf[d] < 0 && in_idx < ends_buf[d])) { \
                done = false; \
                break; \
            } \
            in_offset += static_cast<size_t>(in_idx) * input_stride[d]; \
            tmp_i /= output_shape[d]; \
        }\
        if (!done) { \
            continue; \
        } \
        output[i] = input[in_offset]; \
    } \
} \
