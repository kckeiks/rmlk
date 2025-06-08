#define LONG_EXPAND_OP(TYPENAME, FORWARD) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,   /* The number of elements in the output.       */\
    const size_t rank,        /* The rank of the tensors (must be > 0).      */\
    const size_t *info,       /* The shape and stride of input and output.   */\
    const TYPENAME *input,    /* The input data.                             */\
    TYPENAME *output          /* The output data.                            */\
) { \
    const size_t *input_shape = info; \
    const size_t *input_strides = info + 1 * rank; \
    const size_t *output_shape = info + 2 * rank; \
    const size_t *output_strides = info + 3 * rank; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        size_t tmp_out = i; \
        size_t output_offset = 0; \
        size_t input_offset = 0; \
        for(int dim = rank - 1; dim >= 0; dim--) { \
            size_t out_idx = tmp_out % output_shape[dim]; \
            output_offset += out_idx * output_strides[dim]; \
            tmp_out /= output_shape[dim]; \
            \
            if (input_shape[dim] != 1) { \
                input_offset += out_idx * input_strides[dim]; \
            } \
        } \
        output[output_offset] = input[input_offset]; \
    } \
} \

#define EXPAND_OP(TYPENAME, FORWARD) \
    LONG_EXPAND_OP(TYPENAME, FORWARD)