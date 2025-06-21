#define LONG_EXPAND_OP(TYPENAME, FORWARD) \
extern "C" __global__ void FORWARD( \
    const size_t num_elems,         /* The number of elements in the output.          */\
    const size_t input_rank,        /* The rank of the input tensor (must be > 0).    */\
    const size_t output_rank,       /* The rank of the output tensor (must be > 0).   */\
    const size_t *info,             /* The shape and stride of input and output.      */\
    const TYPENAME *input,          /* The input data.                                */\
    TYPENAME *output                /* The output data.                               */\
) { \
    const size_t *input_shape = info; \
    const size_t *input_strides = info + input_rank; \
    const size_t *output_shape = info + 2 * input_rank; \
    const size_t *output_strides = info + 2 * input_rank + output_rank; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        size_t tmp_in = i; \
        size_t tmp_out = i; \
        size_t output_offset = 0; \
        size_t input_offset = 0; \
        for(int dim = output_rank - 1; dim >= 0; dim--) { \
            size_t out_idx = tmp_out % output_shape[dim]; \
            output_offset += out_idx * output_strides[dim]; \
            tmp_out /= output_shape[dim]; \
            \
            int input_dim = dim - (output_rank - input_rank);\
            if (input_dim >= 0 && input_shape[input_dim] != 1) {\
                input_offset += out_idx * input_strides[input_dim];\
            }\
        } \
        output[output_offset] = input[input_offset]; \
    } \
} \

#define EXPAND_OP(TYPENAME, FORWARD) \
    LONG_EXPAND_OP(TYPENAME, FORWARD)