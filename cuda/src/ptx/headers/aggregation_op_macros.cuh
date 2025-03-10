#define REDUCE_MEAN_OP_THREAD_PER_OUTPUT(TYPENAME, FORWARD, ACCUMULATOR_TYPE, INPUT_FUNC, DIVISION_TYPE, DIV_FUNC) \
extern "C" __global__ void FORWARD( \
    const size_t *axes,            /* The axes to reduce.                                 */\
    const size_t num_axes,         /* The number of axes to reduce (must be > 0).         */\
    const size_t rank,             /* The rank of input tensor (must be > 0).             */\
    const size_t *info,            /* The shape and stride of the input.                  */\
    const TYPENAME *input,         /* The input tensor data.                              */\
    const size_t reduced_dim_prod, /* The product of reduced dimensions (must be > 0).    */\
    const size_t num_out_elems,    /* Number of output elements (must be > 0).            */\
    TYPENAME *output               /* The output data.                                    */\
) {\
    bool reduced[32] = {0};\
    for (unsigned int i = 0; i < num_axes; i++) {\
        reduced[axes[i]] = 1;\
    }\
    \
    const size_t* shape = info;\
    const size_t* strides = info + rank;\
    for (unsigned int out_idx = blockIdx.x * blockDim.x + threadIdx.x; out_idx < num_out_elems; out_idx += blockDim.x * gridDim.x) { \
        size_t coord[32] = {0};\
        size_t tmp = out_idx;\
        for (int d = rank - 1; d >= 0; d--) {\
            if (!reduced[d]) {\
                coord[d] = tmp % shape[d];\
                tmp /= shape[d];\
            }\
        }\
        \
        ACCUMULATOR_TYPE acc = 0;\
        for (unsigned int k = 0; k < reduced_dim_prod; k++) {\
            size_t rem = k;\
            for (int d = rank - 1; d >= 0; d--) {\
                if (reduced[d]) {\
                    coord[d] = rem % shape[d];\
                    rem /= shape[d];\
                }\
            }\
            size_t flat_i = 0;\
            for (unsigned int d = 0; d < rank; d++) {\
                flat_i += coord[d] * strides[d];\
            }\
            TYPENAME x = input[flat_i];\
            acc +=  static_cast<ACCUMULATOR_TYPE>(INPUT_FUNC);\
        }\
        \
        DIVISION_TYPE div = static_cast<DIVISION_TYPE>(acc) / static_cast<DIVISION_TYPE>(reduced_dim_prod);\
        output[out_idx] = DIV_FUNC;\
    }\
}
