mod common;

mod cuda {
    mod add_three_inputs;
    mod add_two_inputs;
    mod add_with_constants;

    mod add_broadcast;
    mod add_broadcast_diff_len_shapes;
    mod constant_of_shape_simple;
    mod constant_of_shape_with_attrs;
    mod gather_higher_dim;
    mod gather_higher_dim_indices;
    mod gather_negative_axis;
    mod gather_simple;
    mod gather_with_axis;
    mod gemm_simple;
    mod reduce_mean_simple;
    mod shape_axis;
    mod shape_simple;
    mod transpose_validate_data;
    mod transpose_validate_shape;
    mod unsqueeze_start;
    mod unsqueeze_mid;
    mod unsqueeze_end;
    mod unsqueeze_multiple;
    mod unsqueeze_negative_axes;
    mod unsqueeze_any_order;
    mod where_broadcast_diff_shape_len;
    mod where_two_inputs;
}
