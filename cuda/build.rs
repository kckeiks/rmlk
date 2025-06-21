fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=src/ptx/headers/ternary_op_macro.cuh");
    println!("cargo:rerun-if-changed=src/ptx/headers/binary_op_macros.cuh");
    println!("cargo:rerun-if-changed=src/ptx/headers/unary_op_macros.cuh");

    let builder = bindgen_cuda::Builder::default().kernel_paths(vec![
        "src/ptx/add.cu",
        "src/ptx/cast.cu",
        "src/ptx/cos.cu",
        "src/ptx/div.cu",
        "src/ptx/equal.cu",
        "src/ptx/expand.cu",
        "src/ptx/greater.cu",
        "src/ptx/mul.cu",
        "src/ptx/neg.cu",
        "src/ptx/reduce_mean.cu",
        "src/ptx/pow.cu",
        "src/ptx/transpose.cu",
        "src/ptx/trilu.cu",
        "src/ptx/slice.cu",
        "src/ptx/scatter_nd.cu",
        "src/ptx/sin.cu",
        "src/ptx/sqrt.cu",
        "src/ptx/sub.cu",
        "src/ptx/where.cu",
    ]);
    println!("cargo:info={builder:?}");

    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/ptx/output.rs").unwrap();
}
