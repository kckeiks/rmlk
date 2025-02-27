fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=src/ptx/headers/ternary_op_macro.cuh");
    println!("cargo:rerun-if-changed=src/ptx/headers/binary_op_macros.cuh");
    println!("cargo:rerun-if-changed=src/ptx/headers/unary_op_macros.cuh");

    let builder = bindgen_cuda::Builder::default().kernel_paths(vec![
        "src/ptx/add.cu",
        "src/ptx/cast.cu",
        "src/ptx/mul.cu",
        "src/ptx/div.cu",
        "src/ptx/sqrt.cu",
        "src/ptx/where.cu",
    ]);
    println!("cargo:info={builder:?}");

    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/ptx/output.rs").unwrap();
}
