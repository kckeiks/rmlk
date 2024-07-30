fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=src/kernels/headers/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/kernels/headers/binary_op_macros.cuh");

    let builder = bindgen_cuda::Builder::default().kernel_paths(vec![
        "src/kernels/binary_add.cu",
        "src/kernels/binary_mul.cu",
    ]);
    println!("cargo:info={builder:?}");

    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/kernels/ptx.rs").unwrap();
}
