fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=src/headers/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/headers/binary_op_macros.cuh");

    let builder = bindgen_cuda::Builder::default().kernel_paths(vec![
        "src/binary_add.cu",
        "src/binary_mul.cu"
    ]);
    println!("cargo:info={builder:?}");

    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs").unwrap();
}
