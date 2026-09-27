use std::fmt::Write;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=src/ptx/headers/ternary_op_macro.cuh");
    println!("cargo:rerun-if-changed=src/ptx/headers/binary_op_macros.cuh");
    println!("cargo:rerun-if-changed=src/ptx/headers/unary_op_macros.cuh");

    let kernel_paths = [
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
    ];
    let builder = bindgen_cuda::Builder::default().kernel_paths(kernel_paths.to_vec());
    println!("cargo:info={builder:?}");

    builder.build_ptx().unwrap();

    let mut bindings = String::new();
    for kernel_path in kernel_paths {
        let kernel_name = Path::new(kernel_path)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();
        writeln!(
            bindings,
            "pub const {}: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/{kernel_name}.ptx\"));",
            kernel_name.to_uppercase()
        )
        .unwrap();
    }

    let output = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("output.rs");
    std::fs::write(output, bindings).unwrap();
}
