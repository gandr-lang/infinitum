//! Builds the `cxx` bridge and links tt-mlir when `device` is on.
//!
//! Without `device` the crate is the planner and the reference alone and this
//! script does nothing.

/// Compile the bridge and the host, and link tt-mlir's compiler and runtime.
///
/// # Specification
/// - requires: with `device`, `INFINITUM_TTMLIR_SOURCE` names a tt-mlir
///   checkout, `INFINITUM_TTMLIR_BUILD` its build directory with the runtime
///   enabled, and `INFINITUM_TTMLIR_TOOLCHAIN` the LLVM/MLIR installation it
///   was built against; a C++26 compiler is on the path.
/// - ensures: on success the bridge links against `libTTMLIRCompiler.so` and
///   `libTTMLIRRuntime.so`, and this package's binaries find both, and
///   tt-metal's libraries, at run time through an rpath.
/// - provides: the one place the build learns where tt-mlir lives.
/// - fails: when a variable is unset.
/// - panics: none; `cxx-build` itself reports a compile failure by panicking.
///
/// # Errors
/// - [`std::env::VarError`]: a variable is unset or not Unicode.
#[cfg(feature = "device")]
fn main() -> Result<(), std::env::VarError>
{
    let source = std::env::var("INFINITUM_TTMLIR_SOURCE")?;
    let build = std::env::var("INFINITUM_TTMLIR_BUILD")?;
    let toolchain = std::env::var("INFINITUM_TTMLIR_TOOLCHAIN")?;
    cxx_build::bridge("src/bridge.rs")
        .file("cxx/host.cpp")
        .include(format!("{source}/include"))
        .include(format!("{source}/runtime/include"))
        .include(format!("{build}/include"))
        .include(format!("{build}/runtime/include"))
        .include(format!("{toolchain}/include"))
        .std("c++26")
        // LLVM and MLIR are built without RTTI; a TU with it references typeinfo they never emit.
        .flag("-fno-rtti")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("infinitum-tenstorrent-host");
    println!("cargo::rerun-if-changed=src/bridge.rs");
    println!("cargo::rerun-if-changed=cxx/host.cpp");
    println!("cargo::rerun-if-changed=cxx/host.h");
    println!("cargo::rerun-if-env-changed=INFINITUM_TTMLIR_SOURCE");
    println!("cargo::rerun-if-env-changed=INFINITUM_TTMLIR_BUILD");
    println!("cargo::rerun-if-env-changed=INFINITUM_TTMLIR_TOOLCHAIN");
    let libraries = [
        format!("{build}/lib"),
        format!("{build}/runtime/lib"),
        format!("{source}/third_party/tt-metal/src/tt-metal/build_Release/lib"),
    ];
    for directory in &libraries {
        println!("cargo::rustc-link-search=native={directory}");
        println!("cargo::rustc-link-arg=-Wl,-rpath,{directory}");
    }
    println!("cargo::rustc-link-lib=dylib=TTMLIRCompiler");
    println!("cargo::rustc-link-lib=dylib=TTMLIRRuntime");
    return Ok(());
}

/// Nothing to build without `device`.
///
/// # Specification
/// trivial.
#[cfg(not(feature = "device"))]
fn main()
{
}
