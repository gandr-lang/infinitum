//! Builds the `cxx` bridge and links ninfer's Engine when `engine` is on.
//!
//! Without `engine` the crate is the planner alone and this script does
//! nothing.

/// Compile the bridge and the adapter, and link `libninfer_engine.so`.
///
/// # Specification
/// - requires: with `engine`, `INFINITUM_NINFER_INCLUDE` names ninfer's public
///   header directory and `INFINITUM_NINFER_LIB` the directory holding its
///   shared Engine library; a C++26 compiler is on the path.
/// - ensures: on success the bridge links against that library, this package's
///   own test binaries find it at run time through an rpath, and dependents'
///   build scripts read the directory as `DEP_NINFER_ENGINE_LIB`.
/// - provides: the one place the build learns where ninfer lives.
/// - fails: when either variable is unset.
/// - panics: none; `cxx-build` itself reports a compile failure by panicking.
///
/// # Errors
/// - [`std::env::VarError`]: a variable is unset or not Unicode.
#[cfg(feature = "engine")]
fn main() -> Result<(), std::env::VarError>
{
    let include = std::env::var("INFINITUM_NINFER_INCLUDE")?;
    let lib = std::env::var("INFINITUM_NINFER_LIB")?;
    cxx_build::bridge("src/bridge.rs")
        .file("cxx/adapter.cpp")
        .include(&include)
        .std("c++26")
        .compile("infinitum-ninfer-bridge");
    println!("cargo::rerun-if-changed=src/bridge.rs");
    println!("cargo::rerun-if-changed=cxx/adapter.cpp");
    println!("cargo::rerun-if-changed=cxx/adapter.h");
    println!("cargo::rerun-if-env-changed=INFINITUM_NINFER_INCLUDE");
    println!("cargo::rerun-if-env-changed=INFINITUM_NINFER_LIB");
    println!("cargo::rustc-link-search=native={lib}");
    println!("cargo::rustc-link-lib=dylib=ninfer_engine");
    println!("cargo::rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo::metadata=lib={lib}");
    return Ok(());
}

/// Nothing to build without `engine`.
///
/// # Specification
/// trivial.
#[cfg(not(feature = "engine"))]
fn main()
{
}
