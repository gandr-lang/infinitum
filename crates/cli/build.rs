//! Gives the driver binary an rpath to ninfer's Engine library when the
//! `ninfer` feature links it.

/// Emit the rpath for the directory `infinitum-ninfer`'s build reported.
///
/// # Specification
/// - requires: nothing.
/// - ensures: when `DEP_NINFER_ENGINE_LIB` is set, which happens exactly when
///   the `ninfer` feature builds the bridge, the binary is linked with an rpath
///   to that directory; otherwise nothing is emitted.
/// - provides: an `infinitum` binary that finds `libninfer_engine.so` without
///   `LD_LIBRARY_PATH`.
/// - fails: never.
/// - panics: none.
fn main()
{
    println!("cargo::rerun-if-env-changed=DEP_NINFER_ENGINE_LIB");
    if let Ok(lib) = std::env::var("DEP_NINFER_ENGINE_LIB") {
        println!("cargo::rustc-link-arg-bins=-Wl,-rpath,{lib}");
    }
}
