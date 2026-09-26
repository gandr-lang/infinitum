//! The `cxx` bridge to the C++ host over tt-mlir.
//!
//! The types here are the boundary's wire shapes, in the primitives C++ reads;
//! [`crate::device`] wraps every one of them before it reaches a caller.
#![expect(
    clippy::multiple_unsafe_ops_per_block,
    clippy::renamed_function_params,
    clippy::semicolon_outside_block,
    reason = "the lints fire on the code `cxx::bridge` generates, which this crate does not write"
)]

/// The declarations `cxx` checks on both sides.
///
/// # Safety
/// The `unsafe extern "C++"` block below asserts that every declared C++
/// function is safe to call from safe Rust with any arguments its Rust
/// signature admits.
/// - unsafe invariants: each declaration matches its definition in `cxx/host.h`
///   and `cxx/host.cpp`, which `cxx` checks at compile time through the
///   generated header; every function declared below is `noexcept` and catches
///   every exception at its call into MLIR or the runtime, so no exception
///   crosses into Rust; `Program` and `Device` are used only through the
///   `UniquePtr`s the host returns, and only when non-null; `run_accept` reads
///   `logits`, `planes`, and `tail` only for the call.
#[cxx::bridge(namespace = "infinitum::tenstorrent")]
pub mod ffi
{
    /// How a host call ended.
    #[derive(Debug)]
    enum Status
    {
        /// The call succeeded.
        Completed,
        /// The arguments were refused before any MLIR or runtime call.
        InvalidArgument,
        /// MLIR reported diagnostics: verification, a pass, or translation.
        Diagnostics,
        /// The runtime threw a `std::exception`.
        Runtime,
        /// Something that is not a `std::exception` was thrown.
        Unknown,
    }

    /// A host call's status and, on failure, the message.
    #[derive(Debug)]
    struct Outcome
    {
        /// How the call ended.
        status: Status,
        /// The message, empty on success.
        message: String,
    }

    /// Where the accepted-length prefix is computed.
    #[derive(Debug)]
    enum PrefixSite
    {
        /// On the device, in the same program as the argmax.
        Device,
        /// On the host, from the target ids the device returns.
        Host,
    }

    /// The shapes the Accept module is lowered for.
    #[derive(Debug)]
    struct AcceptGeometry
    {
        /// The draft width `K`.
        drafts: u32,
        /// The valid vocabulary size.
        vocabulary: u32,
        /// Logits per device row.
        chunk: u32,
        /// Where the prefix runs.
        prefix: PrefixSite,
    }

    /// What an in-process lowering cost and produced.
    #[derive(Debug, Default)]
    struct CompileRecord
    {
        /// Building and verifying the module.
        build_ns: u64,
        /// The TTIR-to-TTMetal pipeline.
        pipeline_ns: u64,
        /// The flatbuffer translation.
        translate_ns: u64,
        /// Device programs the lowered module enqueues.
        programs: u32,
    }

    /// What one run returned and cost.
    #[derive(Debug, Default)]
    struct RunOutput
    {
        /// The program's outputs, flattened in order, as bfloat16 bits.
        values: Vec<u16>,
        /// Submission to completion.
        submit_ns: u64,
        /// Reading the outputs back to the host.
        readback_ns: u64,
    }

    // SAFETY: the declarations below are sound to call from safe Rust under
    // the unsafe invariants stated in this module's `# Safety` section.
    unsafe extern "C++" {
        include!("infinitum-tenstorrent/cxx/host.h");

        /// A lowered Accept fragment.
        type Program;

        /// An open device.
        type Device;

        /// Print the builder's Accept module as TTIR.
        fn print_accept(
            geometry: &AcceptGeometry,
            text: &mut String,
            outcome: &mut Outcome,
        );

        /// Build, lower, and translate the Accept module in process; an
        /// empty `system_desc` lowers against the mock Blackhole descriptor.
        fn compile_accept(
            geometry: &AcceptGeometry,
            system_desc: &str,
            record: &mut CompileRecord,
            outcome: &mut Outcome,
        ) -> UniquePtr<Program>;

        /// Load a flatbuffer the pipeline tools wrote.
        fn load_program(
            path: &str,
            outcome: &mut Outcome,
        ) -> UniquePtr<Program>;

        /// Open the visible device for `program`'s runtime.
        fn open_device(
            program: &Program,
            outcome: &mut Outcome,
        ) -> UniquePtr<Device>;

        /// Write the current system descriptor.
        fn save_system_desc(
            path: &str,
            outcome: &mut Outcome,
        );

        /// Run `program` once over the logits, the constant planes, and
        /// the per-call tail.
        fn run_accept(
            device: Pin<&mut Device>,
            program: &Program,
            logits: &[u16],
            planes: &[u16],
            tail: &[u16],
            output: &mut RunOutput,
            outcome: &mut Outcome,
        );
    }
}
