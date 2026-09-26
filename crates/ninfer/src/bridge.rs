//! The `cxx` bridge to the C++ adapter over ninfer's Engine.
//!
//! The types here are the boundary's wire shapes, in the primitives C++ reads;
//! [`crate::session`] wraps every one of them before it reaches a caller.
#![expect(
    clippy::inline_trait_bounds,
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
/// - unsafe invariants: each declaration matches its definition in
///   `cxx/adapter.h` and `cxx/adapter.cpp`, which `cxx` checks at compile time
///   through the generated header; every function declared below is `noexcept`,
///   and every internal helper that can throw is called only inside a catch-all
///   in one of them, so no exception crosses into Rust (an escaping one would
///   terminate the process, never unwind); `generate` touches the `ReviewSink`
///   only through a controller whose pointer to it is cleared under a mutex
///   before `generate` returns, so the Engine never reaches the sink after its
///   borrow ends; `Session` is used only through the `UniquePtr` `open_session`
///   returns, and only when that pointer is non-null.
#[cxx::bridge(namespace = "infinitum::ninfer")]
pub mod ffi
{
    /// How an adapter call ended.
    #[derive(Debug)]
    enum Status
    {
        /// The call succeeded.
        Completed,
        /// ninfer threw `std::invalid_argument`.
        InvalidArgument,
        /// ninfer threw another `std::exception`.
        Runtime,
        /// ninfer threw something that is not a `std::exception`.
        Unknown,
    }

    /// An adapter call's status and, on failure, ninfer's message.
    #[derive(Debug)]
    struct Outcome
    {
        /// How the call ended.
        status: Status,
        /// ninfer's message, empty on success.
        message: String,
    }

    /// CUDA graph capture, as the adapter reads it.
    #[derive(Debug)]
    enum CudaGraph
    {
        /// Launch every kernel directly.
        Off,
        /// Capture decode rounds.
        On,
    }

    /// The Engine options the adapter sets; every other option keeps ninfer's
    /// default.
    #[derive(Debug)]
    struct EngineConfig
    {
        /// The artifact entry file.
        artifact: String,
        /// The CUDA device ordinal.
        device: i32,
        /// The context ceiling, which also sizes the KV cache.
        max_context: u32,
        /// DFlash2's draft width `K`.
        draft_width: u32,
        /// CUDA graph capture.
        cuda_graph: CudaGraph,
    }

    /// Why a generation ended, as ninfer reports it.
    #[derive(Debug)]
    enum Finish
    {
        /// ninfer reported no reason.
        Unfinished,
        /// The output budget, or a round verdict's limit, was reached.
        OutputLimit,
        /// The context ceiling was reached.
        ContextCapacity,
        /// A stop token was generated.
        StopToken,
        /// A stop string was generated.
        StopString,
        /// The request was cancelled.
        Cancelled,
        /// A reason this adapter does not know.
        Unrecognized,
    }

    /// What one generation produced and how its speculation went.
    #[derive(Debug)]
    struct GenerationRecord
    {
        /// The generated ids.
        generated: Vec<i32>,
        /// Why it ended.
        finish: Finish,
        /// Speculative rounds run.
        decode_rounds: u64,
        /// Tokens drafted over all rounds.
        drafted: u64,
        /// Drafted tokens accepted.
        accepted: u64,
        /// Steps that fell back to plain decode.
        fallback_steps: u64,
        /// Wall time from the first to the last generated token, in
        /// nanoseconds.
        generation_wall_ns: u64,
    }

    /// Which kind of round an offer comes from.
    #[derive(Debug)]
    enum RoundKind
    {
        /// A decode round.
        Decode,
        /// Prefill finalization.
        PrefillFinalization,
    }

    /// The preview's decision, as the adapter reads it.
    #[derive(Debug)]
    enum Verdict
    {
        /// Leave the round to the output policy.
        Continue,
        /// Keep at most `limit` licensed tokens.
        Limit,
    }

    /// The preview's answer for one round.
    #[derive(Debug)]
    struct ReviewAnswer
    {
        /// The decision.
        verdict: Verdict,
        /// The limit under [`Verdict::Limit`], at least one; zero otherwise.
        limit: u32,
    }

    extern "Rust" {
        /// The Rust side of a round review.
        type ReviewSink;

        /// Review one round before its commit.
        fn review_round(
            sink: &mut ReviewSink,
            licensed: &[i32],
            kind: RoundKind,
        ) -> ReviewAnswer;
    }

    // SAFETY: the declarations below are sound to call from safe Rust under
    // the unsafe invariants stated in this module's `# Safety` section.
    unsafe extern "C++" {
        include!("infinitum-ninfer/cxx/adapter.h");

        /// One open Engine.
        type Session;

        /// Open an Engine; null with a failed outcome on failure.
        fn open_session(
            config: &EngineConfig,
            outcome: &mut Outcome,
        ) -> UniquePtr<Session>;

        /// Encode raw text into `ids`.
        fn tokenize(
            session: &Session,
            text: &str,
            ids: &mut Vec<i32>,
            outcome: &mut Outcome,
        );

        /// Generate greedily from `prompt` within `budget`, reviewing each
        /// round through `sink`.
        fn generate(
            session: Pin<&mut Session>,
            prompt: &[i32],
            budget: u32,
            sink: &mut ReviewSink,
            record: &mut GenerationRecord,
            outcome: &mut Outcome,
        );

        /// Render ids into `bytes`.
        fn detokenize(
            session: &Session,
            ids: &[i32],
            bytes: &mut Vec<u8>,
            outcome: &mut Outcome,
        );
    }
}

pub use crate::session::ReviewSink;
pub use crate::session::review_round;
