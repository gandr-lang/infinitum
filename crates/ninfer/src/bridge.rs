//! The `cxx` bridge to the C++ adapter over ninfer's Engine.
//!
//! The types here are the boundary's wire shapes, in the primitives C++ reads;
//! [`crate::session`] wraps every one of them before it reaches a caller.
#![expect(
    clippy::inline_trait_bounds,
    clippy::multiple_unsafe_ops_per_block,
    clippy::renamed_function_params,
    clippy::semicolon_outside_block,
    clippy::too_many_arguments,
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
///   terminate the process, never unwind); `generate` and `run_chat` touch the
///   `ReviewSink` only through a controller whose pointer to it is cleared
///   under a mutex before they return, so the Engine never reaches the sink
///   after its borrow ends; `run_chat` reaches the `ChatSink` only from the
///   thread that called it, through an output sink that lives on its stack and
///   is gone when it returns, and reaches the `CancelFlag`, an atomic flag
///   readable from any thread, only through cancellation views that end with
///   the calls they are passed to; `Session` is used only through the
///   `UniquePtr` `open_session` returns, and only when that pointer is
///   non-null.
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
        /// ninfer refused the request with a classified `RequestError`.
        Refused,
    }

    /// Why ninfer refused a request, when [`Status::Refused`].
    #[derive(Debug)]
    enum Refusal
    {
        /// Not refused.
        None,
        /// The prompt exceeds the context ceiling.
        ContextLength,
        /// The thinking budget leaves no room for its closing control
        /// tokens.
        ThinkingBudgetCapacity,
        /// Media exceeded its budget or was invalid.
        Media,
        /// The queue is full.
        Overloaded,
        /// The request expired while pending.
        QueueTimeout,
        /// The request was cancelled before admission.
        Cancelled,
        /// The Engine is not serving.
        Unavailable,
    }

    /// An adapter call's status and, on failure, ninfer's message.
    #[derive(Debug)]
    struct Outcome
    {
        /// How the call ended.
        status: Status,
        /// The refusal's class under [`Status::Refused`]; `None` otherwise.
        refusal: Refusal,
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

    /// How the adapter sizes the Main KV cache.
    #[derive(Debug)]
    enum KvSizing
    {
        /// `kv_capacity` tokens.
        Explicit,
        /// From device memory, with ninfer's default headroom.
        Automatic,
    }

    /// The KV storage, as the adapter reads it.
    #[derive(Debug)]
    enum KvStorage
    {
        /// bfloat16.
        BFloat16,
        /// 8-bit integers in groups of 64.
        Int8,
        /// FP8 E4M3 rows of 256.
        Fp8,
        /// NVFP4 groups of 16.
        Nvfp4,
        /// FP8 keys, NVFP4 values.
        Fp8KeyNvfp4Value,
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
        /// The context ceiling.
        max_context: u32,
        /// How the Main KV cache is sized.
        kv_sizing: KvSizing,
        /// The Main KV capacity in tokens when sized explicitly; zero when
        /// automatic.
        kv_capacity: u32,
        /// The KV storage.
        kv_storage: KvStorage,
        /// Prompt tokens per prefill step.
        prefill_chunk: u32,
        /// Requests the Engine runs at once.
        max_concurrency: u32,
        /// Whether `device_state_slots` is set; unset, the Engine keeps one
        /// per lane.
        device_state_set: bool,
        /// Device checkpoint slots beyond the lanes.
        device_state_slots: u32,
        /// Host checkpoint slots.
        host_state_slots: u32,
        /// Host KV capacity in bytes.
        host_kv_bytes: usize,
        /// How long a request may wait for admission, in milliseconds.
        pending_timeout_ms: u32,
        /// DFlash2's draft width `K`.
        draft_width: u32,
        /// CUDA graph capture.
        cuda_graph: CudaGraph,
        /// A chat template overriding the artifact's; empty keeps the
        /// artifact's own.
        chat_template: String,
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

    /// Who wrote a turn.
    #[derive(Debug)]
    enum ChatRole
    {
        /// A system instruction.
        System,
        /// A developer instruction.
        Developer,
        /// The user.
        User,
        /// The model.
        Assistant,
        /// A tool's result.
        Tool,
    }

    /// One tool call of an assistant turn.
    #[derive(Debug)]
    struct ChatToolCall
    {
        /// The call's wire identity.
        id: String,
        /// The function's name.
        name: String,
        /// The arguments' JSON text.
        arguments: String,
    }

    /// One turn.
    #[derive(Debug)]
    struct ChatTurn
    {
        /// Who wrote it.
        role: ChatRole,
        /// Its text parts, in order.
        parts: Vec<String>,
        /// Carried reasoning.
        reasoning: String,
        /// An assistant turn's calls.
        tool_calls: Vec<ChatToolCall>,
        /// The call a tool turn answers.
        tool_call_id: String,
    }

    /// A chat-template switch.
    #[derive(Debug)]
    enum TemplateSwitch
    {
        /// Left to the template.
        Unset,
        /// On.
        On,
        /// Off.
        Off,
    }

    /// A requested reasoning effort.
    #[derive(Debug)]
    enum EffortLevel
    {
        /// None requested.
        Unset,
        /// No reasoning.
        None,
        /// Minimal.
        Minimal,
        /// Low.
        Low,
        /// Medium.
        Medium,
        /// High.
        High,
        /// Extra high.
        XHigh,
        /// Maximum.
        Max,
    }

    /// Where a prompt-cache marker sits.
    #[derive(Debug)]
    enum MarkLocation
    {
        /// After `count` bytes of the leading instruction's text.
        LeadingInstruction,
        /// After `parts` parts of the `count`-th message.
        MessagePart,
        /// After `count` messages.
        Message,
        /// After `count` tool definitions.
        Tool,
    }

    /// One prompt-cache marker, a shared stable prefix.
    #[derive(Debug)]
    struct CacheMark
    {
        /// Where it sits.
        location: MarkLocation,
        /// Bytes, messages or tools, as `location` reads it.
        count: u32,
        /// Parts, for a message-part boundary; zero otherwise.
        parts: u32,
        /// ninfer's evidence bits: 1 explicit, 2 requested automatic, 4
        /// default automatic.
        evidence: u8,
    }

    /// What the prompt renders from.
    #[derive(Debug)]
    struct ChatPrompt
    {
        /// The conversation.
        turns: Vec<ChatTurn>,
        /// Tool definitions as JSON object text.
        tools: Vec<String>,
        /// Extra template arguments as JSON object text; empty for none.
        template_arguments: String,
        /// Thinking on the new turn.
        thinking: TemplateSwitch,
        /// Whether closed turns keep their reasoning.
        preserve_thinking: TemplateSwitch,
        /// The requested effort.
        effort: EffortLevel,
        /// The prompt-cache markers, in order.
        cache_marks: Vec<CacheMark>,
        /// Whether the Engine may add structural shared prefixes.
        structural_prefixes: bool,
    }

    /// Sampling overrides for one phase; each value applies only when its
    /// flag is set.
    #[derive(Debug)]
    struct SamplingFields
    {
        /// Temperature.
        temperature: f32,
        /// Whether `temperature` is set.
        temperature_set: bool,
        /// Top-k.
        top_k: i32,
        /// Whether `top_k` is set.
        top_k_set: bool,
        /// Top-p.
        top_p: f32,
        /// Whether `top_p` is set.
        top_p_set: bool,
        /// Min-p.
        min_p: f32,
        /// Whether `min_p` is set.
        min_p_set: bool,
        /// Presence penalty.
        presence_penalty: f32,
        /// Whether `presence_penalty` is set.
        presence_penalty_set: bool,
        /// Frequency penalty.
        frequency_penalty: f32,
        /// Whether `frequency_penalty` is set.
        frequency_penalty_set: bool,
    }

    /// How to generate.
    #[derive(Debug)]
    struct ChatSettings
    {
        /// The output limit.
        output_tokens: u32,
        /// Initial-phase overrides.
        sampling: SamplingFields,
        /// Initial-phase seed.
        seed: u64,
        /// Post-thinking overrides.
        post_thinking: SamplingFields,
        /// Post-thinking seed, when `post_thinking_seed_set`.
        post_thinking_seed: u64,
        /// Whether the post-thinking seed is its own rather than inherited.
        post_thinking_seed_set: bool,
        /// The thinking budget; zero is unlimited.
        thinking_budget: u32,
        /// Stop strings.
        stops: Vec<String>,
        /// Whether stop strings also end reasoning.
        stops_end_reasoning: bool,
        /// Whether special tokens survive into the output.
        preserve_special_tokens: bool,
        /// The longest function name tool-call parsing accepts.
        tool_name_limit: u32,
        /// Whether the prefix cache is read and written.
        prefix_reuse: bool,
        /// Whether deltas are published as they commit.
        streaming: bool,
    }

    /// One generated tool call.
    #[derive(Debug)]
    struct ChatGeneratedCall
    {
        /// The function's name.
        name: String,
        /// The arguments' JSON text.
        arguments: String,
    }

    /// What one chat request produced.
    #[derive(Debug)]
    struct ChatRecord
    {
        /// The answer.
        content: String,
        /// The thinking.
        reasoning: String,
        /// Parsed tool calls.
        tool_calls: Vec<ChatGeneratedCall>,
        /// Why it ended.
        finish: Finish,
        /// Prompt tokens.
        prompt_tokens: u32,
        /// Prompt tokens reused from a cached prefix.
        reused_tokens: u32,
        /// Every generated id.
        generated: Vec<i32>,
        /// Generated tokens in thinking.
        reasoning_tokens: u32,
        /// Admission to first output token, in nanoseconds.
        prompt_wall_ns: u64,
        /// First to last output token, in nanoseconds.
        generation_wall_ns: u64,
        /// Tokens drafted.
        drafted: u64,
        /// Drafted tokens accepted.
        accepted: u64,
    }

    /// The channel a published delta belongs to.
    #[derive(Debug)]
    enum DeltaChannel
    {
        /// The answer.
        Content,
        /// Thinking.
        Reasoning,
    }

    extern "Rust" {
        /// The Rust side of a round review.
        type ReviewSink;

        /// The Rust side of a chat request's streamed output.
        type ChatSink<'events>;

        /// A chat request's cancellation flag.
        type CancelFlag;

        /// Review one round before its commit.
        fn review_round(
            sink: &mut ReviewSink,
            licensed: &[i32],
            kind: RoundKind,
        ) -> ReviewAnswer;

        /// The request passed preparation and was submitted.
        fn chat_submitted(sink: &mut ChatSink<'_>);

        /// The request was admitted.
        fn chat_admitted(
            sink: &mut ChatSink<'_>,
            prompt_tokens: u32,
            reused_tokens: u32,
        );

        /// Committed text on one channel, as bytes the Rust side validates.
        fn chat_publish(
            sink: &mut ChatSink<'_>,
            channel: DeltaChannel,
            text: &[u8],
        );

        /// Whether the consumer asked to stop.
        fn chat_cancelled(flag: &CancelFlag) -> bool;
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

        /// The model name the artifact records.
        fn model_name(
            session: &Session,
            name: &mut String,
            outcome: &mut Outcome,
        );

        /// Run one chat request, publishing to `sink` when streaming, polling
        /// `cancel`, and reviewing each round through `review`.
        fn run_chat(
            session: &Session,
            prompt: &ChatPrompt,
            settings: &ChatSettings,
            sink: &mut ChatSink<'_>,
            cancel: &CancelFlag,
            review: &mut ReviewSink,
            record: &mut ChatRecord,
            outcome: &mut Outcome,
        );
    }
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the one field is `cxx`'s opaque marker, which is `!Send` only because `cxx` cannot see the C++ type; the Engine behind it is thread-safe, as the safety comment below states"
)]
// SAFETY: the adapter's `Session` holds one ninfer Engine and nothing else, and
// the Engine synchronizes every public operation internally: its request queue
// admits concurrent `prepare` and `submit` calls and its worker thread owns the
// device state, which is how ninfer's own server shares one Engine across its
// request threads. Moving it to another thread moves only that owner.
unsafe impl Send for ffi::Session
{
}

// SAFETY: as for `Send`; the functions that take `&Session` call only Engine
// operations that ninfer's server calls concurrently from request threads.
unsafe impl Sync for ffi::Session
{
}

pub use crate::chat::CancelFlag;
pub use crate::chat::ChatSink;
pub use crate::chat::chat_admitted;
pub use crate::chat::chat_cancelled;
pub use crate::chat::chat_publish;
pub use crate::chat::chat_submitted;
pub use crate::session::ReviewSink;
pub use crate::session::review_round;
