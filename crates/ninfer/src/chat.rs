//! Chat requests on ninfer's Engine: infinitum's serving contract over the
//! bridge.
//!
//! [`ChatEngine`] implements [`ChatBackend`]. ninfer's Qwen frontend renders
//! the chat template, parses reasoning and tool calls, and runs admission and
//! the prefix cache; infinitum supplies the request, receives the committed
//! output, and reviews every speculative round through its
//! [`infinitum_round::Preview`] before the commit.

use infinitum_chat::Admission;
use infinitum_chat::Automatic;
use infinitum_chat::ByteSize;
use infinitum_chat::CacheBoundary;
use infinitum_chat::CacheMarker;
use infinitum_chat::CancelToken;
use infinitum_chat::Cancellation;
use infinitum_chat::Capacity;
use infinitum_chat::Channel;
use infinitum_chat::ChatBackend;
use infinitum_chat::ChatEvents;
use infinitum_chat::ChatFailure;
use infinitum_chat::ChatOutcome;
use infinitum_chat::ChatRequest;
use infinitum_chat::ContextCache;
use infinitum_chat::CounterDigest;
use infinitum_chat::Delivery;
use infinitum_chat::DeltaText;
use infinitum_chat::Effort;
use infinitum_chat::FailureKind;
use infinitum_chat::Finish;
use infinitum_chat::GeneratedToolCall;
use infinitum_chat::KvSizing;
use infinitum_chat::Marked;
use infinitum_chat::ModelName;
use infinitum_chat::PrefixReuse;
use infinitum_chat::RequestGauges;
use infinitum_chat::ReusePath;
use infinitum_chat::Role;
use infinitum_chat::RuntimeCounters;
use infinitum_chat::Sampling;
use infinitum_chat::Seed;
use infinitum_chat::Setting;
use infinitum_chat::SpecialTokens;
use infinitum_chat::StopScope;
use infinitum_chat::StructuralPrefixes;
use infinitum_chat::Submission;
use infinitum_chat::Switch;
use infinitum_chat::Tally;
use infinitum_chat::Telemetry;
use infinitum_chat::Thinking;
use infinitum_chat::ThinkingBudget;
use infinitum_chat::ThinkingSpend;
use infinitum_round::Maybe;
use infinitum_round::TokenCount;
use infinitum_round::TokenId;

use crate::bridge::ffi;
use crate::options::EngineOptions;
use crate::plan::DFlash2Plan;
use crate::session::EngineFailure;
use crate::session::Operation;
use crate::session::ReviewSink;
use crate::session::Session;
use crate::session::check;
use crate::session::pending;

/// The Rust side of a chat request's streamed output.
pub struct ChatSink<'events>
{
    /// The consumer.
    events: &'events mut dyn ChatEvents,
}

/// A chat request's cancellation flag, as the adapter polls it.
#[repr(transparent)]
pub struct CancelFlag(CancelToken);

/// Tell the consumer the request was submitted, with what preparation
/// decided.
///
/// # Specification
/// - requires: called once, after the request's submission and before
///   [`chat_admitted`].
/// - ensures: the consumer was told whether the turn opens in thinking and
///   under which budget, zero read as unlimited.
/// - provides: the bridge's entry for the point a streamed response begins and
///   the request's start record.
/// - fails: never.
/// - panics: none.
// The bridge fixes this signature; the flag and the count are wrapped on the
// first line.
pub fn chat_submitted(
    sink: &mut ChatSink<'_>,
    thinking_open: bool,
    thinking_budget: u32,
)
{
    sink.events.submitted(Submission {
        thinking: if thinking_open {
            Thinking::Open
        }
        else {
            Thinking::Closed
        },
        thinking_budget: core::num::NonZeroU32::new(thinking_budget)
            .map_or(ThinkingBudget::Unlimited, ThinkingBudget::Tokens),
    });
}

/// Forward the admission record to the consumer.
///
/// # Specification
/// - requires: called once, before any [`chat_publish`].
/// - ensures: the consumer received the counts.
/// - provides: the bridge's entry for `OutputSink::start`.
/// - fails: never.
/// - panics: none.
// The `cxx` border: a shared signature carries primitives, so the counts are
// wrapped as `TokenCount` on the first line past it.
pub fn chat_admitted(
    sink: &mut ChatSink<'_>,
    prompt_tokens: u32,
    reused_tokens: u32,
)
{
    sink.events.admitted(Admission {
        prompt_tokens: TokenCount::from(prompt_tokens),
        reused_tokens: TokenCount::from(reused_tokens),
    });
}

/// Forward committed text to the consumer.
///
/// # Specification
/// - requires: deltas arrive in commit order.
/// - ensures: non-empty text reaches the consumer on its channel; bytes that
///   are not UTF-8 are replaced with U+FFFD, which ninfer's frontend does not
///   produce because it publishes only complete characters.
/// - provides: the bridge's entry for `OutputSink::publish`.
/// - fails: never.
/// - panics: none.
pub fn chat_publish(
    sink: &mut ChatSink<'_>,
    channel: ffi::DeltaChannel,
    text: &[u8],
)
{
    if text.is_empty() {
        return;
    }
    let channel = if channel == ffi::DeltaChannel::Reasoning {
        Channel::Reasoning
    }
    else {
        Channel::Content
    };
    sink.events
        .publish(channel, DeltaText(&String::from_utf8_lossy(text)));
}

/// Whether the consumer asked to stop.
///
/// # Specification
/// - requires: nothing; callable from any thread.
/// - ensures: true exactly after the request's token was cancelled.
/// - provides: the bridge's entry for the Engine's cancellation view.
/// - fails: never.
/// - panics: none.
pub fn chat_cancelled(flag: &CancelFlag) -> bool
{
    return flag.0.state() == Cancellation::Requested;
}

/// Lower one phase's overrides onto the wire.
///
/// # Specification
/// - requires: nothing.
/// - ensures: each set field is carried with its flag raised; each unset field
///   is zero with its flag lowered.
/// - provides: the adapter's view of a phase's overrides.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 — a set field and an unset field of each numeric type.
/// - witness: `tests::set_and_unset_fields_keep_their_flags`
fn lower_sampling(sampling: &Sampling) -> ffi::SamplingFields
{
    /// Split a setting into its value and flag.
    ///
    /// # Specification
    /// trivial.
    fn split<Value>(setting: Setting<Value>) -> (Value, bool)
    where
        Value: Default + Copy,
    {
        return match setting {
            | Setting::ModelDefault => (Value::default(), false),
            | Setting::Set(value) => (value, true),
        };
    }
    let (temperature, temperature_set) = split(sampling.temperature);
    let (top_k, top_k_set) = split(sampling.top_k);
    let (top_p, top_p_set) = split(sampling.top_p);
    let (min_p, min_p_set) = split(sampling.min_p);
    let (presence_penalty, presence_penalty_set) = split(sampling.presence_penalty);
    let (frequency_penalty, frequency_penalty_set) = split(sampling.frequency_penalty);
    return ffi::SamplingFields {
        temperature,
        temperature_set,
        top_k,
        top_k_set,
        top_p,
        top_p_set,
        min_p,
        min_p_set,
        presence_penalty,
        presence_penalty_set,
        frequency_penalty,
        frequency_penalty_set,
    };
}

/// Lower a template switch.
///
/// # Specification
/// trivial.
const fn lower_switch(switch: Switch) -> ffi::TemplateSwitch
{
    return match switch {
        | Switch::ModelDefault => ffi::TemplateSwitch::Unset,
        | Switch::On => ffi::TemplateSwitch::On,
        | Switch::Off => ffi::TemplateSwitch::Off,
    };
}

/// Lower the request's prompt.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the wire prompt carries every turn, tool definition, template
///   argument and choice of `request`, in order.
/// - provides: the adapter's `PromptInput` source.
/// - fails: never.
/// - panics: none.
fn lower_prompt(request: &ChatRequest) -> ffi::ChatPrompt
{
    let prompt = &request.prompt;
    let turns = prompt
        .messages
        .iter()
        .map(|message| {
            return ffi::ChatTurn {
                role: match message.role {
                    | Role::System => ffi::ChatRole::System,
                    | Role::Developer => ffi::ChatRole::Developer,
                    | Role::User => ffi::ChatRole::User,
                    | Role::Assistant => ffi::ChatRole::Assistant,
                    | Role::Tool => ffi::ChatRole::Tool,
                },
                parts: message.parts.clone(),
                reasoning: message.reasoning.clone(),
                tool_calls: message
                    .tool_calls
                    .iter()
                    .map(|call| {
                        return ffi::ChatToolCall {
                            id: call.id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                        };
                    })
                    .collect(),
                tool_call_id: message.tool_call_id.clone(),
            };
        })
        .collect();
    return ffi::ChatPrompt {
        turns,
        tools: prompt.tools.clone(),
        template_arguments: prompt.template_arguments.clone(),
        thinking: lower_switch(prompt.thinking),
        preserve_thinking: lower_switch(prompt.preserve_thinking),
        effort: match prompt.effort {
            | Effort::Unrequested => ffi::EffortLevel::Unset,
            | Effort::None => ffi::EffortLevel::None,
            | Effort::Minimal => ffi::EffortLevel::Minimal,
            | Effort::Low => ffi::EffortLevel::Low,
            | Effort::Medium => ffi::EffortLevel::Medium,
            | Effort::High => ffi::EffortLevel::High,
            | Effort::XHigh => ffi::EffortLevel::XHigh,
            | Effort::Max => ffi::EffortLevel::Max,
        },
        cache_marks: prompt.cache.markers.iter().map(lower_mark).collect(),
        structural_prefixes: prompt.cache.structural == StructuralPrefixes::Allowed,
    };
}

/// Lower one cache marker to its location, counts and ninfer's evidence
/// bits.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the location and counts are `marker`'s boundary; the evidence
///   sets bit 1 for an explicit mark, and bit 2 for a requested or bit 4 for a
///   default automatic write.
/// - provides: the adapter's `PromptCacheMarker` source.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 exhaustive over the four locations and the evidence
///   combinations.
/// - witness: `tests::cache_marks_lower_to_ninfers_bits`
fn lower_mark(marker: &CacheMarker) -> ffi::CacheMark
{
    let (location, count, parts) = match marker.boundary {
        | CacheBoundary::LeadingInstruction(bytes) => {
            (ffi::MarkLocation::LeadingInstruction, bytes.0, 0)
        },
        | CacheBoundary::MessagePart { message, parts } => {
            (ffi::MarkLocation::MessagePart, message.0, parts.0)
        },
        | CacheBoundary::Message(messages) => (ffi::MarkLocation::Message, messages.0, 0),
        | CacheBoundary::Tool(tools) => (ffi::MarkLocation::Tool, tools.0, 0),
    };
    let explicit = match marker.marked {
        | Marked::Unmarked => 0_u8,
        | Marked::Explicit => 1,
    };
    let automatic = match marker.automatic {
        | Automatic::Not => 0_u8,
        | Automatic::Requested => 2,
        | Automatic::Default => 4,
    };
    return ffi::CacheMark {
        location,
        count,
        parts,
        evidence: explicit | automatic,
    };
}

/// Lower the request's generation settings.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the wire settings carry every setting of `request`; an unlimited
///   thinking budget is zero, and an inherited post-thinking seed leaves its
///   flag lowered.
/// - provides: the adapter's `RequestOptions` source.
/// - fails: never.
/// - panics: none.
fn lower_settings(request: &ChatRequest) -> ffi::ChatSettings
{
    let generation = &request.generation;
    let (post_thinking_seed, post_thinking_seed_set) = match generation.post_thinking_seed {
        | Seed::Inherited => (0, false),
        | Seed::Fixed(seed) => (seed, true),
    };
    return ffi::ChatSettings {
        output_tokens: u32::from(generation.output_tokens),
        sampling: lower_sampling(&generation.sampling),
        seed: generation.seed,
        post_thinking: lower_sampling(&generation.post_thinking),
        post_thinking_seed,
        post_thinking_seed_set,
        thinking_budget: match generation.thinking_budget {
            | ThinkingBudget::Unlimited => 0,
            | ThinkingBudget::Tokens(tokens) => tokens.get(),
        },
        stops: generation.stops.clone(),
        stops_end_reasoning: generation.stop_scope == StopScope::ContentAndReasoning,
        preserve_special_tokens: generation.special_tokens == SpecialTokens::Preserved,
        tool_name_limit: generation.tool_name_limit.get(),
        prefix_reuse: generation.prefix_reuse == PrefixReuse::ReadWrite,
        streaming: request.delivery == Delivery::Streaming,
    };
}

/// Classify an adapter outcome as a chat failure.
///
/// # Specification
/// - requires: the adapter wrote `outcome`.
/// - ensures: `Ok` exactly for a completed call; a refusal maps to its
///   [`FailureKind`], `std::invalid_argument` to
///   [`FailureKind::InvalidPrompt`], and anything else to
///   [`FailureKind::Internal`]; the message is ninfer's.
/// - provides: the chat path's reading of the bridge's status.
/// - fails: as stated.
/// - panics: none.
///
/// # Errors
/// - [`ChatFailure`]: the call did not complete.
///
/// # Adequacy
/// - hypothesis: L3 — completion, each refusal class, and the two exception
///   classes.
/// - witness: `tests::outcomes_classify_by_status_and_refusal`
fn classify(outcome: ffi::Outcome) -> Result<(), ChatFailure>
{
    let kind = match outcome.status {
        | ffi::Status::Completed => return Ok(()),
        | ffi::Status::InvalidArgument => FailureKind::InvalidPrompt,
        | ffi::Status::Refused => match outcome.refusal {
            | ffi::Refusal::ContextLength => FailureKind::ContextLength,
            | ffi::Refusal::ThinkingBudgetCapacity => FailureKind::ThinkingBudgetCapacity,
            | ffi::Refusal::Overloaded => FailureKind::Overloaded,
            | ffi::Refusal::QueueTimeout => FailureKind::QueueTimeout,
            | ffi::Refusal::Cancelled => FailureKind::Cancelled,
            | ffi::Refusal::Unavailable => FailureKind::Unavailable,
            | _ => FailureKind::InvalidPrompt,
        },
        | _ => FailureKind::Internal,
    };
    return Err(ChatFailure {
        kind,
        message: outcome.message,
    });
}

/// Read ninfer's finish reason.
///
/// # Specification
/// trivial.
const fn finish_of(finish: ffi::Finish) -> Finish
{
    return match finish {
        | ffi::Finish::OutputLimit => Finish::OutputLimit,
        | ffi::Finish::ContextCapacity => Finish::ContextCapacity,
        | ffi::Finish::StopToken => Finish::StopToken,
        | ffi::Finish::StopString => Finish::StopString,
        | ffi::Finish::Cancelled => Finish::Cancelled,
        | _ => Finish::Unfinished,
    };
}

/// Read the adapter's reuse source.
///
/// # Specification
/// trivial.
fn reuse_of(source: ffi::ReuseSource) -> ReusePath
{
    return match source {
        | ffi::ReuseSource::PrivateEndpoint => ReusePath::PrivateEndpoint,
        | ffi::ReuseSource::TurnClosure => ReusePath::TurnClosure,
        | ffi::ReuseSource::ResponseReplay => ReusePath::ResponseReplay,
        | ffi::ReuseSource::LongAnchor => ReusePath::LongAnchor,
        | ffi::ReuseSource::SharedPrefix => ReusePath::SharedPrefix,
        | _ => ReusePath::Root,
    };
}

/// An empty record for the adapter to fill.
///
/// # Specification
/// trivial.
fn blank_record() -> ffi::ChatRecord
{
    return ffi::ChatRecord {
        content: String::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
        finish: ffi::Finish::Unrecognized,
        prompt_tokens: 0,
        reused_tokens: 0,
        generated: Vec::new(),
        reasoning_tokens: 0,
        prompt_wall_ns: 0,
        generation_wall_ns: 0,
        drafted: 0,
        accepted: 0,
        first_token_ns: 0,
        total_ns: 0,
        prefill_ns: 0,
        decode_ns: 0,
        queue_wait_ns: 0,
        reuse: ffi::ReuseSource::Root,
        thinking_budget: 0,
        thinking_model_tokens: 0,
        thinking_injected_tokens: 0,
        accepted_per_position: Vec::new(),
    };
}

/// ninfer's Engine serving chat requests through infinitum's DFlash2 round.
#[derive(Debug)]
pub struct ChatEngine
{
    /// The open Engine.
    session: Session,
    /// The model name the artifact records.
    model: ModelName,
}

impl ChatEngine
{
    /// Open an Engine for `options` running `plan`, and read its model name.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the Engine is open as [`Session::open`] opens it,
    ///   and the model name is the artifact's.
    /// - provides: the chat backend over ninfer.
    /// - fails: as [`Session::open`] fails, or when reading the name throws.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`EngineFailure`]: opening or reading the name failed.
    ///
    /// # Adequacy
    /// - hypothesis: L1 — exercised on a GPU host by the `serve` acceptance
    ///   run.
    #[inline]
    pub fn open(
        options: &EngineOptions,
        plan: DFlash2Plan,
    ) -> Result<Self, EngineFailure>
    {
        let session = Session::open(options, plan)?;
        let mut model = String::new();
        let mut outcome = pending();
        ffi::model_name(session.engine(), &mut model, &mut outcome);
        check(Operation::Open, outcome)?;
        return Ok(Self {
            session,
            model: ModelName(model),
        });
    }
    /// The capacities the Engine resolved when it opened.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the Engine's KV capacity, storage by its option
    ///   name, sizing, page groups, runtime reservation, free memory and
    ///   effective context-cache capacities.
    /// - provides: the startup capacity lines.
    /// - fails: when ninfer throws.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`EngineFailure`]: reading the summary failed.
    ///
    /// # Adequacy
    /// - hypothesis: L1 — exercised on a GPU host by the `serve` acceptance
    ///   run, whose capacity lines are compared with `ninfer-serve`'s.
    #[inline]
    pub fn capacity(&self) -> Result<Capacity, EngineFailure>
    {
        let mut record = ffi::CapacityRecord {
            kv_tokens: 0,
            kv_storage: ffi::KvStorage::BFloat16,
            kv_sizing: ffi::KvSizing::Explicit,
            pages: 0,
            max_pages: 0,
            runtime_bytes: 0,
            free_bytes: 0,
            cache_enabled: false,
            lanes: 0,
            device_states: 0,
            host_states: 0,
            host_kv_bytes: 0,
            private_continuations: 0,
            shared_prefixes: 0,
            long_anchors: 0,
        };
        let mut outcome = pending();
        ffi::capacity(self.session.engine(), &mut record, &mut outcome);
        check(Operation::Capacity, outcome)?;
        return Ok(capacity_of(&record));
    }
}

/// The bridge's capacity record in the chat crate's terms.
///
/// # Specification
/// - requires: nothing.
/// - ensures: every count carried over; the storage named as its command-line
///   option; the cache root-only when disabled.
/// - provides: [`ChatEngine::capacity`]'s result.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 on both cache forms.
/// - witness: `tests::capacity_records_translate`
fn capacity_of(record: &ffi::CapacityRecord) -> Capacity
{
    let kv_storage = match record.kv_storage {
        | ffi::KvStorage::Int8 => "int8",
        | ffi::KvStorage::Fp8 => "fp8",
        | ffi::KvStorage::Nvfp4 => "nvfp4",
        | ffi::KvStorage::Fp8KeyNvfp4Value => "k8v4",
        | _ => "bf16",
    };
    let cache = if record.cache_enabled {
        ContextCache::Enabled {
            lanes: Tally(u64::from(record.lanes)),
            device_states: Tally(u64::from(record.device_states)),
            host_states: Tally(u64::from(record.host_states)),
            host_kv: ByteSize(record.host_kv_bytes),
            private: Tally(u64::from(record.private_continuations)),
            shared: Tally(u64::from(record.shared_prefixes)),
            anchors: Tally(u64::from(record.long_anchors)),
        }
    }
    else {
        ContextCache::RootOnly
    };
    return Capacity {
        kv_tokens: Tally(u64::from(record.kv_tokens)),
        kv_storage: String::from(kv_storage),
        kv_sizing: if record.kv_sizing == ffi::KvSizing::Automatic {
            KvSizing::Automatic
        }
        else {
            KvSizing::Explicit
        },
        pages: Tally(u64::from(record.pages)),
        max_pages: Tally(u64::from(record.max_pages)),
        runtime: ByteSize(record.runtime_bytes),
        free: ByteSize(record.free_bytes),
        cache,
    };
}

/// The bridge's counter record in the chat crate's terms.
///
/// # Specification
/// trivial.
fn counters_of(record: &ffi::CounterRecord) -> RuntimeCounters
{
    return RuntimeCounters {
        prefill_tokens: Tally(record.prefill_tokens),
        decode_tokens: Tally(record.decode_tokens),
        decode_rounds: Tally(record.decode_rounds),
        decode_rows: Tally(record.decode_rows),
        requests: RequestGauges {
            running: Tally(u64::from(record.running)),
            prefilling: Tally(u64::from(record.prefilling)),
            decode_ready: Tally(u64::from(record.decode_ready)),
            waiting: Tally(u64::from(record.waiting)),
            materializing: Tally(u64::from(record.materializing)),
            capture_pending: Tally(u64::from(record.capture_pending)),
            terminal_pending: Tally(u64::from(record.terminal_pending)),
        },
        host_active: core::time::Duration::from_nanos(record.host_active_ns),
        digest: CounterDigest(record.digest),
    };
}

impl ChatBackend for ChatEngine
{
    /// The artifact's model name.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn model_name(&self) -> &ModelName
    {
        return &self.model;
    }

    /// The Engine's published runtime statistics.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: as [`ChatBackend::counters`] specifies; the totals, gauges,
    ///   host-active time and digest are ninfer's.
    /// - provides: the throughput report's samples on ninfer.
    /// - fails: when ninfer throws, as [`FailureKind::Internal`].
    /// - panics: none.
    ///
    /// # Errors
    /// - [`ChatFailure`]: as stated.
    ///
    /// # Adequacy
    /// - hypothesis: L1 — exercised on a GPU host by the `serve` acceptance
    ///   run, whose throughput lines are compared with `ninfer-serve`'s.
    #[inline]
    fn counters(&self) -> Result<RuntimeCounters, ChatFailure>
    {
        let mut record = ffi::CounterRecord {
            prefill_tokens: 0,
            decode_tokens: 0,
            decode_rounds: 0,
            decode_rows: 0,
            running: 0,
            prefilling: 0,
            decode_ready: 0,
            waiting: 0,
            materializing: 0,
            capture_pending: 0,
            terminal_pending: 0,
            host_active_ns: 0,
            digest: 0,
        };
        let mut outcome = pending();
        ffi::counters(self.session.engine(), &mut record, &mut outcome);
        classify(outcome)?;
        return Ok(counters_of(&record));
    }

    /// Run `request` on the Engine, with infinitum's preview reviewing each
    /// round against the request's output limit.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: as [`ChatBackend::run`] specifies; the outcome's text, tool
    ///   calls, ids, accounting and tallies are ninfer's.
    /// - provides: one chat generation on ninfer.
    /// - fails: when ninfer refuses or throws, classified by [`FailureKind`],
    ///   or when the preview's counts overflow, as [`FailureKind::Internal`].
    /// - panics: none.
    ///
    /// # Errors
    /// - [`ChatFailure`]: as stated.
    ///
    /// # Adequacy
    /// - hypothesis: L2 — greedy ids and text equal `ninfer-serve`'s for the
    ///   same request on the same artifact and options, checked on a GPU host
    ///   by the `serve` acceptance run.
    #[inline]
    fn run(
        &self,
        request: &ChatRequest,
        events: &mut dyn ChatEvents,
        cancel: &CancelToken,
    ) -> Result<ChatOutcome, ChatFailure>
    {
        let prompt = lower_prompt(request);
        let settings = lower_settings(request);
        let mut sink = ChatSink { events };
        let flag = CancelFlag(cancel.clone());
        let mut review = ReviewSink::new(request.generation.output_tokens);
        let mut record = blank_record();
        let mut outcome = pending();
        ffi::run_chat(
            self.session.engine(),
            &prompt,
            &settings,
            &mut sink,
            &flag,
            &mut review,
            &mut record,
            &mut outcome,
        );
        classify(outcome)?;
        if let Maybe::Present(overflow) = review.failure() {
            return Err(ChatFailure {
                kind: FailureKind::Internal,
                message: format!("the round preview failed: {overflow}"),
            });
        }
        return Ok(ChatOutcome {
            content: record.content,
            reasoning: record.reasoning,
            tool_calls: record
                .tool_calls
                .into_iter()
                .map(|call| {
                    return GeneratedToolCall {
                        name: call.name,
                        arguments: call.arguments,
                    };
                })
                .collect(),
            finish: finish_of(record.finish),
            admission: Admission {
                prompt_tokens: TokenCount::from(record.prompt_tokens),
                reused_tokens: TokenCount::from(record.reused_tokens),
            },
            generated: record.generated.into_iter().map(TokenId::from).collect(),
            reasoning_tokens: TokenCount::from(record.reasoning_tokens),
            prompt_wall: core::time::Duration::from_nanos(record.prompt_wall_ns),
            generation_wall: core::time::Duration::from_nanos(record.generation_wall_ns),
            drafted: Tally(record.drafted),
            accepted: Tally(record.accepted),
            telemetry: Telemetry {
                first_token: core::time::Duration::from_nanos(record.first_token_ns),
                total: core::time::Duration::from_nanos(record.total_ns),
                prefill: core::time::Duration::from_nanos(record.prefill_ns),
                decode: core::time::Duration::from_nanos(record.decode_ns),
                queue_wait: core::time::Duration::from_nanos(record.queue_wait_ns),
                reuse: reuse_of(record.reuse),
                thinking: ThinkingSpend {
                    budget: core::num::NonZeroU32::new(record.thinking_budget)
                        .map_or(ThinkingBudget::Unlimited, ThinkingBudget::Tokens),
                    model_tokens: TokenCount::from(record.thinking_model_tokens),
                    injected_tokens: TokenCount::from(record.thinking_injected_tokens),
                },
                accepted_per_position: record
                    .accepted_per_position
                    .into_iter()
                    .map(Tally)
                    .collect(),
            },
        });
    }
}

/// Tests for the lowering and the outcome classification.
#[cfg(test)]
mod tests
{
    use infinitum_chat::Automatic;
    use infinitum_chat::CacheBoundary;
    use infinitum_chat::CacheMarker;
    use infinitum_chat::Count;
    use infinitum_chat::FailureKind;
    use infinitum_chat::InstructionBytes;
    use infinitum_chat::Marked;
    use infinitum_chat::Sampling;
    use infinitum_chat::Setting;

    use super::capacity_of;
    use super::classify;
    use super::ffi;
    use super::lower_mark;
    use super::lower_sampling;

    /// Each storage takes ninfer's log name, the sizing and counts carry
    /// over, and a disabled cache is root-only whatever its counts read.
    #[test]
    fn capacity_records_translate()
    {
        let mut record = ffi::CapacityRecord {
            kv_tokens: 539_520,
            kv_storage: ffi::KvStorage::Fp8KeyNvfp4Value,
            kv_sizing: ffi::KvSizing::Automatic,
            pages: 2_107,
            max_pages: 2_108,
            runtime_bytes: 3,
            free_bytes: 4,
            cache_enabled: true,
            lanes: 3,
            device_states: 5,
            host_states: 8,
            host_kv_bytes: 9,
            private_continuations: 6,
            shared_prefixes: 4,
            long_anchors: 2,
        };
        let capacity = capacity_of(&record);
        assert_eq!(
            (
                capacity.kv_tokens,
                capacity.kv_sizing,
                capacity.pages,
                capacity.max_pages
            ),
            (
                infinitum_chat::Tally(539_520),
                infinitum_chat::KvSizing::Automatic,
                infinitum_chat::Tally(2_107),
                infinitum_chat::Tally(2_108)
            ),
            "the KV counts and sizing"
        );
        assert_eq!(
            capacity.cache,
            infinitum_chat::ContextCache::Enabled {
                lanes: infinitum_chat::Tally(3),
                device_states: infinitum_chat::Tally(5),
                host_states: infinitum_chat::Tally(8),
                host_kv: infinitum_chat::ByteSize(9),
                private: infinitum_chat::Tally(6),
                shared: infinitum_chat::Tally(4),
                anchors: infinitum_chat::Tally(2),
            },
            "an enabled cache's capacities"
        );
        for (storage, name) in [
            (ffi::KvStorage::BFloat16, "bf16"),
            (ffi::KvStorage::Int8, "int8"),
            (ffi::KvStorage::Fp8, "fp8"),
            (ffi::KvStorage::Nvfp4, "nvfp4"),
            (ffi::KvStorage::Fp8KeyNvfp4Value, "k8v4"),
        ] {
            record.kv_storage = storage;
            assert_eq!(capacity_of(&record).kv_storage, name, "{name}");
        }
        record.cache_enabled = false;
        record.kv_sizing = ffi::KvSizing::Explicit;
        let disabled = capacity_of(&record);
        assert_eq!(
            disabled.cache,
            infinitum_chat::ContextCache::RootOnly,
            "a disabled cache"
        );
        assert_eq!(
            disabled.kv_sizing,
            infinitum_chat::KvSizing::Explicit,
            "explicit sizing"
        );
    }

    /// Each location carries its counts, and the evidence bits are ninfer's
    /// `SharedCandidateEvidence` values.
    #[test]
    fn cache_marks_lower_to_ninfers_bits()
    {
        let cases = [
            (
                CacheBoundary::LeadingInstruction(InstructionBytes(12)),
                Marked::Explicit,
                Automatic::Not,
                (ffi::MarkLocation::LeadingInstruction, 12_u32, 0_u32, 1_u8),
            ),
            (
                CacheBoundary::MessagePart {
                    message: Count(3),
                    parts: Count(2),
                },
                Marked::Explicit,
                Automatic::Default,
                (ffi::MarkLocation::MessagePart, 3, 2, 5),
            ),
            (
                CacheBoundary::Message(Count(4)),
                Marked::Unmarked,
                Automatic::Default,
                (ffi::MarkLocation::Message, 4, 0, 4),
            ),
            (
                CacheBoundary::Tool(Count(1)),
                Marked::Unmarked,
                Automatic::Requested,
                (ffi::MarkLocation::Tool, 1, 0, 2),
            ),
        ];
        for (boundary, marked, automatic, expected) in cases {
            let mark = lower_mark(&CacheMarker {
                boundary,
                marked,
                automatic,
            });
            assert_eq!(
                (mark.location, mark.count, mark.parts, mark.evidence),
                expected,
                "{boundary:?} {marked:?} {automatic:?}"
            );
        }
    }

    #[test]
    fn set_and_unset_fields_keep_their_flags()
    {
        let sampling = Sampling {
            temperature: Setting::Set(0.0),
            top_k: Setting::Set(20_i32),
            ..Sampling::MODEL_DEFAULT
        };
        let wire = lower_sampling(&sampling);
        assert!(
            wire.temperature_set,
            "an explicit zero temperature stays set"
        );
        assert!(wire.top_k_set, "a set top-k stays set");
        assert_eq!(wire.top_k, 20_i32, "a set top-k keeps its value");
        assert!(!wire.top_p_set, "an unset top-p stays unset");
        assert!(!wire.frequency_penalty_set, "an unset penalty stays unset");
    }

    /// An outcome with `status` and `refusal`.
    ///
    /// # Specification
    /// trivial.
    fn outcome(
        status: ffi::Status,
        refusal: ffi::Refusal,
    ) -> ffi::Outcome
    {
        return ffi::Outcome {
            status,
            refusal,
            message: String::from("message"),
        };
    }

    #[test]
    fn outcomes_classify_by_status_and_refusal()
    {
        assert!(
            classify(outcome(ffi::Status::Completed, ffi::Refusal::None)).is_ok(),
            "a completed call is not a failure"
        );
        let cases = [
            (ffi::Refusal::ContextLength, FailureKind::ContextLength),
            (
                ffi::Refusal::ThinkingBudgetCapacity,
                FailureKind::ThinkingBudgetCapacity,
            ),
            (ffi::Refusal::Overloaded, FailureKind::Overloaded),
            (ffi::Refusal::QueueTimeout, FailureKind::QueueTimeout),
            (ffi::Refusal::Cancelled, FailureKind::Cancelled),
            (ffi::Refusal::Unavailable, FailureKind::Unavailable),
            (ffi::Refusal::Media, FailureKind::InvalidPrompt),
        ];
        for (refusal, kind) in cases {
            let failure = classify(outcome(ffi::Status::Refused, refusal));
            assert_eq!(
                failure.map_err(|failure| return failure.kind),
                Err(kind),
                "a refusal maps to its class"
            );
        }
        assert_eq!(
            classify(outcome(ffi::Status::InvalidArgument, ffi::Refusal::None))
                .map_err(|failure| return failure.kind),
            Err(FailureKind::InvalidPrompt),
            "std::invalid_argument is an invalid prompt"
        );
        assert_eq!(
            classify(outcome(ffi::Status::Runtime, ffi::Refusal::None))
                .map_err(|failure| return failure.kind),
            Err(FailureKind::Internal),
            "any other exception is internal"
        );
    }
}
