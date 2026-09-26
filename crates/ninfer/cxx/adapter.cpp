#include "infinitum-ninfer/cxx/adapter.h"

#include "infinitum-ninfer/src/bridge.rs.h"
#include "ninfer/round_control.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinitum::ninfer {
namespace {

/// Record success.
///
/// # Specification
/// trivial.
void succeed(Outcome& outcome) noexcept {
    outcome.status  = Status::Completed;
    outcome.refusal = Refusal::None;
    outcome.message = rust::String();
}

/// Map ninfer's request-error class onto the bridge's.
///
/// # Specification
/// - requires: nothing.
/// - ensures: each class maps to its namesake; the two media classes map to `Media`.
/// - provides: the refusal field of a refused outcome.
/// - fails: never.
/// - panics: none.
Refusal to_refusal(::ninfer::RequestErrorKind kind) noexcept {
    switch (kind) {
    case ::ninfer::RequestErrorKind::ContextLengthExceeded: return Refusal::ContextLength;
    case ::ninfer::RequestErrorKind::ThinkingBudgetCapacityInsufficient:
        return Refusal::ThinkingBudgetCapacity;
    case ::ninfer::RequestErrorKind::MediaBudgetExceeded:
    case ::ninfer::RequestErrorKind::InvalidMedia: return Refusal::Media;
    case ::ninfer::RequestErrorKind::Overloaded: return Refusal::Overloaded;
    case ::ninfer::RequestErrorKind::QueueTimeout: return Refusal::QueueTimeout;
    case ::ninfer::RequestErrorKind::Cancelled: return Refusal::Cancelled;
    case ::ninfer::RequestErrorKind::Unavailable: return Refusal::Unavailable;
    }
    return Refusal::Unavailable;
}

/// Classify the exception in flight into `outcome`.
///
/// # Specification
/// - requires: called only from inside a catch handler, so an exception is in flight.
/// - ensures: `outcome.status` is `Refused` with the class for `ninfer::RequestError`,
///   `InvalidArgument` for any other `std::invalid_argument`, `Runtime` for any other
///   `std::exception`, and `Unknown` otherwise; `outcome.message` carries `what()` when there is
///   one.
/// - provides: the one mapping from ninfer's exceptions to the bridge's statuses.
/// - fails: never.
/// - panics: none.
void fail(Outcome& outcome) noexcept {
    outcome.refusal = Refusal::None;
    try {
        throw;
    } catch (const ::ninfer::RequestError& error) {
        outcome.status  = Status::Refused;
        outcome.refusal = to_refusal(error.kind());
        outcome.message = rust::String::lossy(error.what());
    } catch (const std::invalid_argument& error) {
        outcome.status  = Status::InvalidArgument;
        outcome.message = rust::String::lossy(error.what());
    } catch (const std::exception& error) {
        outcome.status  = Status::Runtime;
        outcome.message = rust::String::lossy(error.what());
    } catch (...) {
        outcome.status  = Status::Unknown;
        outcome.message = rust::String::lossy("ninfer threw a value that is not a std::exception");
    }
}

/// Map ninfer's finish reason onto the bridge's.
///
/// # Specification
/// - requires: nothing.
/// - ensures: each known reason maps to its namesake; a value outside the enumeration maps to
///   `Unrecognized`.
/// - provides: the record's finish field.
/// - fails: never.
/// - panics: none.
Finish to_finish(::ninfer::FinishReason reason) noexcept {
    switch (reason) {
    case ::ninfer::FinishReason::None: return Finish::Unfinished;
    case ::ninfer::FinishReason::OutputLimit: return Finish::OutputLimit;
    case ::ninfer::FinishReason::ContextCapacity: return Finish::ContextCapacity;
    case ::ninfer::FinishReason::StopToken: return Finish::StopToken;
    case ::ninfer::FinishReason::StopString: return Finish::StopString;
    case ::ninfer::FinishReason::Cancelled: return Finish::Cancelled;
    }
    return Finish::Unrecognized;
}

/// Forwards each round to the Rust sink until detached. The Engine may hold the controller past
/// the request's end, so the sink is reached only through a pointer the generating call clears
/// before it returns.
class Controller final : public ::ninfer::RoundController {
public:
    /// Attach to `sink`.
    ///
    /// # Specification
    /// - requires: `sink` outlives every `review` before `detach`.
    /// - ensures: the controller is attached to `sink`.
    /// - provides: the controller `generate` hands to the Engine.
    /// - fails: never.
    /// - panics: none.
    explicit Controller(ReviewSink& sink) noexcept : sink_(&sink) {}

    /// Review one round through the Rust sink.
    ///
    /// # Specification
    /// - requires: nothing; ninfer calls it on its worker thread.
    /// - ensures: while attached, the verdict is the sink's answer (a limit carries its length);
    ///   once detached, the verdict is to continue and the sink is not reached.
    /// - provides: the host decision each round asks for.
    /// - fails: never; the Rust side holds any failure in the sink.
    /// - panics: none.
    ::ninfer::RoundVerdict review(const ::ninfer::RoundOffer& offer) noexcept override {
        const std::lock_guard lock(mutex_);
        if (sink_ == nullptr) { return {}; }
        const rust::Slice<const std::int32_t> licensed(offer.licensed.data(),
                                                       offer.licensed.size());
        const auto kind = offer.decode_round ? RoundKind::Decode : RoundKind::PrefillFinalization;
        const ReviewAnswer answer = review_round(*sink_, licensed, kind);
        if (answer.verdict == Verdict::Limit) { return {.limit = answer.limit}; }
        return {};
    }

    /// Stop reaching the sink.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: every later `review` returns without touching the sink, and a `review` in
    ///   progress finishes before this returns.
    /// - provides: the lifetime cut `generate` relies on.
    /// - fails: never.
    /// - panics: none.
    void detach() noexcept {
        const std::lock_guard lock(mutex_);
        sink_ = nullptr;
    }

private:
    std::mutex mutex_;
    ReviewSink* sink_;
};

/// Detaches the controller on every exit from generate.
class Detach {
public:
    /// Guard `controller` until the end of the enclosing scope.
    ///
    /// # Specification
    /// - requires: `controller` outlives the guard.
    /// - ensures: the guard refers to `controller`.
    /// - provides: the scope `generate` detaches on leaving.
    /// - fails: never.
    /// - panics: none.
    explicit Detach(Controller& controller) noexcept : controller_(controller) {}
    Detach(const Detach&)            = delete;
    Detach& operator=(const Detach&) = delete;
    Detach(Detach&&)                 = delete;
    Detach& operator=(Detach&&)      = delete;
    /// Detach the controller, on every exit from the guard's scope, exceptional ones included.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: `controller.detach()` has returned, so no later `review` reaches the sink and
    ///   none is still inside it.
    /// - provides: the lifetime cut the bridge's `# Safety` section relies on.
    /// - fails: never.
    /// - panics: none.
    ~Detach() noexcept { controller_.detach(); }

private:
    Controller& controller_;
};

/// The greedy request of ninfer's own C facade: argmax, no penalties, the model's stop tokens.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the request asks for at most `budget` tokens with temperature 0, top-k off, top-p 1,
///   min-p 0, no penalties and seed 0.
/// - provides: the options every generation uses.
/// - fails: when building the options throws, such as on allocation failure; it is not
///   `noexcept`, and its one caller, `generate`, calls it inside its catch-all.
/// - panics: none.
::ninfer::RequestOptions greedy(std::uint32_t budget) {
    ::ninfer::RequestOptions request;
    request.execution.requested_output_tokens    = budget;
    request.execution.sampling.temperature       = 0.0F;
    request.execution.sampling.top_k             = 0;
    request.execution.sampling.top_p             = 1.0F;
    request.execution.sampling.min_p             = 0.0F;
    request.execution.sampling.presence_penalty  = 0.0F;
    request.execution.sampling.frequency_penalty = 0.0F;
    request.execution.sampling.seed              = 0;
    return request;
}

/// Convert seconds to whole nanoseconds.
///
/// # Specification
/// - requires: nothing.
/// - ensures: zero for a non-positive or NaN input; otherwise the truncated count.
/// - provides: the record's wall-time field.
/// - fails: never.
/// - panics: none.
std::uint64_t nanoseconds(double seconds) noexcept {
    if (!(seconds > 0.0)) { return 0; }
    return static_cast<std::uint64_t>(seconds * 1e9);
}

/// Forwards the Engine's streamed output to the Rust sink on the waiting thread.
class EventForwarder final : public ::ninfer::OutputSink {
public:
    /// Forward to `sink`.
    ///
    /// # Specification
    /// - requires: `sink` outlives the forwarder.
    /// - ensures: the forwarder refers to `sink`.
    /// - provides: the output sink `run_chat` hands to `GenerationHandle::wait`.
    /// - fails: never.
    /// - panics: none.
    explicit EventForwarder(ChatSink& sink) noexcept : sink_(sink) {}

    /// Forward the admission record.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the Rust sink received the prompt and reused counts.
    /// - provides: the stream's first event.
    /// - fails: never.
    /// - panics: none.
    void start(::ninfer::GenerationStart start) noexcept override {
        chat_admitted(sink_, start.prompt.prompt_tokens, start.reused_prompt_tokens);
    }

    /// Ignore prompt progress, which the chat surface does not publish.
    ///
    /// # Specification
    /// trivial.
    void progress(::ninfer::PromptProgress) noexcept override {}

    /// Ignore live timings, which the chat surface does not publish.
    ///
    /// # Specification
    /// trivial.
    void timing(::ninfer::GenerationTimingObservation) noexcept override {}

    /// Forward one delta's bytes on its channel.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: non-empty text reached the Rust sink on its channel.
    /// - provides: the streamed output.
    /// - fails: never.
    /// - panics: none.
    void publish(::ninfer::OutputDelta delta) noexcept override {
        if (delta.text.empty()) { return; }
        const rust::Slice<const std::uint8_t> bytes(
            reinterpret_cast<const std::uint8_t*>(delta.text.data()), delta.text.size());
        chat_publish(sink_,
                     delta.channel == ::ninfer::OutputChannel::Reasoning ? DeltaChannel::Reasoning
                                                                         : DeltaChannel::Content,
                     bytes);
    }

private:
    ChatSink& sink_;
};

/// Read a template switch.
///
/// # Specification
/// trivial.
std::optional<bool> to_switch(TemplateSwitch value) noexcept {
    if (value == TemplateSwitch::On) { return true; }
    if (value == TemplateSwitch::Off) { return false; }
    return std::nullopt;
}

/// Read a requested effort.
///
/// # Specification
/// trivial.
std::optional<::ninfer::ReasoningEffort> to_effort(EffortLevel value) noexcept {
    switch (value) {
    case EffortLevel::None: return ::ninfer::ReasoningEffort::None;
    case EffortLevel::Minimal: return ::ninfer::ReasoningEffort::Minimal;
    case EffortLevel::Low: return ::ninfer::ReasoningEffort::Low;
    case EffortLevel::Medium: return ::ninfer::ReasoningEffort::Medium;
    case EffortLevel::High: return ::ninfer::ReasoningEffort::High;
    case EffortLevel::XHigh: return ::ninfer::ReasoningEffort::XHigh;
    case EffortLevel::Max: return ::ninfer::ReasoningEffort::Max;
    default: return std::nullopt;
    }
}

/// Read a KV storage.
///
/// # Specification
/// trivial.
::ninfer::KvCacheStorage to_storage(KvStorage storage) noexcept {
    switch (storage) {
    case KvStorage::Int8: return ::ninfer::KvCacheStorage::Int8Group64;
    case KvStorage::Fp8: return ::ninfer::KvCacheStorage::Fp8E4M3Row256;
    case KvStorage::Nvfp4: return ::ninfer::KvCacheStorage::Nvfp4Group16;
    case KvStorage::Fp8KeyNvfp4Value: return ::ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
    default: return ::ninfer::KvCacheStorage::BFloat16;
    }
}

/// Read a turn's role.
///
/// # Specification
/// trivial.
::ninfer::ChatRole to_role(ChatRole role) noexcept {
    switch (role) {
    case ChatRole::System: return ::ninfer::ChatRole::System;
    case ChatRole::Developer: return ::ninfer::ChatRole::Developer;
    case ChatRole::Assistant: return ::ninfer::ChatRole::Assistant;
    case ChatRole::Tool: return ::ninfer::ChatRole::Tool;
    default: return ::ninfer::ChatRole::User;
    }
}

/// Read one phase's overrides; an unset field stays empty.
///
/// # Specification
/// trivial.
::ninfer::SamplingOverrides to_overrides(const SamplingFields& fields) noexcept {
    ::ninfer::SamplingOverrides overrides;
    if (fields.temperature_set) { overrides.temperature = fields.temperature; }
    if (fields.top_k_set) { overrides.top_k = fields.top_k; }
    if (fields.top_p_set) { overrides.top_p = fields.top_p; }
    if (fields.min_p_set) { overrides.min_p = fields.min_p; }
    if (fields.presence_penalty_set) { overrides.presence_penalty = fields.presence_penalty; }
    if (fields.frequency_penalty_set) { overrides.frequency_penalty = fields.frequency_penalty; }
    return overrides;
}

/// Build the Engine's prompt input from the wire prompt.
///
/// # Specification
/// - requires: nothing.
/// - ensures: every turn, part, carried reasoning, tool call, tool definition, template argument
///   and choice is carried in order; the new turn is a fresh assistant turn; each cache mark
///   becomes a shared-stable-prefix marker at its location with its evidence, and structural
///   shared prefixes are allowed only as the prompt says.
/// - provides: the input `Engine::prepare` renders.
/// - fails: when allocation throws; it is not `noexcept`, and its one caller, `run_chat`, calls
///   it inside its catch-all.
/// - panics: none.
::ninfer::PromptInput chat_prompt_input(const ChatPrompt& prompt) {
    ::ninfer::PromptInput input;
    input.messages.reserve(prompt.turns.size());
    for (const auto& turn : prompt.turns) {
        ::ninfer::ChatMessage message;
        message.role              = to_role(turn.role);
        message.reasoning_content = std::string(turn.reasoning);
        message.tool_call_id      = std::string(turn.tool_call_id);
        for (const auto& part : turn.parts) {
            ::ninfer::MessagePart text;
            text.text = std::string(part);
            message.parts.push_back(std::move(text));
        }
        for (const auto& call : turn.tool_calls) {
            message.tool_calls.push_back(::ninfer::ToolCall{
                .id             = std::string(call.id),
                .name           = std::string(call.name),
                .arguments_json = std::string(call.arguments),
            });
        }
        input.messages.push_back(std::move(message));
    }
    input.options.continuation              = ::ninfer::PromptContinuationMode::NewAssistantTurn;
    input.options.enable_thinking           = to_switch(prompt.thinking);
    input.options.reasoning_effort          = to_effort(prompt.effort);
    input.options.preserve_thinking         = to_switch(prompt.preserve_thinking);
    input.options.chat_template_kwargs_json = std::string(prompt.template_arguments);
    input.options.add_vision_id             = false;
    for (const auto& tool : prompt.tools) { input.options.tool_jsons.emplace_back(tool); }
    for (const auto& mark : prompt.cache_marks) {
        ::ninfer::PromptCacheMarker marker;
        marker.kind     = ::ninfer::PromptCacheMarkerKind::SharedStablePrefix;
        marker.evidence = static_cast<::ninfer::SharedCandidateEvidence>(mark.evidence);
        switch (mark.location) {
        case MarkLocation::LeadingInstruction:
            marker.location = ::ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary;
            marker.leading_instruction_bytes = mark.count;
            break;
        case MarkLocation::MessagePart:
            marker.location = ::ninfer::PromptCacheMarkerLocation::MessagePartBoundary;
            marker.after_message_count      = mark.count;
            marker.after_message_part_count = mark.parts;
            break;
        case MarkLocation::Tool:
            marker.location         = ::ninfer::PromptCacheMarkerLocation::ToolBoundary;
            marker.after_tool_count = mark.count;
            break;
        default:
            marker.location            = ::ninfer::PromptCacheMarkerLocation::MessageBoundary;
            marker.after_message_count = mark.count;
            break;
        }
        input.context_cache.markers.push_back(marker);
    }
    input.context_cache.allow_engine_automatic_shared_prefixes = prompt.structural_prefixes;
    return input;
}

/// Build the Engine's request options from the wire settings.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the output limit, prefix participation, thinking budget (none for zero), both
///   phases' overrides and seeds (the post-thinking seed inherits the initial one unless set),
///   output options and stop strings (each once for content, and again for reasoning when
///   `stops_end_reasoning`) are those of `settings`.
/// - provides: the options `Engine::submit` runs.
/// - fails: when allocation throws; as for `chat_prompt_input`.
/// - panics: none.
::ninfer::RequestOptions chat_request_options(const ChatSettings& settings) {
    ::ninfer::RequestOptions request;
    request.execution.requested_output_tokens = settings.output_tokens;
    request.execution.allow_prefix_reuse      = settings.prefix_reuse;
    if (settings.thinking_budget != 0) {
        request.execution.thinking.budget = settings.thinking_budget;
    }
    request.execution.sampling      = to_overrides(settings.sampling);
    request.execution.sampling.seed = settings.seed;
    request.execution.post_thinking_sampling = to_overrides(settings.post_thinking);
    request.execution.post_thinking_sampling.seed =
        settings.post_thinking_seed_set ? settings.post_thinking_seed : settings.seed;
    request.output.raw                     = false;
    request.output.preserve_special_tokens = settings.preserve_special_tokens;
    request.output.tool_name_max_length    = settings.tool_name_limit;
    for (const auto& stop : settings.stops) {
        request.stop.strings.push_back(::ninfer::StopString{
            .text              = std::string(stop),
            .channel           = ::ninfer::OutputChannel::Content,
            .include_in_output = false,
        });
        if (settings.stops_end_reasoning) {
            request.stop.strings.push_back(::ninfer::StopString{
                .text              = std::string(stop),
                .channel           = ::ninfer::OutputChannel::Reasoning,
                .include_in_output = false,
            });
        }
    }
    return request;
}

} // namespace

Session::Session(::ninfer::Engine engine_) noexcept : engine(std::move(engine_)) {}

std::unique_ptr<Session> open_session(const EngineConfig& config, Outcome& outcome) noexcept {
    try {
        ::ninfer::EngineOptions options;
        options.artifact_path = std::string(config.artifact);
        if (!config.chat_template.empty()) {
            options.chat_template_path = std::string(config.chat_template);
        }
        options.device        = config.device;
        options.max_context   = config.max_context;
        options.kv_capacity   = config.kv_sizing == KvSizing::Automatic
                                    ? ::ninfer::KvCapacityPolicy::automatic()
                                    : ::ninfer::KvCapacityPolicy::explicit_capacity(config.kv_capacity);
        options.kv_cache      = to_storage(config.kv_storage);
        options.prefill_chunk = config.prefill_chunk;
        options.max_concurrency = config.max_concurrency;
        if (config.device_state_set) {
            options.context_cache.device_state_slots = config.device_state_slots;
        }
        options.context_cache.host_state_slots       = config.host_state_slots;
        options.context_cache.host_kv_capacity_bytes = config.host_kv_bytes;
        options.pending_timeout_ms        = config.pending_timeout_ms;
        options.use_cuda_graph            = config.cuda_graph == CudaGraph::On;
        options.speculative.backend       = ::ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens  = config.draft_width;
        options.speculative.proposal_head = ::ninfer::ProposalHead::Optimized;
        auto session = std::make_unique<Session>(::ninfer::Engine(std::move(options)));
        succeed(outcome);
        return session;
    } catch (...) {
        fail(outcome);
        return nullptr;
    }
}

void tokenize(const Session& session, rust::Str text, rust::Vec<std::int32_t>& ids,
              Outcome& outcome) noexcept {
    try {
        const auto encoded = session.engine.tokenize_text(std::string_view(text.data(), text.size()));
        ids.clear();
        ids.reserve(encoded.size());
        for (const auto id : encoded) { ids.push_back(id); }
        succeed(outcome);
    } catch (...) { fail(outcome); }
}

void generate(Session& session, rust::Slice<const std::int32_t> prompt, std::uint32_t budget,
              ReviewSink& sink, GenerationRecord& record, Outcome& outcome) noexcept {
    try {
        auto controller = std::make_shared<Controller>(sink);
        const Detach detach(*controller);
        auto prepared = session.engine.prepare_tokens(
            std::vector<::ninfer::TokenId>(prompt.begin(), prompt.end()));
        ::ninfer::GenerationObservationOptions observation;
        observation.phase_timings = true;
        auto handle = session.engine.submit(std::move(prepared), greedy(budget),
                                            ::ninfer::OutputConsumerMode::Aggregate, observation,
                                            {}, controller);
        const auto result = handle.wait();
        record.generated.clear();
        record.generated.reserve(result.generated_token_ids.size());
        for (const auto id : result.generated_token_ids) { record.generated.push_back(id); }
        record.finish             = to_finish(result.finish_reason);
        record.decode_rounds      = result.speculative.rounds;
        record.drafted            = result.speculative.drafted_tokens;
        record.accepted           = result.speculative.accepted_tokens;
        record.fallback_steps     = result.speculative.fallback_steps;
        record.generation_wall_ns = nanoseconds(result.timings.generation_wall_seconds);
        succeed(outcome);
    } catch (...) { fail(outcome); }
}

void detokenize(const Session& session, rust::Slice<const std::int32_t> ids,
                rust::Vec<std::uint8_t>& bytes, Outcome& outcome) noexcept {
    try {
        const std::span<const ::ninfer::TokenId> view(ids.data(), ids.size());
        const std::string rendered = session.engine.detokenize(view);
        bytes.clear();
        bytes.reserve(rendered.size());
        for (const char byte : rendered) { bytes.push_back(static_cast<std::uint8_t>(byte)); }
        succeed(outcome);
    } catch (...) { fail(outcome); }
}

void model_name(const Session& session, rust::String& name, Outcome& outcome) noexcept {
    try {
        name = rust::String::lossy(session.engine.load_summary().model_name);
        succeed(outcome);
    } catch (...) { fail(outcome); }
}

void run_chat(const Session& session, const ChatPrompt& prompt, const ChatSettings& settings,
              ChatSink& sink, const CancelFlag& cancel, ReviewSink& review, ChatRecord& record,
              Outcome& outcome) noexcept {
    try {
        auto controller = std::make_shared<Controller>(review);
        const Detach detach(*controller);
        const ::ninfer::CancellationView cancellation(
            [&cancel]() noexcept { return chat_cancelled(cancel); });
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(session.engine.options().pending_timeout_ms);
        ::ninfer::RequestOptions request = chat_request_options(settings);
        auto prepared = session.engine.prepare(chat_prompt_input(prompt),
                                               ::ninfer::PreparationControl{
                                                   .deadline     = deadline,
                                                   .cancellation = cancellation,
                                               });
        // ninfer's server drops the budget when the rendered turn does not open in thinking.
        if (!prepared.summary().starts_in_reasoning) { request.execution.thinking.budget.reset(); }
        ::ninfer::GenerationObservationOptions observation;
        observation.phase_timings = true;
        auto handle = session.engine.submit(std::move(prepared), std::move(request),
                                            settings.streaming
                                                ? ::ninfer::OutputConsumerMode::Streaming
                                                : ::ninfer::OutputConsumerMode::Aggregate,
                                            observation, deadline, controller);
        EventForwarder forwarder(sink);
        if (settings.streaming) { chat_submitted(sink); }
        auto result = handle.wait(settings.streaming ? &forwarder : nullptr, cancellation);
        record.content   = rust::String::lossy(result.content);
        record.reasoning = rust::String::lossy(result.reasoning);
        record.tool_calls.clear();
        for (const auto& call : result.tool_calls) {
            record.tool_calls.push_back(ChatGeneratedCall{
                .name      = rust::String::lossy(call.name),
                .arguments = rust::String::lossy(call.arguments_json),
            });
        }
        record.finish        = to_finish(result.finish_reason);
        record.prompt_tokens = result.prompt.prompt_tokens;
        record.reused_tokens = result.reused_prompt_tokens;
        record.generated.clear();
        record.generated.reserve(result.generated_token_ids.size());
        for (const auto id : result.generated_token_ids) { record.generated.push_back(id); }
        record.reasoning_tokens   = result.reasoning_tokens;
        record.prompt_wall_ns     = nanoseconds(result.timings.prompt_wall_seconds);
        record.generation_wall_ns = nanoseconds(result.timings.generation_wall_seconds);
        record.drafted            = result.speculative.drafted_tokens;
        record.accepted           = result.speculative.accepted_tokens;
        succeed(outcome);
    } catch (...) { fail(outcome); }
}

} // namespace infinitum::ninfer
