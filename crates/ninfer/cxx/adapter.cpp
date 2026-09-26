#include "infinitum-ninfer/cxx/adapter.h"

#include "infinitum-ninfer/src/bridge.rs.h"
#include "ninfer/round_control.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinitum::ninfer {
namespace {

void succeed(Outcome& outcome) noexcept {
    outcome.status = Status::Completed;
    outcome.message = rust::String();
}

// Classifies the exception in flight. Called only from a catch-all handler.
void fail(Outcome& outcome) noexcept {
    try {
        throw;
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

// Forwards each round to the Rust sink until detached. The Engine may hold the controller past
// the request's end, so the sink is reached only through a pointer the generating call clears
// before it returns.
class Controller final : public ::ninfer::RoundController {
public:
    explicit Controller(ReviewSink& sink) noexcept : sink_(&sink) {}

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

    void detach() noexcept {
        const std::lock_guard lock(mutex_);
        sink_ = nullptr;
    }

private:
    std::mutex mutex_;
    ReviewSink* sink_;
};

// Detaches the controller on every exit from generate.
class Detach {
public:
    explicit Detach(Controller& controller) noexcept : controller_(controller) {}
    Detach(const Detach&)            = delete;
    Detach& operator=(const Detach&) = delete;
    Detach(Detach&&)                 = delete;
    Detach& operator=(Detach&&)      = delete;
    ~Detach() { controller_.detach(); }

private:
    Controller& controller_;
};

// The greedy request of ninfer's own C facade: argmax, no penalties, the model's stop tokens.
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

std::uint64_t nanoseconds(double seconds) noexcept {
    if (!(seconds > 0.0)) { return 0; }
    return static_cast<std::uint64_t>(seconds * 1e9);
}

} // namespace

Session::Session(::ninfer::Engine engine_) noexcept : engine(std::move(engine_)) {}

std::unique_ptr<Session> open_session(const EngineConfig& config, Outcome& outcome) noexcept {
    try {
        ::ninfer::EngineOptions options;
        options.artifact_path = std::string(config.artifact);
        options.device        = config.device;
        options.max_context   = config.max_context;
        options.kv_capacity   = ::ninfer::KvCapacityPolicy::explicit_capacity(config.max_context);
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

} // namespace infinitum::ninfer
