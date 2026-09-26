#pragma once

// The C++ side of infinitum's bridge to ninfer's Engine.
//
// Every function here is noexcept. ninfer reports failure by exception, and an exception crossing
// the Rust boundary is undefined behaviour, so each call into ninfer is caught where it is made
// and reported through the Outcome the Rust side passes in.

#include "ninfer/engine.h"
#include "rust/cxx.h"

#include <cstdint>
#include <memory>

namespace infinitum::ninfer {

// Shared with Rust; the bridge's generated header defines them.
struct CancelFlag;
struct ChatPrompt;
struct ChatRecord;
struct ChatSettings;
struct ChatSink;
struct EngineConfig;
struct GenerationRecord;
struct Outcome;
struct ReviewSink;

/// One open ninfer Engine, owned by the Rust side through the `UniquePtr` `open_session` returns.
class Session {
public:
    /// Take ownership of an opened Engine.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: `engine` is the Engine moved in.
    /// - provides: the object every other adapter function operates on.
    /// - fails: never; moving an Engine does not throw.
    /// - panics: none.
    explicit Session(::ninfer::Engine engine) noexcept;

    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&)                 = delete;
    Session& operator=(Session&&)      = delete;
    ~Session()                         = default;

    // Mutable because every Engine operation is internally synchronized: the chat path submits
    // requests from many threads through a shared Session, as ninfer's own server does.
    mutable ::ninfer::Engine engine;
};

/// Open an Engine for `config`: DFlash2 at `config.draft_width`, the optimized proposal head, an
/// explicit KV capacity equal to `config.max_context`, and `config.chat_template` when non-empty.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success a non-null session and `outcome.status` `Completed`; on failure a null
///   pointer and `outcome` holding the exception's class and message.
/// - provides: the only way to create a `Session`.
/// - fails: when ninfer throws while opening (a missing or invalid artifact, an unsupported
///   option, device or allocation failure); reported through `outcome`, never thrown.
/// - panics: none; every exception from ninfer is caught at its call site.
std::unique_ptr<Session> open_session(const EngineConfig& config, Outcome& outcome) noexcept;

/// Encode `text` with the artifact's tokenizer, adding no template or special token.
///
/// # Specification
/// - requires: `session` is open.
/// - ensures: on success `ids` holds exactly the encoding and `outcome.status` is `Completed`;
///   on failure `ids` is unspecified and `outcome` holds the exception.
/// - provides: a prompt's ids.
/// - fails: when ninfer throws; reported through `outcome`, never thrown.
/// - panics: none.
void tokenize(const Session& session, rust::Str text, rust::Vec<std::int32_t>& ids,
              Outcome& outcome) noexcept;

/// Generate greedily from `prompt`, at most `budget` tokens, offering every round to `sink` before
/// its commit.
///
/// # Specification
/// - requires: `session` is open; `sink` outlives the call.
/// - ensures: on success `record` holds ninfer's generated ids, finish reason, speculative tallies
///   and generation wall time, and `outcome.status` is `Completed`; on every exit the controller's
///   pointer to `sink` is cleared under its mutex, so the Engine never reaches `sink` afterwards
///   even when it still holds the controller.
/// - provides: the infinitum-owned DFlash2 round on ninfer, with the host decision per round.
/// - fails: when ninfer throws while preparing, submitting or waiting; reported through
///   `outcome`, never thrown.
/// - panics: none.
void generate(Session& session, rust::Slice<const std::int32_t> prompt, std::uint32_t budget,
              ReviewSink& sink, GenerationRecord& record, Outcome& outcome) noexcept;

/// Render `ids` to bytes: each id's bytes in order, special tokens included.
///
/// # Specification
/// - requires: `session` is open.
/// - ensures: on success `bytes` holds exactly the rendering and `outcome.status` is `Completed`;
///   on failure `bytes` is unspecified and `outcome` holds the exception.
/// - provides: the generated text.
/// - fails: when ninfer throws, as it does for an id outside the vocabulary; reported through
///   `outcome`, never thrown.
/// - panics: none.
void detokenize(const Session& session, rust::Slice<const std::int32_t> ids,
                rust::Vec<std::uint8_t>& bytes, Outcome& outcome) noexcept;

/// Read the model name the artifact records.
///
/// # Specification
/// - requires: `session` is open.
/// - ensures: on success `name` holds the Engine's `load_summary().model_name` and
///   `outcome.status` is `Completed`; on failure `outcome` holds the exception.
/// - provides: the default public model id.
/// - fails: when ninfer throws; reported through `outcome`, never thrown.
/// - panics: none.
void model_name(const Session& session, rust::String& name, Outcome& outcome) noexcept;

/// Run one chat request: render and prepare `prompt`, submit it with `settings`, and wait for
/// its result, publishing to `sink` when streaming and reviewing every round through `review`.
///
/// # Specification
/// - requires: `session` is open; `sink`, `cancel` and `review` outlive the call.
/// - ensures: on success `record` holds ninfer's content, reasoning, tool calls, finish reason,
///   prompt accounting, generated ids, reasoning count, phase times and speculative tallies, and
///   `outcome.status` is `Completed`; a refusal sets `Refused` and its class. `sink` is reached
///   only on the calling thread and only before return; `cancel` only through cancellation views
///   passed to calls that end before return; `review` only through a controller detached before
///   return.
/// - provides: infinitum's serving path on ninfer.
/// - fails: when ninfer refuses or throws; reported through `outcome`, never thrown.
/// - panics: none.
void run_chat(const Session& session, const ChatPrompt& prompt, const ChatSettings& settings,
              ChatSink& sink, const CancelFlag& cancel, ReviewSink& review, ChatRecord& record,
              Outcome& outcome) noexcept;

} // namespace infinitum::ninfer
