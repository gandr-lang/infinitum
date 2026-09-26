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
struct EngineConfig;
struct GenerationRecord;
struct Outcome;
struct ReviewSink;

// One open ninfer Engine.
class Session {
public:
    explicit Session(::ninfer::Engine engine) noexcept;

    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&)                 = delete;
    Session& operator=(Session&&)      = delete;
    ~Session()                         = default;

    ::ninfer::Engine engine;
};

// Opens an Engine for the config. On failure the pointer is null and the outcome says why.
std::unique_ptr<Session> open_session(const EngineConfig& config, Outcome& outcome) noexcept;

// Encodes raw text with the artifact's tokenizer into ids, replacing their contents.
void tokenize(const Session& session, rust::Str text, rust::Vec<std::int32_t>& ids,
              Outcome& outcome) noexcept;

// Generates greedily from the prompt within the budget. The sink reviews every round before its
// commit, on the Engine's worker thread; it is not touched after this returns.
void generate(Session& session, rust::Slice<const std::int32_t> prompt, std::uint32_t budget,
              ReviewSink& sink, GenerationRecord& record, Outcome& outcome) noexcept;

// Renders ids back to bytes, replacing the contents of bytes.
void detokenize(const Session& session, rust::Slice<const std::int32_t> ids,
                rust::Vec<std::uint8_t>& bytes, Outcome& outcome) noexcept;

} // namespace infinitum::ninfer
