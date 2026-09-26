#pragma once

// The C++ side of infinitum's bridge to tt-mlir.
//
// The host builds the Accept fragment as a TTIR module with MLIR builder calls, runs tt-mlir's
// `ttir-to-ttmetal-pipeline` in process, translates the result to a flatbuffer in memory, and runs
// it on a Tenstorrent device through tt-mlir's runtime. It also loads a flatbuffer the pipeline
// tools wrote, so a module lowered outside the process runs through the same device path.
//
// Every function here is noexcept. MLIR reports failure through diagnostics and tt-mlir's runtime
// by exception, and an exception crossing the Rust boundary is undefined behaviour, so each call
// is caught where it is made and reported through the Outcome the Rust side passes in.

#include "rust/cxx.h"
#include "tt/runtime/types.h"

#include <cstdint>
#include <memory>

namespace infinitum::tenstorrent {

// Shared with Rust; the bridge's generated header defines them.
struct AcceptGeometry;
struct CompileRecord;
struct Outcome;
struct RunOutput;

/// A lowered Accept fragment: one flatbuffer binary holding one program.
class Program {
public:
    /// Take ownership of a loaded binary.
    ///
    /// # Specification
    /// trivial.
    explicit Program(::tt::runtime::Binary binary) noexcept;

    ::tt::runtime::Binary binary;
};

/// One open Tenstorrent device, closed when destroyed.
class Device {
public:
    /// Take ownership of an opened mesh device.
    ///
    /// # Specification
    /// trivial.
    explicit Device(::tt::runtime::Device device) noexcept;

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&)                 = delete;
    Device& operator=(Device&&)      = delete;

    /// Close the device.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the mesh device is closed.
    /// - fails: never; a close failure is swallowed, since a destructor cannot report it.
    /// - panics: none.
    ~Device();

    ::tt::runtime::Device device;
};

/// Build the Accept module for `geometry` and print it as TTIR text.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success `text` holds the module the builder produces, verified, and
///   `outcome.status` is `Completed`; on failure `text` is unspecified and `outcome` holds the
///   verifier's diagnostics.
/// - provides: the builder's module, for comparison with the hand-written sample.
/// - fails: when `geometry` names an empty draft block, a vocabulary the chunk width does not
///   cover, or the module fails verification; reported through `outcome`, never thrown.
/// - panics: none.
void print_accept(const AcceptGeometry& geometry, rust::String& text, Outcome& outcome) noexcept;

/// Build the Accept module for `geometry`, lower it through `ttir-to-ttmetal-pipeline` against the
/// system descriptor at `system_desc`, and translate it to a flatbuffer, all in process. An empty
/// `system_desc` lowers against tt-mlir's mock single-chip Blackhole descriptor instead.
///
/// # Specification
/// - requires: a non-empty `system_desc` names a descriptor flatbuffer written for the target
///   device.
/// - ensures: on success a non-null program, `record` holding the wall time of each stage and the
///   number of device programs the lowering enqueues, and `outcome.status` `Completed`; on
///   failure a null pointer and `outcome` holding the diagnostics.
/// - provides: the in-process route from builder calls to a runnable binary.
/// - fails: when the module fails verification, the pipeline fails, or the translation fails;
///   reported through `outcome`, never thrown.
/// - panics: none.
std::unique_ptr<Program> compile_accept(const AcceptGeometry& geometry, rust::Str system_desc,
                                        CompileRecord& record, Outcome& outcome) noexcept;

/// Load a flatbuffer binary from `path`.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success a non-null program and `outcome.status` `Completed`; on failure a null
///   pointer and `outcome` holding the runtime's message.
/// - provides: the route for a module the pipeline tools lowered outside the process.
/// - fails: when the file is missing or is not a TTMetal flatbuffer; reported through `outcome`.
/// - panics: none.
std::unique_ptr<Program> load_program(rust::Str path, Outcome& outcome) noexcept;

/// Open the one visible device, with the runtime `program` needs selected.
///
/// # Specification
/// - requires: a device is visible to the process.
/// - ensures: on success a non-null device and `outcome.status` `Completed`; on failure a null
///   pointer and `outcome` holding the message.
/// - provides: the device every run submits to; destroying it closes it.
/// - fails: when no device opens; reported through `outcome`.
/// - panics: none.
std::unique_ptr<Device> open_device(const Program& program, Outcome& outcome) noexcept;

/// Write the current system descriptor to `path`.
///
/// # Specification
/// - requires: a device is visible to the process and none is open in it.
/// - ensures: on success the file holds the descriptor and `outcome.status` is `Completed`.
/// - provides: the descriptor `compile_accept` and the pipeline tools compile against.
/// - fails: when the runtime cannot query the device or the file cannot be written.
/// - panics: none.
void save_system_desc(rust::Str path, Outcome& outcome) noexcept;

/// Run `program` once on `device`.
///
/// # Specification
/// - requires: `logits`, `planes`, and `tail` hold bfloat16 bit patterns; taken in that order they
///   fill the program's inputs one after another, no input straddling two of them: `logits` the
///   `[K + 1, C, W]` block, `planes` the constant pad, local, hi, and lo planes, `tail` the
///   per-call drafts' digits, positions, `K` fill, and zeros when the prefix runs on the device.
/// - ensures: on success `output.values` holds the program's bfloat16 outputs flattened in order
///   (the targets' hi, lo, and local digits, masked past the accepted length and followed by it
///   when the prefix runs on the device), `output` the wall time of the submission to completion
///   and of the read back, and `outcome.status` is `Completed`.
/// - provides: one acceptance decision, or its target digits, computed on the device.
/// - fails: when an input is not bfloat16, the segments do not fill the inputs exactly, the
///   runtime rejects the inputs, or the submission fails; reported through `outcome`, never
///   thrown.
/// - panics: none.
void run_accept(Device& device, const Program& program, rust::Slice<const std::uint16_t> logits,
                rust::Slice<const std::uint16_t> planes, rust::Slice<const std::uint16_t> tail,
                RunOutput& output, Outcome& outcome) noexcept;

} // namespace infinitum::tenstorrent
