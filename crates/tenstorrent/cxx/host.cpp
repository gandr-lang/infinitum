#include "infinitum-tenstorrent/cxx/host.h"
#include "infinitum-tenstorrent/src/bridge.rs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"
#include "tt/runtime/runtime.h"
#include "ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Target/TTMetal/TTMetalToFlatbuffer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infinitum::tenstorrent {

namespace {

using Clock = std::chrono::steady_clock;

/// Nanoseconds elapsed since `start`.
std::uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

/// Mark `outcome` completed.
void complete(Outcome& outcome) {
    outcome.status = Status::Completed;
    outcome.message = rust::String();
}

/// Record a failure of `status` with `message`.
void fail(Outcome& outcome, Status status, const std::string& message) {
    outcome.status  = status;
    outcome.message = rust::String(message);
}

/// Record the exception in flight, naming the `stage` it escaped.
void fail_current(Outcome& outcome, std::string_view stage = {}) {
    const std::string prefix = stage.empty() ? std::string() : std::string(stage) + ": ";
    try {
        throw;
    } catch (const std::exception& error) {
        fail(outcome, Status::Runtime, prefix + error.what());
    } catch (...) {
        fail(outcome, Status::Unknown, prefix + "a non-standard exception");
    }
}

/// The shapes the Accept module is built from.
struct Shape {
    std::int64_t drafts;  // K
    std::int64_t columns; // R = K + 1: the anchor and the drafts
    std::int64_t chunk;   // W, logits of one chunk
    std::int64_t chunks;  // C, chunks per vocabulary row
};

/// The digit base of a chunk index: a chunk `c` travels as `c / 32` and `c % 32`.
constexpr std::int64_t chunk_digit = 32;

/// Every integer up to this bound is exact in bfloat16, which carries every index on the device.
constexpr std::int64_t bf16_exact = 256;

/// Derive the module shapes from `geometry`, or explain why there are none.
std::optional<Shape> shape_of(const AcceptGeometry& geometry, std::string& why) {
    const std::int64_t drafts = geometry.drafts;
    const std::int64_t vocab  = geometry.vocabulary;
    const std::int64_t chunk  = geometry.chunk;
    if (drafts < 1) {
        why = "the draft block is empty";
        return std::nullopt;
    }
    if (chunk < 1 || chunk > bf16_exact) {
        why = "the chunk width must be between 1 and 256";
        return std::nullopt;
    }
    if (vocab < 1) {
        why = "the vocabulary is empty";
        return std::nullopt;
    }
    const std::int64_t chunks = (vocab + chunk - 1) / chunk;
    if (chunks > chunk_digit * bf16_exact) {
        why = "the vocabulary needs more than 8192 chunks";
        return std::nullopt;
    }
    return Shape{drafts, drafts + 1, chunk, chunks};
}

/// Every dialect tt-mlir knows, registered once per context.
mlir::DialectRegistry& registry() {
    static mlir::DialectRegistry* instance = [] {
        auto* registry = new mlir::DialectRegistry();
        mlir::tt::registerAllDialects(*registry);
        mlir::tt::registerAllExtensions(*registry);
        return registry;
    }();
    return *instance;
}

/// A context that collects every diagnostic it emits as text.
class Session {
public:
    Session() : context(registry()), handler(&context, [this](mlir::Diagnostic& diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        stream << diagnostic.getLocation() << ": " << diagnostic << "\n";
        return mlir::success();
    }) {
        context.loadAllAvailableDialects();
    }

    mlir::MLIRContext context;
    std::string diagnostics;
    mlir::ScopedDiagnosticHandler handler;
};

/// Create `Op` at `loc` and return its single result.
template <class Op, class... Args>
mlir::Value make(mlir::OpBuilder& b, mlir::Location loc, Args&&... args) {
    return b.create<Op>(loc, std::forward<Args>(args)...).getResult();
}

/// Build the Accept module for `shape`.
///
/// Every value the device touches is bfloat16 and every index it carries is an integer below
/// 1024, exact in bfloat16: this lowering's elementwise float32 path rounds its operands to
/// TF32, which corrupts ids past 2^11, and its row-to-grid reshapes move the wrong elements, so
/// neither is used.
///
/// The logits arrive as the row-major `[R, C * W]` block viewed as `[R, C, W]`, the same bytes.
/// Positions past the vocabulary are forced to negative infinity through the `pad` plane. Stage
/// one reduces each chunk to its maximum and to the lowest local index holding it, read from the
/// `local` plane (`id % W`); stage two picks, per column, the lowest chunk holding the column's
/// maximum digit by digit (`c / 32` from `hi`, then `c % 32` from `lo`), then that chunk's local
/// index. A target id is `(hi * 32 + lo) * W + local`; the host composes it.
///
/// With the prefix on the device, the drafts arrive as the same three digits, the bonus column
/// holding a sentinel no target matches. The accepted length is the lowest column whose target
/// differs from its draft, a minimum over `where(match, K, position)`; the targets past it are
/// zeroed.
///
/// Inputs, in order: logits, pad, local `[R, C, W]`; hi, lo `[R, C]`; with the prefix on the
/// device, the drafts' hi, lo, local, then position, the `K` fill, and zeros, all `[R, 1]`.
/// Results: the targets' hi, lo, local `[R, 1]`; with the prefix on the device, masked past the
/// accepted length and followed by the accepted length `[1, 1]`.
mlir::OwningOpRef<mlir::ModuleOp> build_accept(mlir::MLIRContext& context, const Shape& shape,
                                               PrefixSite prefix) {
    namespace ttir = mlir::tt::ttir;
    const auto loc = mlir::UnknownLoc::get(&context);
    mlir::OpBuilder b(&context);
    auto module = mlir::ModuleOp::create(loc);
    b.setInsertionPointToEnd(module.getBody());

    const auto bf16 = b.getBF16Type();
    const auto f32  = b.getF32Type();
    const std::int64_t R = shape.columns, C = shape.chunks, W = shape.chunk;
    const auto block_t   = mlir::RankedTensorType::get({R, C, W}, bf16);
    const auto chunk3_t  = mlir::RankedTensorType::get({R, C, 1}, bf16);
    const auto grid_t    = mlir::RankedTensorType::get({R, C}, bf16);
    const auto column_t  = mlir::RankedTensorType::get({R, 1}, bf16);
    const auto scalar_t  = mlir::RankedTensorType::get({1, 1}, bf16);
    const bool on_device = prefix == PrefixSite::Device;

    llvm::SmallVector<mlir::Type, 11> inputs{block_t, block_t, block_t, grid_t, grid_t};
    llvm::SmallVector<mlir::Type, 4> results{column_t, column_t, column_t};
    if (on_device) {
        inputs.append(6, column_t);
        results.push_back(scalar_t);
    }
    auto fn = b.create<mlir::func::FuncOp>(loc, "accept", b.getFunctionType(inputs, results));
    mlir::Block* entry = fn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    const auto arg = [&](unsigned index) -> mlir::Value { return entry->getArgument(index); };

    const auto full = [&](mlir::RankedTensorType type, llvm::APFloat value) {
        return make<ttir::FullOp>(b, loc, type, b.getFloatAttr(f32, value));
    };
    const auto big = [&](mlir::RankedTensorType type) {
        return full(type, llvm::APFloat(1024.0f));
    };
    const auto reduce = [&](auto tag, mlir::RankedTensorType type, mlir::Value input,
                            std::int32_t dim, bool keep) {
        using Op = typename decltype(tag)::type;
        return make<Op>(b, loc, type, input, keep, b.getI32ArrayAttr({dim}));
    };
    const auto broadcast = [&](mlir::RankedTensorType type, mlir::Value input,
                               llvm::ArrayRef<std::int64_t> times) {
        return make<ttir::BroadcastOp>(b, loc, type, input, times);
    };
    struct MaxTag {
        using type = ttir::MaxOp;
    };
    struct MinTag {
        using type = ttir::MinOp;
    };

    const mlir::Value logits = arg(0), pad = arg(1), local = arg(2), hi = arg(3), lo = arg(4);

    // Stage one: per chunk, the maximum and the lowest local index holding it.
    const auto x = make<ttir::WhereOp>(
        b, loc, block_t, pad, full(block_t, llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(), true)),
        logits);
    const auto chunk_max_kept = reduce(MaxTag{}, chunk3_t, x, 2, true);
    const auto chunk_max      = reduce(MaxTag{}, grid_t, x, 2, false);
    const auto is_max =
        make<ttir::EqualOp>(b, loc, block_t, x, broadcast(block_t, chunk_max_kept, {1, 1, W}));
    const auto chunk_local = reduce(MinTag{}, grid_t,
                                    make<ttir::WhereOp>(b, loc, block_t, is_max, local, big(block_t)),
                                    2, false);

    // Stage two: per column, the lowest chunk holding the maximum, digit by digit, then its local
    // index.
    const auto column_max = reduce(MaxTag{}, column_t, chunk_max, 1, true);
    const auto holds =
        make<ttir::EqualOp>(b, loc, grid_t, chunk_max, broadcast(grid_t, column_max, {1, C}));
    const auto lowest = [&](mlir::Value candidates, mlir::Value digits) {
        return reduce(MinTag{}, column_t,
                      make<ttir::WhereOp>(b, loc, grid_t, candidates, digits, big(grid_t)), 1, true);
    };
    const auto narrow = [&](mlir::Value candidates, mlir::Value digits, mlir::Value chosen) {
        const auto same =
            make<ttir::EqualOp>(b, loc, grid_t, digits, broadcast(grid_t, chosen, {1, C}));
        return make<ttir::LogicalAndOp>(b, loc, grid_t, candidates, same);
    };
    const auto target_hi    = lowest(holds, hi);
    const auto holds_hi     = narrow(holds, hi, target_hi);
    const auto target_lo    = lowest(holds_hi, lo);
    const auto holds_lo     = narrow(holds_hi, lo, target_lo);
    const auto target_local = lowest(holds_lo, chunk_local);

    if (!on_device) {
        b.create<mlir::func::ReturnOp>(loc,
                                       mlir::ValueRange{target_hi, target_lo, target_local});
        return module;
    }

    // Acceptance: the lowest column whose target differs from its draft.
    const mlir::Value draft_hi = arg(5), draft_lo = arg(6), draft_local = arg(7);
    const mlir::Value position = arg(8), k_fill = arg(9), zeros = arg(10);
    const auto same = [&](mlir::Value target, mlir::Value draft) {
        return make<ttir::EqualOp>(b, loc, column_t, target, draft);
    };
    const auto match = make<ttir::LogicalAndOp>(
        b, loc, column_t,
        make<ttir::LogicalAndOp>(b, loc, column_t, same(target_hi, draft_hi),
                                 same(target_lo, draft_lo)),
        same(target_local, draft_local));
    const auto accepted = reduce(MinTag{}, scalar_t,
                                 make<ttir::WhereOp>(b, loc, column_t, match, k_fill, position),
                                 0, true);
    const auto keep = make<ttir::LessEqualOp>(b, loc, column_t, position,
                                              broadcast(column_t, accepted, {R, 1}));
    const auto licensed = [&](mlir::Value digits) {
        return make<ttir::WhereOp>(b, loc, column_t, keep, digits, zeros);
    };
    b.create<mlir::func::ReturnOp>(
        loc, mlir::ValueRange{licensed(target_hi), licensed(target_lo), licensed(target_local),
                              accepted});
    return module;
}

/// Build and verify the module for `geometry` in `session`, or report why not.
mlir::OwningOpRef<mlir::ModuleOp> verified_accept(Session& session, const AcceptGeometry& geometry,
                                                  Outcome& outcome) {
    std::string why;
    const auto shape = shape_of(geometry, why);
    if (!shape) {
        fail(outcome, Status::InvalidArgument, why);
        return {};
    }
    auto module = build_accept(session.context, *shape, geometry.prefix);
    if (mlir::failed(mlir::verify(module.get()))) {
        fail(outcome, Status::Diagnostics, session.diagnostics);
        return {};
    }
    return module;
}

/// The bytes `memcpy` writes when reading `host` back: TTMetal copies the description's physical
/// size, TTNN the tensor's physical volume. TTNN's description leaves the physical volume unset,
/// and TTMetal's volume is the logical one, so the larger of the two covers either runtime. Both
/// runtimes implement every call made here.
std::size_t copy_bytes(const ::tt::runtime::Tensor& host) {
    const auto desc = ::tt::runtime::getTensorDesc(host);
    return std::max(desc.sizeBytes(), std::size_t{::tt::runtime::getTensorVolume(host)} *
                                          ::tt::runtime::getTensorElementSize(host));
}

/// Append `host`'s bfloat16 values to `values`, or say why not. Only a dense output is read: one
/// whose copy holds exactly its logical values.
bool read_bf16(const ::tt::runtime::Tensor& host, rust::Vec<std::uint16_t>& values,
               std::string& why) {
    const auto desc = ::tt::runtime::getTensorDesc(host);
    if (desc.dataType != ::tt::target::DataType::BFloat16) {
        why = "an output is not bfloat16";
        return false;
    }
    std::size_t logical = 1;
    for (const auto extent : desc.shape) {
        logical *= extent;
    }
    const auto bytes = copy_bytes(host);
    if (bytes != logical * sizeof(std::uint16_t)) {
        why = "an output copies " + std::to_string(bytes) + " bytes for " +
              std::to_string(logical) + " bfloat16 values; a padded output is not read";
        return false;
    }
    std::vector<std::uint16_t> dense(logical, 0);
    ::tt::runtime::memcpy(dense.data(), host, ::tt::target::DataType::BFloat16);
    for (const std::uint16_t value : dense) {
        values.push_back(value);
    }
    return true;
}

} // namespace

Program::Program(::tt::runtime::Binary binary) noexcept : binary(std::move(binary)) {}

Device::Device(::tt::runtime::Device device) noexcept : device(std::move(device)) {}

Device::~Device() {
    try {
        ::tt::runtime::closeMeshDevice(device);
    } catch (...) {
    }
}

void print_accept(const AcceptGeometry& geometry, rust::String& text, Outcome& outcome) noexcept {
    try {
        Session session;
        auto module = verified_accept(session, geometry, outcome);
        if (!module) {
            return;
        }
        std::string printed;
        llvm::raw_string_ostream stream(printed);
        module->print(stream);
        text = rust::String(printed);
        complete(outcome);
    } catch (...) {
        fail_current(outcome);
    }
}

std::unique_ptr<Program> compile_accept(const AcceptGeometry& geometry, rust::Str system_desc,
                                        CompileRecord& record, Outcome& outcome) noexcept {
    try {
        Session session;
        auto start  = Clock::now();
        auto module = verified_accept(session, geometry, outcome);
        if (!module) {
            return nullptr;
        }
        record.build_ns = elapsed_ns(start);

        start = Clock::now();
        mlir::PassManager pm(&session.context, mlir::ModuleOp::getOperationName());
        mlir::tt::ttmetal::D2MPipelineOptions options;
        if (system_desc.empty()) {
            options.mockSystemDescArch = mlir::tt::ttcore::Arch::Blackhole;
        } else {
            options.systemDescPath = std::string(system_desc);
        }
        mlir::tt::ttmetal::createTTIRToTTMetalPipeline(pm, options);
        if (mlir::failed(pm.run(module.get()))) {
            fail(outcome, Status::Diagnostics, session.diagnostics);
            return nullptr;
        }
        record.pipeline_ns = elapsed_ns(start);

        std::uint32_t programs = 0;
        module->walk([&](mlir::Operation* op) {
            if (op->getName().getStringRef() == "ttmetal.enqueue_program") {
                ++programs;
            }
        });
        record.programs = programs;

        start     = Clock::now();
        auto blob = mlir::tt::ttmetal::translateTTMetalToFlatbuffer(module.get());
        if (!blob) {
            fail(outcome, Status::Diagnostics,
                 "flatbuffer translation failed: " + session.diagnostics);
            return nullptr;
        }
        record.translate_ns = elapsed_ns(start);

        auto program = std::make_unique<Program>(::tt::runtime::Binary(blob));
        complete(outcome);
        return program;
    } catch (...) {
        fail_current(outcome);
        return nullptr;
    }
}

std::unique_ptr<Program> load_program(rust::Str path, Outcome& outcome) noexcept {
    try {
        const std::string file(path);
        auto program = std::make_unique<Program>(::tt::runtime::Binary::loadFromPath(file.c_str()));
        complete(outcome);
        return program;
    } catch (...) {
        fail_current(outcome);
        return nullptr;
    }
}

std::unique_ptr<Device> open_device(const Program& program, Outcome& outcome) noexcept {
    try {
        ::tt::runtime::setCompatibleDeviceRuntime(program.binary);
        ::tt::runtime::MeshDeviceOptions options;
        options.meshShape = std::vector<std::uint32_t>{1, 1};
        auto device       = std::make_unique<Device>(::tt::runtime::openMeshDevice(options));
        complete(outcome);
        return device;
    } catch (...) {
        fail_current(outcome);
        return nullptr;
    }
}

void save_system_desc(rust::Str path, Outcome& outcome) noexcept {
    try {
        ::tt::runtime::setCurrentDeviceRuntime(::tt::runtime::DeviceRuntime::TTMetal);
        const std::string file(path);
        ::tt::runtime::getCurrentSystemDesc().store(file.c_str());
        complete(outcome);
    } catch (...) {
        fail_current(outcome);
    }
}

void run_accept(Device& device, const Program& program, rust::Slice<const std::uint16_t> logits,
                rust::Slice<const std::uint16_t> planes, rust::Slice<const std::uint16_t> tail,
                RunOutput& output, Outcome& outcome) noexcept {
    std::string_view stage = "reading the program's inputs";
    try {
        const auto descs = program.binary.getProgramInputs(0);
        // Each input takes the next values of the first segment it fits: the logits, then the
        // constant planes, then the per-call tail. An input never straddles two segments.
        const std::array<rust::Slice<const std::uint16_t>, 3> segments{logits, planes, tail};
        std::size_t segment = 0, offset = 0;
        stage = "creating the input tensors";
        std::vector<::tt::runtime::Tensor> inputs;
        inputs.reserve(descs.size());
        for (const auto& desc : descs) {
            if (desc.dataType != ::tt::target::DataType::BFloat16) {
                fail(outcome, Status::InvalidArgument, "every program input must be bfloat16");
                return;
            }
            while (segment < segments.size() && offset == segments[segment].size()) {
                ++segment;
                offset = 0;
            }
            if (segment == segments.size() || segments[segment].size() - offset < desc.volume()) {
                fail(outcome, Status::InvalidArgument,
                     "input " + std::to_string(inputs.size()) + " takes " +
                         std::to_string(desc.volume()) + " values the segments do not hold");
                return;
            }
            // Borrowed, not owned: the TTMetal runtime's owned host tensor wraps a bare
            // `HostBuffer` that its executor cannot read as an input. The slices outlive the
            // submit and its wait, and the runtime only reads them.
            inputs.push_back(::tt::runtime::createBorrowedHostTensor(
                const_cast<std::uint16_t*>(segments[segment].data() + offset), desc));
            offset += desc.volume();
        }
        while (segment < segments.size() && offset == segments[segment].size()) {
            ++segment;
            offset = 0;
        }
        if (segment != segments.size()) {
            fail(outcome, Status::InvalidArgument, "the segments hold values no input takes");
            return;
        }

        stage       = "submitting";
        auto start  = Clock::now();
        auto result = ::tt::runtime::submit(device.device, program.binary, 0, inputs);
        stage       = "waiting";
        ::tt::runtime::wait(result);
        output.submit_ns = elapsed_ns(start);

        stage = "reading back";
        start = Clock::now();
        output.values.clear();
        for (auto& tensor : result) {
            auto host = ::tt::runtime::toHost(tensor, true);
            if (host.size() != 1) {
                fail(outcome, Status::Runtime,
                     "an output came back as " + std::to_string(host.size()) + " host tensors");
                return;
            }
            std::string why;
            if (!read_bf16(host[0], output.values, why)) {
                fail(outcome, Status::Runtime, why);
                return;
            }
        }
        output.readback_ns = elapsed_ns(start);
        complete(outcome);
    } catch (...) {
        fail_current(outcome, stage);
    }
}

void probe_program(Device& device, const Program& program, std::uint32_t calls,
                   ProbeOutput& output, Outcome& outcome) noexcept {
    std::string_view stage = "reading the program's inputs";
    try {
        const auto descs = program.binary.getProgramInputs(0);
        std::vector<std::vector<std::byte>> storage;
        std::vector<::tt::runtime::Tensor> host;
        storage.reserve(descs.size());
        host.reserve(descs.size());
        stage = "creating the input tensors";
        for (const auto& desc : descs) {
            storage.emplace_back(desc.sizeBytes(), std::byte{0});
            host.push_back(::tt::runtime::createBorrowedHostTensor(storage.back().data(), desc));
        }
        const auto move_inputs = [&] {
            std::vector<::tt::runtime::Tensor> moved;
            moved.reserve(host.size());
            for (std::uint32_t index = 0; index < host.size(); ++index) {
                const auto layout = ::tt::runtime::getLayout(program.binary, 0, index);
                moved.push_back(::tt::runtime::toLayout(host[index], device.device, layout, true));
            }
            return moved;
        };
        std::size_t read_bytes = 0;
        const auto read_back   = [&](std::vector<::tt::runtime::Tensor>& results) {
            for (auto& tensor : results) {
                for (auto& copy : ::tt::runtime::toHost(tensor, true)) {
                    std::vector<std::byte> bytes(copy_bytes(copy), std::byte{0});
                    ::tt::runtime::memcpy(bytes.data(), copy);
                    read_bytes += bytes.size();
                }
                ::tt::runtime::deallocateTensor(tensor, true);
            }
        };
        stage = "running with the inputs moved on every call";
        for (std::uint32_t call = 0; call < calls; ++call) {
            auto start  = Clock::now();
            auto inputs = move_inputs();
            output.per_call_move.push_back(elapsed_ns(start));
            start        = Clock::now();
            auto results = ::tt::runtime::submit(device.device, program.binary, 0, inputs);
            ::tt::runtime::wait(results);
            output.per_call_submit.push_back(elapsed_ns(start));
            start = Clock::now();
            read_back(results);
            output.per_call_readback.push_back(elapsed_ns(start));
        }
        stage       = "running with the inputs moved once";
        auto staged = move_inputs();
        for (std::uint32_t call = 0; call < calls; ++call) {
            auto start   = Clock::now();
            auto results = ::tt::runtime::submit(device.device, program.binary, 0, staged);
            ::tt::runtime::wait(results);
            output.staged_submit.push_back(elapsed_ns(start));
            start = Clock::now();
            read_back(results);
            output.staged_readback.push_back(elapsed_ns(start));
        }
        output.bytes_read = read_bytes;
        complete(outcome);
    } catch (...) {
        fail_current(outcome, stage);
    }
}

} // namespace infinitum::tenstorrent
