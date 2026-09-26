# Spike report: Accept on a Blackhole p150a through tt-mlir

This report answers issue #7. The fragment is `Accept` under `SparseRejection` at temperature zero. It was lowered through tt-mlir's TTIR-to-TTMetal kernel path and run on one Blackhole p150a, where it agrees with the host reference on every sample, at every rung. The route holds, but only with a reformulated fragment: `ttir.argmax` does not lower at the served vocabulary, and the pipeline aborts at some draft widths.

## Pins

| Component | Revision |
| --------- | -------- |
| tt-mlir | `1704bc67365a` |
| LLVM/MLIR | `4efe170d858e` (tt-mlir `env/CMakeLists.txt`) |
| tt-metal | `d04395ed862b` (tt-mlir's pin) |
| SFPI | 7.72.0 |
| p150a firmware bundle | 19.15.0 |
| KMD | 2.11.0 |

tt-mlir was built from source at the pin with the runtime enabled. The build needed these local patches:

- tt-metal: `ENABLE_DISTRIBUTED=OFF`, `WARNING_AS_ERROR=OFF`, and `-include cstdint`.
- tt-mlir: `-Werror` dropped.
- LLVM: benchmarks off.

tt-lang's wheels (`tt_lang-1.1.10.dev20260921+light`) are not usable for this route. They carry upstream MLIR's Python dialects plus ttcore, ttkernel, and ttl, but no `ttmlir-opt`, no `ttmlir-translate`, and no TTIR or D2M dialect. Their C API library contains none of `ttir-to-ttmetal-pipeline`, `d2m-fe-pipeline`, `ttir.argmax`, or `d2m.tile_argmax`.

## Expected holds and fails

| Brief item | Outcome |
| ---------- | ------- |
| Argmax and comparison lower through D2M to TTKernel and C++ and compile with SFPI | **Holds, reformulated.** `ttir.argmax` `[16, V]` lowers only up to V = 10,880. At the served V it asserts in D2M. A max/min/select formulation over bfloat16 lowers and compiles at K = 1, 2, 3, 5, 6, 7, 11, 12, 13, and 15. |
| Licensed tokens and accepted length agree with the host reference on every sample | **Holds.** 16 of 16 at K = 15 on every route, and 8 of 8 at K = 1, 7, and 12. The cases are full acceptance, rejection at the first draft, rejection mid-block, and planted argmax ties, 4 seeds each at K = 15 (2 at the other widths). |
| The served vocabulary tiles into 32×32 tiles with stated padding and masking | **Holds.** 248,077 ids become 970 chunks of 256, i.e. 248,320 logits. The last 243 are padding, masked to −∞ by a pad plane. |
| The first-mismatch prefix has no direct D2M op | **Confirmed.** D2M has no `cumsum` or `cumprod`. The run supports a fourth answer beside the brief's three: a min reduction over `where(match, K, position)` in existing ops, with no custom lowering. The host answer was measured too, and its device-to-host copy costs about 3 µs. |
| Argmax over about 248k entries may need a multi-core split and a second reduction | **Confirmed.** The formulation splits explicitly into per-chunk reductions followed by reductions across chunks. At K = 15 the pipeline emits 108 programs, 66 of them multi-core; the grids are listed under Rung 2. |

## The formulation

The logits `[K+1, V]` are viewed as `[K+1, C, W]`, with W = 256 and C = 970. Each column's target must be the lowest id holding the column's maximum. That is ninfer's tie rule in `speculative_accept_sparse_drafts`.

Two defects shape the formulation. Float32 elementwise ops round their operands to TF32, so an id above 2^11 corrupts in flight; the first formulation (ids as float32 global indices) disagreed on 16 of 16 samples. And an in-program `arrange` over dimension 2 of a rank-3 tensor is wrong.

So every index the device touches is an integer below 1024, which bfloat16 holds exactly. An id travels as three digits: `hi = (id / W) / 32`, `lo = (id / W) % 32`, and `local = id % W`. The index planes are program inputs rather than computed in-program.

1. Padding logits are replaced by −∞ through the pad plane.
2. Per chunk, the program takes the maximum and then the lowest local index holding it.
3. Per column, it takes the maximum over chunks. Among chunks holding it, it takes the lowest `hi`, then the lowest `lo`, then that chunk's local index. The host composes the id as `(hi * 32 + lo) * W + local`.
4. With the prefix on the device, the drafts arrive as the same three digits, and the bonus column carries a sentinel no target matches. The accepted length is the minimum of `where(match, K, position)` over columns, and targets past it are zeroed.

`ttir/accept-k15.mlir` is the module for K = 15 with the prefix on the device. `ttir/accept-k15-host-prefix.mlir` returns the targets alone.

## Rung 1: IR in, by the pipeline tools

The module printed by `accept-differential emit-ttir` was lowered by `ttmlir-opt --ttir-to-ttmetal-pipeline` and serialized by `ttmlir-translate --ttmetal-to-flatbuffer`. The harness then loaded the flatbuffer (`run --flatbuffer`).

| Measure | Value |
| ------- | ----- |
| Samples, K = 15, device prefix | 16 of 16 agree |
| Warm submit, 20 calls | min 20.5 ms, median 25.2 ms, max 35.3 ms |
| Readback | median 5.1 µs |

## Rung 2: IR in, from a C++ host

`cxx/host.{h,cpp}` (C++26, behind a `cxx` bridge) builds the module with MLIR builder calls against the pinned LLVM. It runs the pipeline and the flatbuffer translation in process, and submits through tt-mlir's runtime with borrowed host tensors.

| Placement of the prefix | Programs | Warm submit, 20 calls (min / median / max) | Readback, median |
| ----------------------- | -------- | ------------------------------------------ | ---------------- |
| Device | 108 | 18.6 / 22.9 / 26.2 ms | 3.6 µs |
| Host | 83 | 18.3 / 21.6 / 26.3 ms | 3.3 µs |

Keeping the prefix on the device adds 25 programs and costs no measurable time.

Core grid at K = 15 with the device prefix:

| Grid | Programs |
| ---- | -------- |
| 1×1 | 42 |
| 3×11 | 25 |
| 8×10 | 19 |
| 10×8 | 14 |
| 8×11 | 4 |
| 10×10 | 4 |

## Rung 3: from infinitum

The round crate gained `BackendName::Tenstorrent`, `RefusalReason::NoLowering`, and `RoundGraph::entries`. This crate's `Tenstorrent` implements `infinitum_round::Backend` fragment by fragment, with Accept as its one lowering.

`accept-differential round` runs the chain end to end:

1. It builds `infinitum_round::dflash2(K)` and plans it.
2. The whole round is refused at fragment #0 (context catch-up) with `NoLowering`.
3. Planned fragment by fragment, only fragment #5 (Accept) lowers.
4. The harness compiles that fragment's program at the width taken from the round, and runs each sample as one round on the device.
5. Each device answer goes to an `infinitum_round::Preview` under the request's budget, until the budget ends the request.

| Run | Outcome |
| --- | ------- |
| K = 15, budget 70 | 16 of 16 agree. 9 rounds reviewed, the ninth answered `limit 2`. The preview's 70 admitted tokens equal the reference's licensed tokens cut at 70. |
| K = 7, budget 40 | 16 of 16 agree. 9 rounds reviewed, the budget spent exactly. The admitted tokens equal the reference's. |
| K = 4 | Fragment #5 is refused with `UnsupportedWidth(4)` before the pipeline runs, so no program is compiled and the device is not touched. |

The round's wall time from Rust, at K = 15 after the first call, is 17.1–24.9 ms per round. That time covers the digit fill, submit, readback, and composition; the Rust side beyond submit is 11–17 µs.

## Wall time per call (K = 15, device prefix, in process)

| Stage | Time |
| ----- | ---- |
| Module build (builder calls) | 0.2 ms |
| `ttir-to-ttmetal-pipeline` | 259–267 ms |
| Flatbuffer translation | 24–26 ms |
| First submit, cold kernel cache (SFPI JIT) | 10.1 s |
| First submit, warm kernel cache | 147–255 ms |
| Warm submit | median 20.7–22.9 ms |
| Readback | 2.7–3.6 µs |

Warm submit medians at other widths: K = 1, 17.6 ms; K = 7, 19.0 ms; K = 12, 29.4 ms. A submit covers the host-to-device copy of the logits, the two constant `[K+1, C, W]` planes, and the `[K+1, C]` digit planes, as well as dispatch and completion. The split between copy and compute was not measured. In serving, the logits would already be on the device and the constant planes could stay resident.

## Defects and limits found at the pin

- **`ttir.argmax` over `[16, V]` bf16:** above V = 10,880 the row no longer fits L1. At V = 248,077, D2M reblocking asserts at `lib/Dialect/D2M/Utils/Utils.cpp:109` (`(oldGridShape[idx] * oldShardShape[idx]) % gridDim == 0`) and the process aborts. A full `[16, 248320]` row needs about 11.5 MB of L1 against 1.47 MB.
- **Draft-width aborts:** the same assert aborts the pipeline for the Accept formulation at K = 4, 8, 9, 10, and 14. The planner refuses those widths.
- **`d2m-to-ttnn-pipeline`:** it overflows L1 in TTNN mode. Applied to metal-mode IR, it fails with `Unsupported memref.alloc`.
- **Numerics:** float32 elementwise ops round operands to TF32, while float32 min reductions are exact. Bfloat16 `eq`, `ge`, `where`, `max`, `min`, `logical_and`, and `broadcast` are exact; bfloat16 `add` and `mul` round (LoFi).
- **`reshape`:** between `[N, 1]` and `[R, C]` it mis-maps values. `[32, 32]` → `[1024, 1]` fails a circular-buffer bound (`TT_FATAL`, CB 2048 > 1792).
- **In-program `arrange`:** it is wrong over dimension 2 of a rank-3 tensor, and `arrange * full` yields zeros.
- **Runtime:** a host tensor from `createOwnedHostTensor` fails at submit with `variant is valueless`. Borrowed host tensors work.
- **Build:** the host translation unit must build with clang and `-fno-rtti` to match tt-mlir's toolchain.

## Follow-on: where a round's time goes

This section draws on the static structure of the K = 15 program and on the warm timings above. It also uses round trips of two trivial binaries per runtime, run with `accept-differential probe`.

| Measure | Value |
| ------- | ----- |
| Host-to-device writes per round | 16, totalling 24.9 MB: logits, pad plane, and local plane at 8.2 MB each, the rest under 70 KB |
| Device programs per round | 108, with 239 kernel configurations and 203 buffer creations and deallocations |
| Host synchronisations per round | 4 `finish` commands |
| Warm submit against bytes written (K = 1, 7, 15) | about 17.0 ms fixed plus 0.19 ms per MB (about 5 GB/s over the card's PCIe Gen4 ×4 link) |
| Share at K = 15 | about 4.7 ms copying, about 17 ms program-bound (about 157 µs per program) |
| TTMetal, `ttir.add` 32×32 (4 programs) | 323–350 µs per round trip |
| TTMetal, one max over `[16, 970, 256]` (7 programs, 10.5 MB written) | 3.37 ms, within 10% of the fit |
| TTNN, `ttir.add` 32×32 | about 62 µs per round trip (inputs moved each call), about 41 µs with inputs resident |
| TTNN, the same max over the logits | about 1.35 ms with the logits copied each call, about 0.18 ms with them resident |

The TTMetal runtime rebuilds every `tt_metal::Program` (kernels, circular buffers, runtime arguments) on each submit, in `runtime/lib/ttmetal/executor.cpp`. It also cannot keep inputs on the device (`getLayout`/`toLayout` are marked TODO), so the constant planes cross PCIe on every round. The TTNN runtime keeps inputs resident and dispatches a cached op in about 22 µs. At this pin, `ttir.argmax` over the full `[16, 248077]` lowers through the TTNN backend pipeline to a single `ttnn.argmax`; that lowering was only compiled, not run.

The split of the program-bound 17 ms into host program construction and device execution was not measured. It needs the device profiler, which this build lacks.

The K-width aborts reduce to one rank-3 reduction in `d2m-grid-selection`; the standalone repro is on issue #7.

## Reproducing

The `device` feature's build variables are in [`README.md`](README.md). Every device run above is one of:

```sh
accept-differential run --system-desc p150a.ttsys --drafts 15 --prefix device --seeds 4 --warm 20
accept-differential run --system-desc p150a.ttsys --drafts 15 --prefix host --seeds 4 --warm 20
accept-differential run --flatbuffer accept.ttm --drafts 15 --seeds 4 --warm 20
accept-differential round --system-desc p150a.ttsys --drafts 15 --budget 70
```
