# infinitum-tenstorrent

The Tenstorrent backend of the [infinitum](https://github.com/silvanshade-org/infinitum) engine, as a spike: the `Accept` fragment under sparse rejection at temperature zero, lowered through tt-mlir's TTIR-to-TTMetal kernel path and run on a Blackhole p150a. The findings, rung by rung, are in [`REPORT.md`](REPORT.md).

## The fragment on the device

The device receives the target's logits over the anchor and `K` drafts as the `[K + 1, C, W]` block (`C` chunks of `W` logits; the served vocabulary of 248,077 ids is 970 chunks of 256, padded by 243). It returns, per column, the lowest id holding the column's maximum, and, with the prefix on the device, the accepted length and the targets up to it.

Every value the device touches is bfloat16 and every index is an integer below 1024. An id travels as three digits, the chunk index's `c / 32` and `c % 32` and the local index `id % W`; the host composes `(hi * 32 + lo) * W + local`. `digits.rs` owns the split, the composition, and the constant planes the program reads: the padding mask and the local index over `[K + 1, C, W]`, the chunk digits over `[K + 1, C]`.

`ttir/accept-k15.mlir` is the module the builder emits for `K = 15` with the prefix on the device; `ttir/accept-k15-host-prefix.mlir` returns the targets alone.

## The host reference

`reference_accept` is the answer the device must match: each column's target is the argmax over the valid vocabulary with the lowest id winning a tie, as ninfer's greedy acceptance breaks it; the accepted length is the first column whose draft differs from its target. `Sample` generates verify blocks for full acceptance, rejection at the first draft, rejection mid-block, and planted ties, with padding logits above every valid one.

## Planning

`Tenstorrent` implements `infinitum_round::Backend` fragment by fragment, and Accept under sparse rejection is its one lowering. A round plans only when every fragment lowers, so today every round is refused at its first fragment with `NoLowering`. `Tenstorrent::plan_fragment` lowers one fragment on its own. It takes the draft width from the round's DFlash2 block forward and refuses K = 4, 8, 9, 10, and 14 with `UnsupportedWidth`: at those widths the pinned pipeline aborts the process instead of reporting a diagnostic. Planning is pure Rust and builds on every host.

## The device

The `device` feature adds the C++ host (`cxx/host.{h,cpp}`, C++26, behind a `cxx` bridge): it builds the module with MLIR builder calls, runs `ttir-to-ttmetal-pipeline` and the flatbuffer translation in process, and submits the program through tt-mlir's runtime. `accept-differential` compares the device with the reference sample by sample and times every call. Its `round` command starts from infinitum's planner: it plans the canonical DFlash2 round, lowers the round's Accept fragment, runs each sample as one round on the device, and offers every device answer to an `infinitum_round::Preview` under the request's budget.

Building with `device` needs a tt-mlir build with the runtime enabled and the LLVM/MLIR toolchain it was built against. tt-mlir builds that toolchain with clang and without RTTI, so the host is compiled the same way:

```sh
export CXX=clang++
export INFINITUM_TTMLIR_SOURCE=/path/to/tt-mlir
export INFINITUM_TTMLIR_BUILD=/path/to/tt-mlir/build
export INFINITUM_TTMLIR_TOOLCHAIN=/path/to/ttmlir-toolchain
cargo build --release -p infinitum-tenstorrent --features device
```

The binary carries an rpath to tt-mlir's compiler and runtime libraries and to tt-metal's. At run time tt-metal needs `TT_METAL_RUNTIME_ROOT` (and `TT_METAL_HOME`) pointing at its source tree, for the kernels it compiles on first use:

```sh
accept-differential system-desc --out p150a.ttsys
accept-differential run --system-desc p150a.ttsys --drafts 15 --prefix device
accept-differential emit-ttir --drafts 15 > accept.mlir   # for the pipeline tools
accept-differential run --flatbuffer accept.ttm --drafts 15
accept-differential round --system-desc p150a.ttsys --drafts 15 --budget 70
```

## License

Apache-2.0 WITH LLVM-exception.
