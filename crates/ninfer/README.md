# infinitum-ninfer

The ninfer backend of the [infinitum](https://github.com/silvanshade-org/infinitum) engine.

## Planning

`Ninfer` implements `infinitum_round::Backend`. ninfer runs the DFlash2 round as one fused Engine round and has no fragment-by-fragment lowering, so a graph plans exactly when it is `infinitum_round::dflash2(width)` for a width from 1 to 15. Any other graph is refused with the reason: no DFlash2 drafter, a width out of range, or the first fragment where the graph leaves the canonical round. Planning is pure Rust and builds on every host.

## The Engine

The `engine` feature adds `Session`, which opens ninfer's C++ Engine through a `cxx` bridge, runs the plan, and hands the Engine a round controller that forwards every round's licensed tokens to an `infinitum_round::Preview` before the commit. infinitum makes the output decision; ninfer applies it and commits.

`ChatEngine` serves chat requests on the same Engine as an `infinitum_chat::ChatBackend`. ninfer's frontend renders the chat template (the artifact's, or a file chosen with `ChatTemplate::File`), parses reasoning and tool calls, and runs admission and the prefix cache; the adapter forwards the submission, the admission record and every committed delta to the consumer, polls the consumer's cancellation, and returns the outcome with ninfer's token accounting, phase times and speculative tallies. A refusal keeps ninfer's class — invalid prompt, context length, thinking-budget capacity, queue full, queue timeout, cancelled, unavailable — so the HTTP surface can answer as ninfer's server does.

The C++ side is `cxx/adapter.{h,cpp}`, compiled as C++26. Every function it defines is `noexcept`: ninfer reports failure by exception, and an exception crossing into Rust is undefined behaviour, so each call into ninfer is caught at the call site and returned as a status and message. The generated bridge is compiled apart from the adapter, so the one GCC diagnostic that misfires on `cxx`'s generated `rust::Vec` constructors is silenced for that translation unit alone.

Building with `engine` needs ninfer's public headers and its shared Engine library (`libninfer_engine.so`, built with `NINFER_ENGINE_SHARED=ON`):

```sh
export INFINITUM_NINFER_INCLUDE=/path/to/ninfer/include
export INFINITUM_NINFER_LIB=/path/to/ninfer/build/src/runtime
cargo build --release -p infinitum --features ninfer
```

The build gives this crate's test binaries and the `infinitum` binary an rpath to `INFINITUM_NINFER_LIB`.

## License

Apache-2.0 WITH LLVM-exception.
