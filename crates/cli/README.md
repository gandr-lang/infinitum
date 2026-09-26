# infinitum

Driver for the [infinitum](https://github.com/silvanshade-org/infinitum) inference engine.

Installs the `infinitum` binary. The driver owns the argument surface and the process boundary: it parses an invocation, renders the outcome, and leaves.

## Status

Pre-release. The engine itself is under active development in the [infinitum repository](https://github.com/silvanshade-org/infinitum). Today the driver runs one prompt through infinitum's DFlash2 speculative round on the ninfer engine: infinitum composes the round and decides each round's output, and ninfer runs it as one fused round.

## Usage

```sh
infinitum generate --artifact path/to/model.ninfer "The capital of France is"
```

`generate` composes the DFlash2 round at `--draft-width` (default 7) and has the ninfer backend plan it; a round ninfer cannot run is refused before anything opens. With the `ninfer` feature (see [`infinitum-ninfer`](../ninfer/README.md) for the build), it then opens ninfer's Engine, encodes the prompt as raw text with the artifact's tokenizer, generates greedily, and prints:

```text
prompt ids: <the prompt's token ids>
generated ids: <the generated token ids>
generated text:
<the bytes the generated ids render as>
rounds: <R> drafted: <D> accepted: <A> fallback steps: <F> wall us: <W> reviewed rounds: <N> finish: <reason>
```

Greedy generation is deterministic, so two runs of one prompt print the same ids. The text is written as the tokenizer's bytes, unmodified: a generation cut short by `--max-new-tokens` can end inside a multi-byte character. The last line is ninfer's speculative tallies, the wall time from the first to the last generated token, and the number of rounds infinitum's preview reviewed. Built without the `ninfer` feature, `generate` stops after planning and says so.

`--max-context` (default 4096) bounds prompt and generation together and sizes the engine's KV cache. `--device` selects the CUDA device, and `--cuda-graph on` captures decode rounds as CUDA graphs.

## License

Apache-2.0 WITH LLVM-exception.
