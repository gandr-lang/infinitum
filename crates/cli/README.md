# infinitum

Driver for the [infinitum](https://github.com/silvanshade-org/infinitum) inference engine.

Installs the `infinitum` binary. The driver owns the argument surface and the process boundary: it parses an invocation, renders the outcome, and leaves.

## Status

Pre-release. The engine itself is under active development in the [infinitum repository](https://github.com/silvanshade-org/infinitum). Today the driver runs one prompt through the ninfer engine's C facade, which serves as the reference an implementation of the same model is compared against.

## Usage

```sh
infinitum generate --library path/to/libninfer_capi.so --artifact path/to/model.ninfer "The capital of France is"
```

`generate` loads the facade from the shared library named by `--library` when it runs, so building the driver needs no ninfer on the machine. It encodes the prompt as raw text with the artifact's tokenizer, generates greedily, and prints:

```text
prompt ids: <the prompt's token ids>
generated ids: <the generated token ids>
generated text:
<the bytes the generated ids render as>
```

Greedy generation is deterministic, so two runs of one prompt print the same ids. The text is written as the tokenizer's bytes, unmodified: a generation cut short by `--max-new-tokens` can end inside a multi-byte character.

`--max-context` (default 4096) bounds prompt and generation together and sizes the engine's KV cache. `--device` selects the CUDA device, and `--cuda-graph on` captures decode rounds as CUDA graphs.

## License

Apache-2.0 WITH LLVM-exception.
