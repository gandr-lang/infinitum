# infinitum

Driver for the [infinitum](https://github.com/silvanshade-org/infinitum) inference engine.

Installs the `infinitum` binary. The driver owns the argument surface and the process boundary: it parses an invocation, renders the outcome, and leaves.

## Status

Pre-release. The engine itself is under active development in the [infinitum repository](https://github.com/silvanshade-org/infinitum). Today the driver runs infinitum's DFlash2 speculative round on the ninfer engine, for one prompt or behind an OpenAI-compatible server: infinitum composes the round and decides each round's output, and ninfer runs it as one fused round.

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

### Serving

```sh
infinitum serve --artifact path/to/model.ninfer --chat-template template.jinja --api-key "$KEY" --port 8080
```

`serve` plans the round as `generate` does, then opens the Engine with the chat template (the artifact's own when `--chat-template` is absent), runs one short warm-up request, and prints `listening on <address> as <model id>`. It answers `GET /health`, `GET /v1/models` and `POST /v1/chat/completions`, streamed or not, with tools, the reasoning channel and `chat_template_kwargs`, through [`infinitum-serve`](../serve/README.md). `--api-key` requires that key as a bearer token or `x-api-key` on every route but `/health`, and an empty key leaves the API open, as ninfer's does; `--model-id` names the model clients ask for, the artifact's model name by default. `--max-context` (default 8192) is the per-request ceiling; `--kv-capacity` (default the context ceiling) sizes the Main KV cache shared by every request, at least `--max-context`, or from device memory with `auto`; `--kv-dtype` (`bf16`, `int8`, `fp8`, `nvfp4` or `k8v4`, default `bf16`) stores it; `--prefill-chunk` (default 1024, a positive multiple of 128) sets the prompt tokens per prefill step; `--max-concurrency` (default 1, at most 8) is how many requests run at once, each on its own Engine lane, with the rest waiting for a lane; `--default-max-tokens` (default 8192) is the output limit of a request that names none, which the Engine clamps to the context left; `--pending-timeout-ms` (default 30000) bounds how long a request waits for admission; and `--max-request-mib` (default 384) caps the request body, refusing a larger one with 413 `request_too_large`. `--cuda-graph` defaults to `on`, and `--host`/`--port` default to `127.0.0.1:8080`. A client that disconnects cancels its request.

## License

Apache-2.0 WITH LLVM-exception.
