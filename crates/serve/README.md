# infinitum-serve

The OpenAI-compatible HTTP surface of [infinitum](https://github.com/silvanshade-org/infinitum). It serves any `infinitum-chat` backend over `axum`, translating requests and rendering responses field for field as ninfer's server does, so a client moves between the two unchanged.

| Route | Answer |
| ----- | ------ |
| `GET /health` | `{"status":"ok"}`; needs no key. |
| `GET /v1/models`, `GET /v1/models/{id}` | The one served model, with its `max_model_len`. |
| `POST /v1/chat/completions` | A completion, aggregate or streamed as server-sent events. |

| Piece | Module | What it does |
| ----- | ------ | ------------ |
| Translation | `parse` | Validates a Chat Completions body and lowers it to a `ChatRequest`: turns, tools and `tool_choice`, `chat_template_kwargs` with `enable_thinking`, `preserve_thinking` and `reasoning_effort` merged, sampling for both thinking phases, seeds, stops. Prompt-cache hints become the cache markers ninfer's server derives: explicit `prompt_cache_breakpoint`s (the last four kept) and the automatic write `prompt_cache_options` selects. A body ninfer's server refuses is refused with the same parameter and code. |
| Rendering | `render` | The `chat.completion` body and the `chat.completion.chunk` stream, with `reasoning_content`, `tool_calls`, `usage` and llama.cpp-style `timings`. |
| Errors | `error` | The `{"error":{...}}` body, and the status and code each backend failure class takes. |
| Server | `server` | The API-key gate (bearer token or `x-api-key`), `x-request-id`, the configured request-body cap (413 `request_too_large` past it), and the bridge from a blocking backend to an async response: a streamed request begins its event stream once the backend has submitted it, sends a keep-alive comment every five seconds of quiet, and is cancelled when its client goes away. |
| Operational log | `oplog`, `pretty` | ninfer's stderr lines, number formats included: the startup capacity and context-cache lines, `warmup complete` for a warm-up of a quarter second or more, `listening on`, one `req#<n> started` and one `done` or `failed` line per request, and a `throughput` line each stats interval that saw activity. |
