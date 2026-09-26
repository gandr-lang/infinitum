# infinitum-chat

The chat request and its outcome as [infinitum](https://github.com/silvanshade-org/infinitum) hands them to a serving backend. Protocol-neutral: an HTTP surface translates its wire format into these types, and a backend runs them.

| Piece | Type | What it is |
| ----- | ---- | ---------- |
| Conversation | `Message`, `Role`, `ToolCall` | The turns in order, each with its text parts, carried reasoning, tool calls and tool-call identity. |
| Prompt options | `Prompt`, `Switch`, `Effort` | Tool definitions as JSON, chat-template arguments, and the thinking, preserved-thinking and reasoning-effort choices a request made or left to the model. |
| Generation | `Generation`, `Sampling`, `Setting`, `Seed` | The output limit, sampling overrides for the thinking and post-thinking phases, the seed, stop strings and output options. |
| Streaming | `ChatEvents`, `Channel`, `CancelToken` | The submission, the admission record and each published text delta, in order, and the consumer's cancellation. |
| Outcome | `ChatOutcome`, `Finish`, `GeneratedToolCall`, `Telemetry` | Content, reasoning, parsed tool calls, the finish reason, token accounting, phase times and speculative tallies, and the request's timing, reuse path and thinking spend. |
| Runtime | `Capacity`, `RuntimeCounters` | The capacities a backend resolved at startup, and snapshots of its cumulative token and round totals and request gauges. |
| Startup | `StartupEvent`, `StartupObserver`, `Unobserved` | Each startup phase's beginning, byte progress, completion or failure, reported while a backend opens, and the observer that takes them. |
| Contract | `ChatBackend`, `ChatFailure` | A backend runs one request to its outcome, publishing to the events as it goes, or fails with a classified reason. |
