//! The operational log: one line per request start, end or failure, in
//! ninfer's server's words and number formats, so the journal reads the same
//! from either server.

use core::fmt::Write as _;

use infinitum_chat::Admission;
use infinitum_chat::Channel;
use infinitum_chat::ChatEvents;
use infinitum_chat::ChatOutcome;
use infinitum_chat::ChatRequest;
use infinitum_chat::Delivery;
use infinitum_chat::DeltaText;
use infinitum_chat::Effort;
use infinitum_chat::Finish;
use infinitum_chat::ReusePath;
use infinitum_chat::Submission;
use infinitum_chat::Switch;
use infinitum_chat::Thinking;
use infinitum_chat::ThinkingBudget;
use infinitum_round::TokenCount;

use crate::error::ApiError;
pub use crate::pretty::Bytes;
pub use crate::pretty::Count;
use crate::pretty::Duration;
use crate::pretty::Percent;
use crate::pretty::Text;
use crate::pretty::TokenRate;

/// A request's number in the log, counted from one.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestId(pub u64);

/// A fixed name a log line spells.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Label(&'static str);

/// How severe a log line is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity
{
    /// Routine.
    Info,
    /// Degraded service.
    Warning,
    /// A server fault.
    Error,
}

/// One rendered log line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record
{
    /// Its severity.
    pub severity: Severity,
    /// Its text, without timestamp or level.
    pub message: String,
}

/// What the log knows of a request before it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestShape
{
    /// Its number.
    pub id: RequestId,
    /// Streamed or aggregate.
    pub delivery: Delivery,
    /// Messages in the conversation.
    pub messages: Count,
    /// Tools offered.
    pub tools: Count,
    /// The output limit.
    pub max_output: TokenCount,
    /// The reasoning effort asked for.
    pub effort: Effort,
    /// Whether closed turns keep their reasoning.
    pub preserve_thinking: Switch,
}

/// Where in its life a request failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase
{
    /// Before submission.
    Prepare,
    /// After submission.
    Generation,
}

/// ninfer's failure class for an API error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class
{
    /// The client's request was refused.
    ClientInput,
    /// The client went away.
    ClientDisconnected,
    /// The server was full.
    Overload,
    /// A deadline passed.
    Timeout,
    /// The service is unavailable.
    Unavailable,
    /// An upstream fetch failed.
    Upstream,
    /// A server fault.
    Internal,
}

impl Class
{
    /// Classify `error` as ninfer's `make_request_failure` does.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: 499 or `client_disconnected` is a disconnect; 429 or 529
    ///   overload; `request_queue_timeout`, `media_fetch_timeout` or 504 a
    ///   timeout; `service_unavailable` or 503 unavailable;
    ///   `media_fetch_failed` or 502 upstream; any other 4xx client input;
    ///   everything else internal; the first matching rule wins.
    /// - provides: every failure line's class.
    /// - fails: never.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 on each rule's representative.
    /// - witness: `tests::failure_lines_follow_ninfer`
    fn of(error: &ApiError) -> Self
    {
        let status = error.status.as_u16();
        let code = error.code.as_str();
        if status == 499 || code == "client_disconnected" {
            return Self::ClientDisconnected;
        }
        if status == 429 || status == 529 {
            return Self::Overload;
        }
        if matches!(code, "request_queue_timeout" | "media_fetch_timeout") || status == 504 {
            return Self::Timeout;
        }
        if code == "service_unavailable" || status == 503 {
            return Self::Unavailable;
        }
        if code == "media_fetch_failed" || status == 502 {
            return Self::Upstream;
        }
        if (400 .. 500).contains(&status) {
            return Self::ClientInput;
        }
        return Self::Internal;
    }

    /// ninfer's name for the class.
    ///
    /// # Specification
    /// trivial.
    const fn name(self) -> Label
    {
        return match self {
            | Self::ClientInput => Label("client input"),
            | Self::ClientDisconnected => Label("client disconnected"),
            | Self::Overload => Label("overload"),
            | Self::Timeout => Label("timeout"),
            | Self::Unavailable => Label("unavailable"),
            | Self::Upstream => Label("upstream error"),
            | Self::Internal => Label("internal error"),
        };
    }

    /// ninfer's severity for the class.
    ///
    /// # Specification
    /// trivial.
    const fn severity(self) -> Severity
    {
        return match self {
            | Self::ClientInput | Self::ClientDisconnected => Severity::Info,
            | Self::Overload | Self::Timeout | Self::Unavailable | Self::Upstream => {
                Severity::Warning
            },
            | Self::Internal => Severity::Error,
        };
    }
}

/// Append formatted text to a line.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `args` rendered at the end of `out`; writing to a `String` cannot
///   fail, so the formatter's result carries nothing.
/// - provides: every line's clauses.
/// - fails: never.
/// - panics: none.
fn put(
    out: &mut String,
    args: core::fmt::Arguments<'_>,
)
{
    out.write_fmt(args).unwrap_or(());
}

/// ninfer's protocol name for Chat Completions.
const PROTOCOL: &str = "openai-chat";

/// ninfer's name for a delivery.
///
/// # Specification
/// trivial.
const fn delivery_name(delivery: Delivery) -> Label
{
    return match delivery {
        | Delivery::Streaming => Label("stream"),
        | Delivery::Aggregate => Label("non-stream"),
    };
}

/// Append ninfer's HTTP status and code, or class, to a failure line.
///
/// # Specification
/// - requires: nothing.
/// - ensures: ` | HTTP <status>` then ` | ` and the code with `_` as spaces, or
///   the class name when the code is empty, as ninfer's
///   `append_failure_fields`.
/// - provides: every failure line's tail.
/// - fails: never.
/// - panics: none.
fn failure_fields(
    out: &mut String,
    error: &ApiError,
)
{
    put(out, format_args!(" | HTTP {}", error.status.as_u16()));
    if error.code.is_empty() {
        put(out, format_args!(" | {}", Class::of(error).name().0));
    }
    else {
        put(out, format_args!(" | {}", error.code.replace('_', " ")));
    }
}

/// The line for a request submitted to the backend.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `req#<id> started | openai-chat <stream|non-stream> | <n>
///   message(s) | max output <n> | thinking <effort, or template default>`,
///   with `, budget <n>` under a budget, or `thinking off` when the turn does
///   not open in thinking; then ` | tools <n>` when tools are offered and ` |
///   preserve thinking` when asked, as ninfer's `render_request_start`.
/// - provides: the start record, logged at submission.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 on thinking on with a budget, off, and the optional
///   clauses.
/// - witness: `tests::request_lines_follow_ninfer`
#[inline]
#[must_use]
pub fn started(
    shape: &RequestShape,
    submission: Submission,
) -> Record
{
    let mut out = format!(
        "req#{} started | {PROTOCOL} {} | {} {} | max output {}",
        shape.id.0,
        delivery_name(shape.delivery).0,
        shape.messages,
        if shape.messages == Count(1) {
            "message"
        }
        else {
            "messages"
        },
        Count(u64::from(u32::from(shape.max_output)))
    );
    out.push_str(" | thinking ");
    match submission.thinking {
        | Thinking::Open => {
            out.push_str(effort_name(shape.effort).0);
            if let ThinkingBudget::Tokens(budget) = submission.thinking_budget {
                put(
                    &mut out,
                    format_args!(
                        ", budget {}",
                        Count(core::num::NonZeroU64::from(budget).get())
                    ),
                );
            }
        },
        | Thinking::Closed => out.push_str("off"),
    }
    if shape.tools != Count(0) {
        put(&mut out, format_args!(" | tools {}", shape.tools));
    }
    if shape.preserve_thinking == Switch::On {
        out.push_str(" | preserve thinking");
    }
    return Record {
        severity: Severity::Info,
        message: out,
    };
}

/// ninfer's name for a requested effort.
///
/// # Specification
/// trivial.
const fn effort_name(effort: Effort) -> Label
{
    return match effort {
        | Effort::Unrequested => Label("template default"),
        | Effort::None => Label("none"),
        | Effort::Minimal => Label("minimal"),
        | Effort::Low => Label("low"),
        | Effort::Medium => Label("medium"),
        | Effort::High => Label("high"),
        | Effort::XHigh => Label("xhigh"),
        | Effort::Max => Label("max"),
    };
}

/// ninfer's name for a finish.
///
/// # Specification
/// trivial.
const fn finish_name(finish: Finish) -> Label
{
    return match finish {
        | Finish::Unfinished => Label("none"),
        | Finish::OutputLimit => Label("output limit"),
        | Finish::ContextCapacity => Label("context capacity"),
        | Finish::StopToken => Label("stop token"),
        | Finish::StopString => Label("stop string"),
        | Finish::Cancelled => Label("cancelled"),
    };
}

/// ninfer's name for a reuse path.
///
/// # Specification
/// trivial.
const fn reuse_name(reuse: ReusePath) -> Label
{
    return match reuse {
        | ReusePath::Root => Label("root"),
        | ReusePath::PrivateEndpoint => Label("private endpoint"),
        | ReusePath::TurnClosure => Label("turn closure"),
        | ReusePath::ResponseReplay => Label("response replay"),
        | ReusePath::LongAnchor => Label("long anchor"),
        | ReusePath::SharedPrefix => Label("shared prefix"),
    };
}

/// The line for a request that finished.
///
/// # Specification
/// - requires: `shape` is the request's.
/// - ensures: `req#<id> done | openai-chat | <finish, or tool calls <n>> |
///   prompt <n> | output <n> | cache <n> (<pct>[, <path>]) | TTFT <d> | total
///   <d>`, then ` | queue <d>` from 10 ms, ` | prefill <rate>` over the
///   computed prompt tokens and ` | decode <rate>` over all but the first
///   output token when their times are positive, ` | dflash2 accepted <a>/<d>
///   (<pct>)` when tokens were drafted, ` | thinking <m>/<b>[, control <i>]`
///   under a budget, as ninfer's `render_request_done`; then infinitum's ` |
///   per position <a1>/<a2>/...` when positions were tallied.
/// - provides: the end record.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 on every optional clause present and absent.
/// - witness: `tests::request_lines_follow_ninfer`
#[inline]
pub fn done(
    shape: &RequestShape,
    outcome: &ChatOutcome,
) -> Record
{
    let telemetry = &outcome.telemetry;
    let prompt = outcome.admission.prompt_tokens;
    let reused = outcome.admission.reused_tokens;
    let output = outcome.generated.len();
    let mut out = format!("req#{} done | {PROTOCOL} | ", shape.id.0);
    if outcome.tool_calls.is_empty() {
        out.push_str(finish_name(outcome.finish).0);
    }
    else {
        put(
            &mut out,
            format_args!(
                "tool calls {}",
                Count(outcome.tool_calls.len().try_into().unwrap_or(u64::MAX))
            ),
        );
    }
    let tokens = |count: TokenCount| return f64::from(u32::from(count));
    let ratio = if tokens(prompt) > 0.0_f64 {
        tokens(reused) / tokens(prompt)
    }
    else {
        0.0_f64
    };
    put(
        &mut out,
        format_args!(
            " | prompt {} | output {} | cache {} ({}",
            Count(u64::from(u32::from(prompt))),
            Count(output.try_into().unwrap_or(u64::MAX)),
            Count(u64::from(u32::from(reused))),
            Percent(ratio)
        ),
    );
    if telemetry.reuse != ReusePath::Root {
        put(
            &mut out,
            format_args!(", {}", reuse_name(telemetry.reuse).0),
        );
    }
    put(
        &mut out,
        format_args!(
            ") | TTFT {} | total {}",
            Duration(telemetry.first_token.as_secs_f64()),
            Duration(telemetry.total.as_secs_f64())
        ),
    );
    if telemetry.queue_wait.as_secs_f64() >= 0.01_f64 {
        put(
            &mut out,
            format_args!(" | queue {}", Duration(telemetry.queue_wait.as_secs_f64())),
        );
    }
    let prefill = telemetry.prefill.as_secs_f64();
    if prefill > 0.0_f64 {
        let computed = (tokens(prompt) - tokens(reused)).max(0.0_f64);
        put(
            &mut out,
            format_args!(" | prefill {}", TokenRate(computed / prefill)),
        );
    }
    let decode = telemetry.decode.as_secs_f64();
    if decode > 0.0_f64 {
        let decoded = u32::try_from(output.saturating_sub(1))
            .map_or_else(|_wide| return f64::from(u32::MAX), f64::from);
        put(
            &mut out,
            format_args!(" | decode {}", TokenRate(decoded / decode)),
        );
    }
    if outcome.drafted.0 != 0 {
        let accepted = u32::try_from(outcome.accepted.0)
            .map_or_else(|_wide| return f64::from(u32::MAX), f64::from);
        let drafted = u32::try_from(outcome.drafted.0)
            .map_or_else(|_wide| return f64::from(u32::MAX), f64::from);
        put(
            &mut out,
            format_args!(
                " | dflash2 accepted {}/{} ({})",
                Count(outcome.accepted.0),
                Count(outcome.drafted.0),
                Percent(accepted / drafted)
            ),
        );
    }
    if let ThinkingBudget::Tokens(budget) = telemetry.thinking.budget {
        put(
            &mut out,
            format_args!(
                " | thinking {}/{}",
                Count(u64::from(u32::from(telemetry.thinking.model_tokens))),
                Count(core::num::NonZeroU64::from(budget).get())
            ),
        );
        let injected = u32::from(telemetry.thinking.injected_tokens);
        if injected != 0 {
            put(
                &mut out,
                format_args!(", control {}", Count(u64::from(injected))),
            );
        }
    }
    if !telemetry.accepted_per_position.is_empty() {
        out.push_str(" | per position ");
        for (index, accepted) in telemetry.accepted_per_position.iter().enumerate() {
            if index != 0 {
                out.push('/');
            }
            put(&mut out, format_args!("{}", Count(accepted.0)));
        }
    }
    return Record {
        severity: Severity::Info,
        message: out,
    };
}

/// The line for a request that failed.
///
/// # Specification
/// - requires: `shape` is the request's.
/// - ensures: before submission, `req#<id> <rejected|cancelled|failed> during
///   prepare | openai-chat <stream|non-stream>` with the failure fields, ` |
///   messages <n>` and ` | tools <n>` when tools were offered, as ninfer's
///   `render_request_rejected`, rejected for client input or overload and
///   cancelled for a disconnect; after it, `req#<id> <failed | cancelled>
///   during <generation | transport> | openai-chat` with the failure fields, as
///   ninfer's `render_request_failure`. The severity is the class's.
/// - provides: the failure record.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 on each phase and verb.
/// - witness: `tests::failure_lines_follow_ninfer`
#[inline]
#[must_use]
pub fn failed(
    shape: &RequestShape,
    phase: Phase,
    error: &ApiError,
) -> Record
{
    let class = Class::of(error);
    let mut out = String::new();
    match phase {
        | Phase::Prepare => {
            let verb = match class {
                | Class::ClientDisconnected => "cancelled",
                | Class::ClientInput | Class::Overload => "rejected",
                | Class::Timeout | Class::Unavailable | Class::Upstream | Class::Internal => {
                    "failed"
                },
            };
            put(
                &mut out,
                format_args!(
                    "req#{} {verb} during prepare | {PROTOCOL} {}",
                    shape.id.0,
                    delivery_name(shape.delivery).0
                ),
            );
            failure_fields(&mut out, error);
            put(&mut out, format_args!(" | messages {}", shape.messages));
            if shape.tools != Count(0) {
                put(&mut out, format_args!(" | tools {}", shape.tools));
            }
        },
        | Phase::Generation => {
            let (verb, during) = if class == Class::ClientDisconnected {
                ("cancelled", "transport")
            }
            else {
                ("failed", "generation")
            };
            put(
                &mut out,
                format_args!("req#{} {verb} during {during} | {PROTOCOL}", shape.id.0),
            );
            failure_fields(&mut out, error);
        },
    }
    return Record {
        severity: class.severity(),
        message: out,
    };
}

impl RequestShape
{
    /// The shape of `request`, numbered `id`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the delivery, message and offered-tool counts, output limit,
    ///   effort and preserve-thinking choice are `request`'s.
    /// - provides: every line's request facts.
    /// - fails: never.
    /// - panics: none.
    #[must_use]
    #[inline]
    pub fn of(
        id: RequestId,
        request: &ChatRequest,
    ) -> Self
    {
        let count = |length: usize| return Count(u64::try_from(length).unwrap_or(u64::MAX));
        return Self {
            id,
            delivery: request.delivery,
            messages: count(request.prompt.messages.len()),
            tools: count(request.prompt.tools.len()),
            max_output: request.generation.output_tokens,
            effort: request.prompt.effort,
            preserve_thinking: request.prompt.preserve_thinking,
        };
    }
}

/// Write `record` to standard error as one service log line, stamped now.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the [`service_line`] for `record` at the current local time is
///   written to standard error, or nothing when it is closed.
/// - provides: the operational log's sink.
/// - fails: never; a closed standard error drops the line, as a logger must not
///   fail the request it describes.
/// - panics: none.
#[inline]
pub fn emit(record: &Record)
{
    use std::io::Write as _;

    let _dropped = writeln!(
        std::io::stderr().lock(),
        "{}",
        service_line(&jiff::Zoned::now(), record)
    );
}

/// One service log line for `record` stamped at `at`.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `YYYY-MM-DD HH:MM:SS.mmm`, the civil time at `at` with its
///   milliseconds truncated, two spaces, the level padded to five characters
///   (`INFO `, `WARN `, `ERROR`), a space and the message, as ninfer's service
///   presentation formats a line.
/// - provides: [`emit`]'s line.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 on each level and a stamp one nanosecond short of the next
///   millisecond.
/// - witness: `tests::lines_carry_ninfers_stamp_and_level`
fn service_line(
    at: &jiff::Zoned,
    record: &Record,
) -> String
{
    let level = match record.severity {
        | Severity::Info => "INFO ",
        | Severity::Warning => "WARN ",
        | Severity::Error => "ERROR",
    };
    return format!(
        "{}  {level} {}",
        at.strftime("%Y-%m-%d %H:%M:%S%.3f"),
        record.message
    );
}

/// A request's events, logging its start at submission and forwarding every
/// event to the wrapped consumer.
pub struct Logged<'events>
{
    /// The consumer.
    inner: &'events mut dyn ChatEvents,
    /// The request.
    shape: RequestShape,
    /// Whether the request was submitted.
    phase: Phase,
}

impl<'events> Logged<'events>
{
    /// Log `shape`'s events around `inner`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    pub fn new(
        inner: &'events mut dyn ChatEvents,
        shape: RequestShape,
    ) -> Self
    {
        return Self {
            inner,
            shape,
            phase: Phase::Prepare,
        };
    }

    /// Log the request's end: its done line on success, else its failure
    /// line in the phase it reached.
    ///
    /// # Specification
    /// - requires: the request ran through these events.
    /// - ensures: [`done`]'s line after `Ok`; [`failed`]'s line for the phase
    ///   reached, before or after submission, after `Err`.
    /// - provides: every request's end record.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    pub fn end(
        &self,
        result: Result<&ChatOutcome, &ApiError>,
    )
    {
        match result {
            | Ok(outcome) => emit(&done(&self.shape, outcome)),
            | Err(error) => emit(&failed(&self.shape, self.phase, error)),
        }
    }
}

impl ChatEvents for Logged<'_>
{
    /// Log the start line and forward.
    ///
    /// # Specification
    /// - requires: as [`ChatEvents::submitted`].
    /// - ensures: [`started`]'s line was emitted, later failures are logged
    ///   after submission, and the consumer was told.
    /// - provides: the start record at ninfer's point, once preparation has
    ///   decided whether the turn thinks.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    fn submitted(
        &mut self,
        submission: Submission,
    )
    {
        emit(&started(&self.shape, submission));
        self.phase = Phase::Generation;
        self.inner.submitted(submission);
    }

    /// Forward.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn admitted(
        &mut self,
        admission: Admission,
    )
    {
        self.inner.admitted(admission);
    }

    /// Forward.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn publish(
        &mut self,
        channel: Channel,
        text: DeltaText<'_>,
    )
    {
        self.inner.publish(channel, text);
    }
}

/// The startup capacity lines.
///
/// # Specification
/// - requires: nothing.
/// - ensures: two lines as ninfer's `engine_capacity`: the KV line `capacity |
///   KV <n> tokens, <storage>, <auto|explicit> | pages <p>/<m> | runtime
///   <bytes> | free <bytes>` and the cache line, `context cache | root only`
///   or, with the cache's counts unseparated, `context cache | <l> active + <d>
///   cached device states | host <h> states, <bytes> KV | private <p> | shared
///   <s> | anchors <a>`.
/// - provides: the startup record of what the Engine resolved.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 on both cache forms.
/// - witness: `tests::startup_lines_follow_ninfer`
#[must_use]
#[inline]
pub fn capacity(summary: &infinitum_chat::Capacity) -> [Record; 2]
{
    let sizing = match summary.kv_sizing {
        | infinitum_chat::KvSizing::Explicit => "explicit",
        | infinitum_chat::KvSizing::Automatic => "auto",
    };
    let first = format!(
        "capacity | KV {} tokens, {}, {sizing} | pages {}/{} | runtime {} | free {}",
        Count(summary.kv_tokens.0),
        summary.kv_storage,
        Count(summary.pages.0),
        Count(summary.max_pages.0),
        Bytes(summary.runtime.0),
        Bytes(summary.free.0)
    );
    let second = match summary.cache {
        | infinitum_chat::ContextCache::RootOnly => String::from("context cache | root only"),
        | infinitum_chat::ContextCache::Enabled {
            lanes,
            device_states,
            host_states,
            host_kv,
            private,
            shared,
            anchors,
        } => format!(
            "context cache | {} active + {} cached device states | host {} states, {} KV | \
             private {} | shared {} | anchors {}",
            lanes.0,
            device_states.0,
            host_states.0,
            Bytes(host_kv.0),
            private.0,
            shared.0,
            anchors.0
        ),
    };
    return [
        Record {
            severity: Severity::Info,
            message: first,
        },
        Record {
            severity: Severity::Info,
            message: second,
        },
    ];
}

/// An interval's throughput record, when the interval saw activity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Throughput
{
    /// Nothing moved; ninfer writes no line.
    Quiet,
    /// The line to write.
    Active(Record),
}

/// The throughput line for the interval from `previous` to `current`.
///
/// # Specification
/// - requires: `current` was read no earlier than `previous`, `interval` apart.
/// - ensures: [`Throughput::Quiet`] when no token total or decode round moved,
///   no request is running, waiting, materializing or pending a capture or
///   terminal record, and the digests agree, as ninfer's `report_has_activity`;
///   otherwise `throughput | <interval>`, then `prefill <rate> (<n> tok)` and
///   `decode <rate> (<n> tok)` when nonzero, `running <n>` with its prefill and
///   decode-ready split when either is nonzero, the nonzero waiting,
///   materializing, capture-pending and terminal-pending gauges, `batch
///   <rows/rounds>` to two places when rounds ran, and `host <percent>
///   (<duration>)`, clause by clause as ninfer's `render_throughput`.
/// - provides: the periodic throughput record.
/// - fails: never; totals that went backwards read as zero.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 on the quiet interval, a digest-only change and a busy
///   interval.
/// - witness: `tests::throughput_lines_follow_ninfer`
#[inline]
#[must_use]
pub fn throughput(
    previous: &infinitum_chat::RuntimeCounters,
    current: &infinitum_chat::RuntimeCounters,
    interval: core::time::Duration,
) -> Throughput
{
    let prefill = current
        .prefill_tokens
        .0
        .saturating_sub(previous.prefill_tokens.0);
    let decode = current
        .decode_tokens
        .0
        .saturating_sub(previous.decode_tokens.0);
    let rounds = current
        .decode_rounds
        .0
        .saturating_sub(previous.decode_rounds.0);
    let rows = current.decode_rows.0.saturating_sub(previous.decode_rows.0);
    let gauges = current.requests;
    let active = prefill != 0
        || decode != 0
        || rounds != 0
        || gauges.running.0 != 0
        || gauges.waiting.0 != 0
        || gauges.materializing.0 != 0
        || gauges.capture_pending.0 != 0
        || gauges.terminal_pending.0 != 0
        || current.digest != previous.digest;
    if !active {
        return Throughput::Quiet;
    }
    let seconds = interval.as_secs_f64();
    let clamped = |count: u64| {
        return u32::try_from(count).map_or_else(|_wide| return f64::from(u32::MAX), f64::from);
    };
    let per_second = |count: u64| {
        return if seconds > 0.0_f64 {
            clamped(count) / seconds
        }
        else {
            0.0_f64
        };
    };
    let mut out = format!("throughput | {}", Duration(seconds));
    if prefill != 0 {
        put(
            &mut out,
            format_args!(
                " | prefill {} ({} tok)",
                TokenRate(per_second(prefill)),
                Count(prefill)
            ),
        );
    }
    if decode != 0 {
        put(
            &mut out,
            format_args!(
                " | decode {} ({} tok)",
                TokenRate(per_second(decode)),
                Count(decode)
            ),
        );
    }
    put(&mut out, format_args!(" | running {}", gauges.running.0));
    match (gauges.prefilling.0, gauges.decode_ready.0) {
        | (0, 0) => {},
        | (prefilling, 0) => put(&mut out, format_args!(" (prefill {prefilling})")),
        | (0, ready) => put(&mut out, format_args!(" (decode-ready {ready})")),
        | (prefilling, ready) => put(
            &mut out,
            format_args!(" (prefill {prefilling}, decode-ready {ready})"),
        ),
    }
    for (label, gauge) in [
        ("waiting", gauges.waiting),
        ("materializing", gauges.materializing),
        ("capture-pending", gauges.capture_pending),
        ("terminal-pending", gauges.terminal_pending),
    ] {
        if gauge.0 != 0 {
            put(&mut out, format_args!(" | {label} {}", gauge.0));
        }
    }
    if rounds != 0 {
        put(
            &mut out,
            format_args!(" | batch {:.2}", clamped(rows) / clamped(rounds)),
        );
    }
    let host = current
        .host_active
        .saturating_sub(previous.host_active)
        .as_secs_f64();
    let ratio = if seconds > 0.0_f64 {
        host / seconds
    }
    else {
        0.0_f64
    };
    put(
        &mut out,
        format_args!(" | host {} ({})", Percent(ratio), Duration(host)),
    );
    return Throughput::Active(Record {
        severity: Severity::Info,
        message: out,
    });
}

/// Log the start of opening the Engine.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `starting engine` is written at info, as ninfer's startup log
///   writes it when the Engine phase begins.
/// - provides: the first startup line.
/// - fails: never; a failed write is dropped as every log write is.
/// - panics: none.
#[inline]
pub fn log_engine_start()
{
    emit(&Record {
        severity: Severity::Info,
        message: String::from("starting engine"),
    });
}

/// The line for an Engine that finished loading.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `engine ready | <model> | total <duration> | weights <bytes> |
///   CUDA sync <mode>`, the model made safe for one line, as ninfer's
///   `engine_ready`.
/// - provides: the load record.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 on one load.
/// - witness: `tests::startup_lines_follow_ninfer`
#[inline]
#[must_use]
pub fn engine_ready(
    model: &infinitum_chat::ModelName,
    load: &infinitum_chat::LoadReport,
) -> Record
{
    return Record {
        severity: Severity::Info,
        message: format!(
            "engine ready | {} | total {} | weights {} | CUDA sync {}",
            Text(&model.0),
            Duration(load.total.as_secs_f64()),
            Bytes(load.weights.0),
            load.cuda_sync
        ),
    };
}

/// Log the Engine's load.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the [`engine_ready`] line is written.
/// - provides: the server's load report at startup.
/// - fails: never; a failed write is dropped as every log write is.
/// - panics: none.
#[inline]
pub fn log_engine_ready(
    model: &infinitum_chat::ModelName,
    load: &infinitum_chat::LoadReport,
)
{
    emit(&engine_ready(model, load));
}

/// Log the startup capacity lines.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the two [`capacity`] lines are written, in order.
/// - provides: the server's capacity report at startup.
/// - fails: never; a failed write is dropped as every log write is.
/// - panics: none.
#[inline]
pub fn log_capacity(summary: &infinitum_chat::Capacity)
{
    for record in capacity(summary) {
        emit(&record);
    }
}

/// The line for a server ready to accept.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `listening on http://<address> | model <id> | auth
///   <bearer|disabled>`, the id made safe for one line, as ninfer's
///   `server_ready`.
/// - provides: the ready record.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 on both access forms.
/// - witness: `tests::startup_lines_follow_ninfer`
#[must_use]
#[inline]
pub fn listening(
    address: core::net::SocketAddr,
    model: &crate::server::ModelId,
    access: &crate::server::Access,
) -> Record
{
    let auth = match *access {
        | crate::server::Access::Open => "disabled",
        | crate::server::Access::Key(_) => "bearer",
    };
    return Record {
        severity: Severity::Info,
        message: format!(
            "listening on http://{address} | model {} | auth {auth}",
            Text(&model.0)
        ),
    };
}

/// Tests for the lines against ninfer's.
#[cfg(test)]
mod tests
{
    use axum::http::StatusCode;
    use infinitum_chat::Admission;
    use infinitum_chat::ChatOutcome;
    use infinitum_chat::Delivery;
    use infinitum_chat::Effort;
    use infinitum_chat::Finish;
    use infinitum_chat::ReusePath;
    use infinitum_chat::Submission;
    use infinitum_chat::Switch;
    use infinitum_chat::Tally;
    use infinitum_chat::Telemetry;
    use infinitum_chat::Thinking;
    use infinitum_chat::ThinkingBudget;
    use infinitum_chat::ThinkingSpend;
    use infinitum_round::TokenCount;
    use infinitum_round::TokenId;

    use super::Count;
    use super::Phase;
    use super::RequestId;
    use super::RequestShape;
    use super::Severity;
    use super::done;
    use super::failed;
    use super::started;
    use crate::error::ApiError;

    /// A one-message streamed request with no tools.
    const SHAPE: RequestShape = RequestShape {
        id: RequestId(7),
        delivery: Delivery::Streaming,
        messages: Count(1),
        tools: Count(0),
        max_output: TokenCount::ZERO,
        effort: Effort::Unrequested,
        preserve_thinking: Switch::ModelDefault,
    };

    #[test]
    fn request_lines_follow_ninfer()
    {
        let shape = RequestShape {
            max_output: TokenCount::from(1_u32 << 17_u32),
            ..SHAPE
        };
        let budget = ThinkingBudget::Tokens(core::num::NonZeroU32::new(98_304).unwrap());
        assert_eq!(
            started(&shape, Submission {
                thinking: Thinking::Open,
                thinking_budget: budget
            })
            .message,
            "req#7 started | openai-chat stream | 1 message | max output 131,072 | thinking template default, budget 98,304",
            "thinking on under a budget"
        );
        let tooled = RequestShape {
            messages: Count(3),
            tools: Count(2),
            effort: Effort::Low,
            preserve_thinking: Switch::On,
            delivery: Delivery::Aggregate,
            ..shape
        };
        assert_eq!(
            started(&tooled, Submission {
                thinking: Thinking::Closed,
                thinking_budget: ThinkingBudget::Unlimited
            })
            .message,
            "req#7 started | openai-chat non-stream | 3 messages | max output 131,072 | thinking off | tools 2 | preserve thinking",
            "thinking off with the optional clauses"
        );
        let outcome = ChatOutcome {
            content: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            finish: Finish::StopToken,
            admission: Admission {
                prompt_tokens: TokenCount::from(2000_u32),
                reused_tokens: TokenCount::from(1000_u32),
            },
            generated: vec![TokenId::from(0_i32); 101],
            reasoning_tokens: TokenCount::ZERO,
            prompt_wall: core::time::Duration::ZERO,
            generation_wall: core::time::Duration::ZERO,
            drafted: Tally(70),
            accepted: Tally(35),
            telemetry: Telemetry {
                first_token: core::time::Duration::from_millis(250),
                total: core::time::Duration::from_millis(3210),
                prefill: core::time::Duration::from_millis(500),
                decode: core::time::Duration::from_secs(2),
                queue_wait: core::time::Duration::from_millis(20),
                reuse: ReusePath::SharedPrefix,
                thinking: ThinkingSpend {
                    budget,
                    model_tokens: TokenCount::from(64_u32),
                    injected_tokens: TokenCount::from(3_u32),
                },
                accepted_per_position: vec![Tally(9), Tally(7), Tally(5)],
            },
        };
        assert_eq!(
            done(&shape, &outcome).message,
            "req#7 done | openai-chat | stop token | prompt 2,000 | output 101 | cache 1,000 (50.0%, shared prefix) | TTFT 250 ms | total 3.2s | queue 20.0 ms | prefill 2.00k tok/s | decode 50.0 tok/s | dflash2 accepted 35/70 (50.0%) | thinking 64/98,304, control 3 | per position 9/7/5",
            "every clause"
        );
        let bare = ChatOutcome {
            drafted: Tally(0),
            accepted: Tally(0),
            telemetry: Telemetry {
                queue_wait: core::time::Duration::from_millis(9),
                prefill: core::time::Duration::ZERO,
                decode: core::time::Duration::ZERO,
                reuse: ReusePath::Root,
                thinking: ThinkingSpend {
                    budget: ThinkingBudget::Unlimited,
                    model_tokens: TokenCount::ZERO,
                    injected_tokens: TokenCount::ZERO,
                },
                accepted_per_position: Vec::new(),
                ..outcome.telemetry.clone()
            },
            ..outcome
        };
        assert_eq!(
            done(&shape, &bare).message,
            "req#7 done | openai-chat | stop token | prompt 2,000 | output 101 | cache 1,000 (50.0%) | TTFT 250 ms | total 3.2s",
            "no optional clause"
        );
    }

    #[test]
    fn startup_lines_follow_ninfer()
    {
        let mut summary = infinitum_chat::Capacity {
            kv_tokens: infinitum_chat::Tally(539_520),
            kv_storage: String::from("nvfp4"),
            kv_sizing: infinitum_chat::KvSizing::Automatic,
            pages: infinitum_chat::Tally(2108),
            max_pages: infinitum_chat::Tally(2108),
            runtime: infinitum_chat::ByteSize(3 << 30_u32),
            free: infinitum_chat::ByteSize(1536 << 20_u32),
            cache: infinitum_chat::ContextCache::Enabled {
                lanes: infinitum_chat::Tally(3),
                device_states: infinitum_chat::Tally(3),
                host_states: infinitum_chat::Tally(8),
                host_kv: infinitum_chat::ByteSize(36_864 << 20_u32),
                private: infinitum_chat::Tally(6),
                shared: infinitum_chat::Tally(4),
                anchors: infinitum_chat::Tally(2),
            },
        };
        let [first, second] = super::capacity(&summary);
        assert_eq!(
            first.message,
            "capacity | KV 539,520 tokens, nvfp4, auto | pages 2,108/2,108 | runtime 3.00 GiB | free 1.50 GiB",
            "the KV line"
        );
        assert_eq!(
            second.message,
            "context cache | 3 active + 3 cached device states | host 8 states, 36.0 GiB KV | private 6 | shared 4 | anchors 2",
            "the enabled cache line"
        );
        assert_eq!(
            super::engine_ready(
                &infinitum_chat::ModelName(String::from("Qwen3.8-27B")),
                &infinitum_chat::LoadReport {
                    total: core::time::Duration::from_millis(95_400),
                    weights: infinitum_chat::ByteSize(17 << 30_u32),
                    cuda_sync: String::from("blocking"),
                }
            )
            .message,
            "engine ready | Qwen3.8-27B | total 1m 35.4s | weights 17.0 GiB | CUDA sync blocking",
            "the load line"
        );
        summary.cache = infinitum_chat::ContextCache::RootOnly;
        assert_eq!(
            super::capacity(&summary)[1].message,
            "context cache | root only",
            "a disabled cache"
        );
        let address = core::net::SocketAddr::from(([127, 0, 0, 1], 18_031));
        let model = crate::server::ModelId(String::from("qwen\n"));
        assert_eq!(
            super::listening(address, &model, &crate::server::Access::Open).message,
            "listening on http://127.0.0.1:18031 | model qwen  | auth disabled",
            "an open server, its id made safe"
        );
    }

    #[test]
    fn lines_carry_ninfers_stamp_and_level()
    {
        let at = jiff::civil::date(2026, 9, 26)
            .at(4, 5, 6, 123_999_999)
            .to_zoned(jiff::tz::TimeZone::UTC)
            .unwrap();
        for (severity, expected) in [
            (Severity::Info, "2026-09-26 04:05:06.123  INFO  ready"),
            (Severity::Warning, "2026-09-26 04:05:06.123  WARN  ready"),
            (Severity::Error, "2026-09-26 04:05:06.123  ERROR ready"),
        ] {
            let record = super::Record {
                severity,
                message: String::from("ready"),
            };
            assert_eq!(super::service_line(&at, &record), expected, "{severity:?}");
        }
    }

    #[test]
    fn throughput_lines_follow_ninfer()
    {
        let idle = infinitum_chat::RuntimeCounters::default();
        let second = core::time::Duration::from_secs(5);
        assert_eq!(
            super::throughput(&idle, &idle, second),
            super::Throughput::Quiet,
            "nothing moved"
        );
        let shuffled = infinitum_chat::RuntimeCounters {
            digest: infinitum_chat::CounterDigest(1),
            ..idle
        };
        assert_eq!(
            super::throughput(&idle, &shuffled, second),
            super::Throughput::Active(super::Record {
                severity: Severity::Info,
                message: String::from("throughput | 5.0s | running 0 | host 0.0% (0 us)"),
            }),
            "only the digest moved, as a cache eviction does"
        );
        let busy = infinitum_chat::RuntimeCounters {
            prefill_tokens: infinitum_chat::Tally(12_000),
            decode_tokens: infinitum_chat::Tally(400),
            decode_rounds: infinitum_chat::Tally(80),
            decode_rows: infinitum_chat::Tally(200),
            requests: infinitum_chat::RequestGauges {
                running: infinitum_chat::Tally(3),
                prefilling: infinitum_chat::Tally(1),
                decode_ready: infinitum_chat::Tally(2),
                waiting: infinitum_chat::Tally(1),
                terminal_pending: infinitum_chat::Tally(1),
                ..infinitum_chat::RequestGauges::default()
            },
            host_active: core::time::Duration::from_millis(250),
            digest: infinitum_chat::CounterDigest(0),
        };
        assert_eq!(
            super::throughput(&idle, &busy, second),
            super::Throughput::Active(super::Record {
                severity: Severity::Info,
                message: String::from(
                    "throughput | 5.0s | prefill 2.40k tok/s (12,000 tok) | decode 80.0 tok/s (400 tok) | \
                     running 3 (prefill 1, decode-ready 2) | waiting 1 | terminal-pending 1 | batch 2.50 | \
                     host 5.0% (250 ms)"
                ),
            }),
            "a busy interval"
        );
    }

    #[test]
    fn failure_lines_follow_ninfer()
    {
        let refused = ApiError {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: String::new(),
            param: String::new(),
            code: String::from("context_length_exceeded"),
        };
        let line = failed(&SHAPE, Phase::Prepare, &refused);
        assert_eq!(
            line.message,
            "req#7 rejected during prepare | openai-chat stream | HTTP 400 | context length exceeded | messages 1",
            "client input before submission"
        );
        assert_eq!(line.severity, Severity::Info, "client input is routine");
        let timeout = ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: String::from("request_queue_timeout"),
            ..refused.clone()
        };
        let line = failed(&SHAPE, Phase::Prepare, &timeout);
        assert_eq!(
            line.message,
            "req#7 failed during prepare | openai-chat stream | HTTP 503 | request queue timeout | messages 1",
            "a queue timeout fails"
        );
        assert_eq!(
            line.severity,
            Severity::Warning,
            "a timeout degrades service"
        );
        let internal = ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: String::new(),
            ..refused
        };
        let line = failed(&SHAPE, Phase::Generation, &internal);
        assert_eq!(
            line.message,
            "req#7 failed during generation | openai-chat | HTTP 500 | internal error",
            "a fault after submission names its class"
        );
        assert_eq!(line.severity, Severity::Error, "a fault is an error");
    }
}
