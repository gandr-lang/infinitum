//! `infinitum serve`: the OpenAI-compatible chat server on ninfer's Engine.
//!
//! The command plans the DFlash2 round like `generate`, then, with the
//! `ninfer` feature, opens the Engine with the chat template, warms it up,
//! binds the listener, reports the address and model id on one line, and
//! serves until the process ends.

use std::io::Write;
use std::path::PathBuf;

use infinitum_ninfer::ChatTemplate;
use infinitum_ninfer::Concurrency;
use infinitum_ninfer::ContextLimit;
use infinitum_ninfer::CudaGraph;
use infinitum_ninfer::DFlash2Plan;
use infinitum_ninfer::DeviceOrdinal;
use infinitum_ninfer::DeviceStateSlots;
use infinitum_ninfer::EngineOptions;
use infinitum_ninfer::HostKvBytes;
use infinitum_ninfer::KvBelowContext;
use infinitum_ninfer::KvCapacity;
use infinitum_ninfer::KvStorage;
use infinitum_ninfer::Ninfer;
use infinitum_ninfer::PendingTimeout;
use infinitum_ninfer::PrefillChunk;
use infinitum_ninfer::RopeFactor;
use infinitum_ninfer::RopeThreshold;
use infinitum_ninfer::StateSlots;
use infinitum_round::Backend as _;
use infinitum_round::BuildFailure;
use infinitum_round::DraftWidth;
use infinitum_round::Refusal;

/// Serve chat completions over HTTP from ninfer's Engine, with infinitum's
/// DFlash2 round reviewing every round.
#[derive(Debug, Clone, clap::Args)]
pub struct Server
{
    /// The `.ninfer` artifact to open; it must carry a DFlash2 companion.
    #[arg(long, value_name = "PATH")]
    artifact: PathBuf,
    /// A Jinja chat template to render prompts with, in place of the
    /// artifact's.
    #[arg(long, value_name = "PATH")]
    chat_template: Option<PathBuf>,
    /// DFlash2's draft width `K`: tokens drafted per round. ninfer runs 1 to
    /// 15.
    #[arg(long, value_name = "TOKENS", default_value = "7")]
    draft_width: DraftWidth,
    /// The per-request context ceiling in tokens.
    #[arg(long, value_name = "TOKENS", default_value = "8192")]
    max_context: ContextLimit,
    /// The Main KV capacity shared by every request: a token count at least
    /// `--max-context`, or `auto` to size it from device memory; absent, the
    /// context ceiling.
    #[arg(long, value_name = "TOKENS|auto")]
    kv_capacity: Option<KvCapacity>,
    /// How the KV cache stores keys and values: `bf16`, `int8`, `fp8`,
    /// `nvfp4` or `k8v4`.
    #[arg(long, value_name = "STORAGE", default_value = "bf16")]
    kv_dtype: KvStorage,
    /// Prompt tokens per prefill step, a positive multiple of 128.
    #[arg(long, value_name = "TOKENS", default_value = "1024")]
    prefill_chunk: PrefillChunk,
    /// Requests the Engine runs at once, one to eight; more wait for a lane
    /// up to `--pending-timeout-ms`.
    #[arg(long, value_name = "LANES", default_value = "1")]
    max_concurrency: Concurrency,
    /// Device recurrent-state checkpoint slots kept beyond the active lanes
    /// for prefix reuse; absent, one per lane.
    #[arg(long, value_name = "SLOTS")]
    device_state_slots: Option<StateSlots>,
    /// Host recurrent-state checkpoint slots for prefix reuse.
    #[arg(long, value_name = "SLOTS", default_value = "8")]
    host_state_slots: StateSlots,
    /// Pinned host memory for reusable KV, in MiB.
    #[arg(long, value_name = "MIB", default_value = "8192")]
    host_kv_mib: HostKvBytes,
    /// How far `RoPE` positions stretch past `--rope-scaling-original-context`,
    /// one to sixteen; one leaves them unscaled.
    #[arg(long, value_name = "FACTOR", default_value = "1")]
    rope_scaling_factor: RopeFactor,
    /// The native position threshold past which `RoPE` positions scale.
    #[arg(long, value_name = "POSITIONS", default_value = "262144")]
    rope_scaling_original_context: RopeThreshold,
    /// The output limit of a request that names none; the Engine also clamps
    /// it to the context left after the prompt.
    #[arg(long, value_name = "TOKENS", default_value = "8192")]
    default_max_tokens: OutputTokens,
    /// The thinking budget of a request that does not turn thinking off; its
    /// control tokens count toward the output limit. Absent, thinking runs
    /// until the model closes it or the output limit binds.
    #[arg(long, value_name = "TOKENS")]
    default_thinking_budget: Option<OutputTokens>,
    /// How long a request may wait for admission before it is refused.
    #[arg(long, value_name = "MILLISECONDS", default_value = "30000")]
    pending_timeout_ms: PendingTimeout,
    /// The largest request body accepted, in MiB; a larger one is refused
    /// with 413 before it is parsed.
    #[arg(long, value_name = "MIB", default_value = "384")]
    max_request_mib: RequestMib,
    /// How often throughput is logged, for intervals that saw activity; 0
    /// never logs it.
    #[arg(long, value_name = "MILLISECONDS", default_value = "5000")]
    log_stats_interval_ms: StatsPeriod,
    /// The CUDA device ordinal.
    #[arg(long, value_name = "ORDINAL", default_value = "0")]
    device: DeviceOrdinal,
    /// Whether decode rounds are captured as CUDA graphs: `off` or `on`.
    #[arg(long, value_name = "CHOICE", default_value = "on")]
    cuda_graph: CudaGraph,
    /// The address to listen on.
    #[arg(long, value_name = "ADDRESS", default_value = "127.0.0.1")]
    host: String,
    /// The port to listen on.
    #[arg(long, value_name = "PORT", default_value = "8080")]
    port: u16,
    /// The key clients present as a bearer token or `x-api-key`; absent or
    /// empty, the API is open, as ninfer's is.
    #[arg(long, value_name = "KEY")]
    api_key: Option<String>,
    /// The model id clients name; absent, the artifact's model name.
    #[arg(long, value_name = "ID")]
    model_id: Option<String>,
}

/// A positive token count: an output limit or a thinking budget.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputTokens(core::num::NonZeroU32);

impl core::str::FromStr for OutputTokens
{
    type Err = core::num::ParseIntError;

    /// Parse a positive decimal token count.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the count is the parsed value, at least one.
    /// - provides: `--default-max-tokens` and `--default-thinking-budget`,
    ///   refusing zero as ninfer's server does.
    /// - fails: with the integer parser's error on zero, a negative value,
    ///   anything above `u32::MAX`, or a non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`core::num::ParseIntError`]: `text` is not a positive `u32`.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the zero boundary.
    /// - witness: `crate::tests::serve_limits_refuse_zero`
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return text.parse::<core::num::NonZeroU32>().map(Self);
    }
}

/// How often the server logs throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsPeriod
{
    /// Never.
    Off,
    /// At this period.
    Every(core::time::Duration),
}

impl core::str::FromStr for StatsPeriod
{
    type Err = core::num::ParseIntError;

    /// Parse a non-negative decimal count of milliseconds.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success zero is [`StatsPeriod::Off`] and any other count
    ///   that many milliseconds, as ninfer's `--log-stats-interval-ms` reads
    ///   it.
    /// - provides: `--log-stats-interval-ms`.
    /// - fails: with the integer parser's error on a negative value, anything
    ///   above `u32::MAX`, or a non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`core::num::ParseIntError`]: `text` is not a `u32`.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the zero boundary.
    /// - witness: `crate::tests::stats_interval_zero_turns_reporting_off`
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        let milliseconds = text.parse::<u32>()?;
        return Ok(if milliseconds == 0 {
            Self::Off
        }
        else {
            Self::Every(core::time::Duration::from_millis(u64::from(milliseconds)))
        });
    }
}

/// A request-body cap in MiB, whose byte count fits `usize`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestMib(core::num::NonZeroUsize);

/// A request-body cap that is zero, not a number, or too large to count in
/// bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestMibOutOfRange;

impl core::fmt::Display for RequestMibOutOfRange
{
    /// Render the refusal, in ninfer's words.
    ///
    /// # Specification
    /// trivial.
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str("--max-request-mib is out of range");
    }
}

impl core::error::Error for RequestMibOutOfRange
{
}

impl core::str::FromStr for RequestMib
{
    type Err = RequestMibOutOfRange;

    /// Parse a positive decimal count of MiB whose bytes fit `usize`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the cap is the parsed MiB in bytes.
    /// - provides: `--max-request-mib`, with ninfer's range.
    /// - fails: on zero, a non-numeric string, or a count whose bytes overflow
    ///   `usize`.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`RequestMibOutOfRange`]: as stated.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at zero and at the overflow boundary.
    /// - witness: `crate::tests::serve_limits_refuse_zero`
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        let mib = text
            .parse::<core::num::NonZeroUsize>()
            .map_err(|_malformed| return RequestMibOutOfRange)?;
        return mib
            .get()
            .checked_mul(1 << 20)
            .and_then(core::num::NonZeroUsize::new)
            .map(Self)
            .ok_or(RequestMibOutOfRange);
    }
}

/// A failure of `serve`.
#[derive(Debug)]
pub enum ServeFailure
{
    /// The DFlash2 round could not be composed.
    Compose(BuildFailure),
    /// ninfer refused the round.
    Plan(Refusal),
    /// The KV capacity is below the context ceiling.
    KvCapacity(KvBelowContext),
    /// This build has no ninfer Engine.
    #[cfg(not(feature = "ninfer"))]
    NoEngine(DFlash2Plan),
    /// Opening the Engine failed.
    #[cfg(feature = "ninfer")]
    Engine(infinitum_ninfer::EngineFailure),
    /// The warm-up request failed.
    #[cfg(feature = "ninfer")]
    WarmUp(infinitum_chat::ChatFailure),
    /// Binding, serving, or writing the ready line failed.
    Io(std::io::Error),
}

impl core::fmt::Display for ServeFailure
{
    /// Render the failure.
    ///
    /// # Specification
    /// trivial.
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return match *self {
            | Self::Compose(ref failure) => {
                write!(f, "cannot compose the DFlash2 round: {failure}")
            },
            | Self::Plan(ref refusal) => core::fmt::Display::fmt(refusal, f),
            | Self::KvCapacity(ref failure) => core::fmt::Display::fmt(failure, f),
            #[cfg(not(feature = "ninfer"))]
            | Self::NoEngine(ref plan) => write!(
                f,
                "ninfer plans the DFlash2 round at draft width {}, but this build has no ninfer \
                 engine; rebuild with `--features ninfer`",
                plan.width()
            ),
            #[cfg(feature = "ninfer")]
            | Self::Engine(ref failure) => core::fmt::Display::fmt(failure, f),
            #[cfg(feature = "ninfer")]
            | Self::WarmUp(ref failure) => write!(f, "the warm-up request failed: {failure}"),
            | Self::Io(ref failure) => write!(f, "cannot serve: {failure}"),
        };
    }
}

impl From<std::io::Error> for ServeFailure
{
    /// Wrap an I/O failure.
    ///
    /// # Specification
    /// trivial.
    fn from(failure: std::io::Error) -> Self
    {
        return Self::Io(failure);
    }
}

/// Plan the server's round on ninfer.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success the plan runs the canonical DFlash2 round at the
///   requested draft width.
/// - provides: the gate every server passes before an Engine opens.
/// - fails: when the round cannot be composed or ninfer refuses it.
/// - panics: none.
///
/// # Errors
/// - [`ServeFailure::Compose`], [`ServeFailure::Plan`]: as named.
///
/// # Adequacy
/// - hypothesis: L3 on the refusal, a width past ninfer's range.
/// - witness: `crate::tests::serve_plans_before_opening`
fn plan(server: &Server) -> Result<DFlash2Plan, ServeFailure>
{
    let graph = infinitum_round::dflash2(server.draft_width).map_err(ServeFailure::Compose)?;
    return Ninfer.plan(&graph).map_err(ServeFailure::Plan);
}

/// Gather the Engine options the flags name.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success the options carry the artifact, device, context, KV
///   capacity (the context ceiling when absent), KV storage, prefill chunk,
///   concurrency, context-cache capacities, `RoPE` scaling, pending timeout,
///   CUDA graph choice and chat template the flags name.
/// - provides: every Engine option `serve` sets, checked before an Engine
///   opens.
/// - fails: when `--kv-capacity` is a token count below `--max-context`.
/// - panics: none.
///
/// # Errors
/// - [`ServeFailure::KvCapacity`]: as stated.
///
/// # Adequacy
/// - hypothesis: L3 on the refusal; the option setters are witnessed in
///   `infinitum-ninfer`.
/// - witness: `crate::tests::serve_refuses_a_kv_capacity_below_the_context`
fn engine_options(server: &Server) -> Result<EngineOptions, ServeFailure>
{
    let template = server
        .chat_template
        .clone()
        .map_or(ChatTemplate::Artifact, ChatTemplate::File);
    let options = EngineOptions::new(
        server.artifact.clone(),
        server.device,
        server.max_context,
        server.cuda_graph,
    )
    .with_chat_template(template)
    .with_pending_timeout(server.pending_timeout_ms)
    .with_kv_storage(server.kv_dtype)
    .with_prefill_chunk(server.prefill_chunk)
    .with_concurrency(server.max_concurrency)
    .with_context_cache(
        server
            .device_state_slots
            .map_or(DeviceStateSlots::PerLane, DeviceStateSlots::Exactly),
        server.host_state_slots,
        server.host_kv_mib,
    )
    .with_rope_scaling(
        server.rope_scaling_factor,
        server.rope_scaling_original_context,
    );
    return match server.kv_capacity {
        | Some(capacity) => options
            .with_kv_capacity(capacity)
            .map_err(ServeFailure::KvCapacity),
        | None => Ok(options),
    };
}

/// Plan, open, warm up and serve.
///
/// # Specification
/// - requires: `out` accepts bytes.
/// - ensures: nothing returns while the server runs; before it serves, `out`
///   holds one `listening on <host>:<port> as <model id>` line.
/// - provides: the driver's server.
/// - fails: with the first failure, in the order options, plan, open, warm up,
///   bind, serve; without the `ninfer` feature, with [`ServeFailure::NoEngine`]
///   after a successful plan.
/// - panics: none.
///
/// # Errors
/// - [`ServeFailure`]: as ordered above.
///
/// # Adequacy
/// - hypothesis: the plan step is L3 in the suite; every later step needs
///   ninfer and a device and is witnessed by the served comparison against
///   `ninfer-serve`.
/// - witness: `crate::tests::serve_plans_before_opening`
pub fn run<Writer>(
    server: &Server,
    out: &mut Writer,
) -> Result<(), ServeFailure>
where
    Writer: Write,
{
    let options = engine_options(server)?;
    let plan = plan(server)?;
    return execute(server, plan, &options, out);
}

/// Without an Engine, a planned server goes no further.
///
/// # Specification
/// - requires: nothing.
/// - ensures: nothing is written.
/// - provides: the explicit end of a build without the `ninfer` feature.
/// - fails: always, with [`ServeFailure::NoEngine`].
/// - panics: none.
///
/// # Errors
/// - [`ServeFailure::NoEngine`]: this build has no Engine.
#[cfg(not(feature = "ninfer"))]
const fn execute<Writer>(
    _server: &Server,
    plan: DFlash2Plan,
    _options: &EngineOptions,
    _out: &mut Writer,
) -> Result<(), ServeFailure>
where
    Writer: Write,
{
    return Err(ServeFailure::NoEngine(plan));
}

/// Open the Engine, warm it up, and serve on a multi-threaded runtime.
///
/// # Specification
/// - requires: `out` accepts bytes.
/// - ensures: as [`run`] states, after the plan.
/// - provides: the engine half of [`run`].
/// - fails: with the first failing step.
/// - panics: none.
///
/// # Errors
/// - [`ServeFailure`]: as [`run`] orders them.
///
/// # Adequacy
/// - hypothesis: L2 outside the suite — every step needs ninfer and a device.
#[cfg(feature = "ninfer")]
fn execute<Writer>(
    server: &Server,
    plan: DFlash2Plan,
    options: &EngineOptions,
    out: &mut Writer,
) -> Result<(), ServeFailure>
where
    Writer: Write,
{
    use alloc::sync::Arc;

    use infinitum_chat::ChatBackend as _;

    let mut startup = infinitum_serve::StartupLog::default();
    let engine = infinitum_ninfer::ChatEngine::open(options, plan, &mut startup)
        .map_err(ServeFailure::Engine)?;
    infinitum_serve::log_engine_ready(engine.model_name(), engine.load());
    infinitum_serve::log_capacity(&engine.capacity().map_err(ServeFailure::Engine)?);
    let thinking_budget = server.default_thinking_budget.map_or(
        infinitum_chat::ThinkingBudget::Unlimited,
        |budget| {
            return infinitum_chat::ThinkingBudget::Tokens(budget.0);
        },
    );
    infinitum_serve::warm_up(&engine, thinking_budget).map_err(ServeFailure::WarmUp)?;
    let model = server
        .model_id
        .clone()
        .unwrap_or_else(|| return engine.model_name().0.clone());
    let access = server
        .api_key
        .clone()
        .map_or(infinitum_serve::Access::Open, |key| {
            return infinitum_serve::Access::of(infinitum_serve::ApiKey(key));
        });
    let config = infinitum_serve::ServeConfig {
        model: infinitum_serve::ModelId(model),
        access,
        max_model_len: infinitum_round::TokenCount::from(core::num::NonZeroU32::from(
            server.max_context,
        )),
        defaults: infinitum_serve::Defaults {
            output_tokens: infinitum_round::TokenCount::from(server.default_max_tokens.0),
            thinking_budget,
        },
        max_request: infinitum_serve::RequestBytes(server.max_request_mib.0),
        stats: match server.log_stats_interval_ms {
            | StatsPeriod::Off => infinitum_serve::StatsInterval::Off,
            | StatsPeriod::Every(period) => infinitum_serve::StatsInterval::Every(period),
        },
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .build()?;
    return runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind((server.host.as_str(), server.port)).await?;
        let address = listener.local_addr()?;
        writeln!(out, "listening on {address} as {}", config.model.0)?;
        out.flush()?;
        infinitum_serve::serve(listener, Arc::new(engine), config).await?;
        return Ok(());
    });
}
