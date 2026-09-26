//! `infinitum serve`: the OpenAI-compatible chat server on ninfer's Engine.
//!
//! The command plans the DFlash2 round like `generate`, then, with the
//! `ninfer` feature, opens the Engine with the chat template, warms it up,
//! binds the listener, reports the address and model id on one line, and
//! serves until the process ends.

use std::io::Write;
use std::path::PathBuf;

use infinitum_ninfer::ContextLimit;
use infinitum_ninfer::CudaGraph;
use infinitum_ninfer::DFlash2Plan;
use infinitum_ninfer::DeviceOrdinal;
use infinitum_ninfer::Ninfer;
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
    /// The per-request context ceiling in tokens; it also sizes the KV cache.
    #[arg(long, value_name = "TOKENS", default_value = "8192")]
    max_context: ContextLimit,
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

/// A failure of `serve`.
#[derive(Debug)]
pub enum ServeFailure
{
    /// The DFlash2 round could not be composed.
    Compose(BuildFailure),
    /// ninfer refused the round.
    Plan(Refusal),
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

/// Plan, open, warm up and serve.
///
/// # Specification
/// - requires: `out` accepts bytes.
/// - ensures: nothing returns while the server runs; before it serves, `out`
///   holds one `listening on <host>:<port> as <model id>` line.
/// - provides: the driver's server.
/// - fails: with the first failure, in the order plan, open, warm up, bind,
///   serve; without the `ninfer` feature, with [`ServeFailure::NoEngine`] after
///   a successful plan.
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
    let plan = plan(server)?;
    return execute(server, plan, out);
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
    out: &mut Writer,
) -> Result<(), ServeFailure>
where
    Writer: Write,
{
    use alloc::sync::Arc;

    use infinitum_chat::ChatBackend as _;
    use infinitum_ninfer::ChatTemplate;

    let template = server
        .chat_template
        .clone()
        .map_or(ChatTemplate::Artifact, ChatTemplate::File);
    let options = infinitum_ninfer::EngineOptions::new(
        server.artifact.clone(),
        server.device,
        server.max_context,
        server.cuda_graph,
    )
    .with_chat_template(template);
    let engine =
        infinitum_ninfer::ChatEngine::open(&options, plan).map_err(ServeFailure::Engine)?;
    infinitum_serve::warm_up(&engine).map_err(ServeFailure::WarmUp)?;
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
            output_tokens: infinitum_round::TokenCount::from(8192_u32),
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
