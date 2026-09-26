//! The HTTP server: routes, the API-key gate, and the bridge between a
//! blocking chat backend and an async response.
//!
//! Routes are ninfer's chat surface: `GET /health`, `GET /v1/models`,
//! `GET /v1/models/{id}` and `POST /v1/chat/completions`. Each request runs
//! on a blocking thread; a streamed one begins its event stream once the
//! backend has submitted it, so a request refused during preparation gets an
//! HTTP error status, and a failure after that ends the stream with an error
//! event, as on ninfer's server.

use alloc::sync::Arc;

use axum::body::Body;
use axum::body::Bytes;
use axum::extract::Path;
use axum::extract::Request;
use axum::extract::State;
use axum::http::HeaderValue;
use axum::http::StatusCode;
use axum::http::header;
use axum::middleware::Next;
use axum::response::IntoResponse as _;
use axum::response::Response;
use infinitum_chat::Admission;
use infinitum_chat::CancelToken;
use infinitum_chat::Channel;
use infinitum_chat::ChatBackend;
use infinitum_chat::ChatEvents;
use infinitum_chat::ChatRequest;
use infinitum_chat::Delivery;
use infinitum_chat::DeltaText;
use infinitum_chat::Submission;
use infinitum_round::TokenCount;
use serde_json::json;

use crate::error::ApiError;
use crate::error::Code;
use crate::error::Param;
use crate::oplog::Logged;
use crate::oplog::RequestId;
use crate::oplog::RequestShape;
use crate::parse::Defaults;
use crate::parse::FreshSeed;
use crate::parse::ParsedChat;
use crate::parse::chat_request;
use crate::render::ChunkStream;
use crate::render::Identity;
use crate::render::completion;
use crate::render::error_event;
use crate::render::fresh_bits;
use crate::render::unix_now;

/// The largest request body accepted.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestBytes(pub core::num::NonZeroUsize);

impl RequestBytes
{
    /// ninfer's default: 384 MiB.
    pub const DEFAULT: Self = Self(match core::num::NonZeroUsize::new(384_usize << 20_u32) {
        | Some(bytes) => bytes,
        | None => core::num::NonZeroUsize::MIN,
    });
}

/// How long a stream may go without an event before a keep-alive comment.
const HEARTBEAT: core::time::Duration = core::time::Duration::from_secs(5);

/// The keep-alive comment.
const HEARTBEAT_COMMENT: &str = ": keep-alive\n\n";

/// The model id the server answers to.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelId(pub String);

/// A client credential.
#[repr(transparent)]
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(pub String);

impl core::fmt::Debug for ApiKey
{
    /// Render the key redacted.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str("ApiKey(..)");
    }
}

/// Who may call the API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access
{
    /// Anyone.
    Open,
    /// Callers presenting the key as a bearer token or an `x-api-key`
    /// header; `/health` stays open.
    Key(ApiKey),
}

impl Access
{
    /// The access a configured key grants.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: [`Access::Open`] for an empty key, as ninfer's server treats
    ///   an empty `--api-key` as no authentication; [`Access::Key`] otherwise,
    ///   so no empty credential is ever accepted.
    /// - provides: the lowering of the `--api-key` flag.
    /// - fails: never.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the empty boundary.
    /// - witness: `tests::an_empty_key_leaves_the_api_open`
    #[inline]
    #[must_use]
    pub fn of(key: ApiKey) -> Self
    {
        if key.0.is_empty() {
            return Self::Open;
        }
        return Self::Key(key);
    }
}

/// How the server answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeConfig
{
    /// The public model id.
    pub model: ModelId,
    /// Who may call.
    pub access: Access,
    /// The per-request context ceiling `/v1/models` reports.
    pub max_model_len: TokenCount,
    /// Request defaults.
    pub defaults: Defaults,
    /// The largest request body accepted.
    pub max_request: RequestBytes,
    /// How often throughput is logged.
    pub stats: StatsInterval,
}

/// How often the server logs throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsInterval
{
    /// Never.
    Off,
    /// At this period, for intervals that saw activity.
    Every(core::time::Duration),
}

/// What every handler shares.
struct Shared
{
    /// The backend.
    backend: Arc<dyn ChatBackend>,
    /// The configuration.
    config: ServeConfig,
    /// The last request number the log used.
    requests: core::sync::atomic::AtomicU64,
}

impl Shared
{
    /// Number the next request.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: a number one more than the last, from one.
    /// - provides: the log's `req#` numbers.
    /// - fails: never.
    /// - panics: none.
    fn next_request(&self) -> RequestId
    {
        let last = self
            .requests
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        return RequestId(last.saturating_add(1));
    }
}

/// A JSON response.
///
/// # Specification
/// trivial.
fn json_response(
    status: StatusCode,
    body: String,
) -> Response
{
    return (status, [(header::CONTENT_TYPE, "application/json")], body).into_response();
}

/// An error response.
///
/// # Specification
/// trivial.
fn error_response(error: &ApiError) -> Response
{
    return json_response(error.status, error.body());
}

/// Whether an `Authorization` value carries `key` as a bearer token.
///
/// # Specification
/// - requires: nothing.
/// - ensures: true exactly when the value is, after optional spaces or tabs,
///   the scheme `Bearer` in any ASCII case, at least one space or tab, and
///   `key` with surrounding spaces and tabs trimmed.
/// - provides: ninfer's bearer check.
/// - fails: never.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 — scheme case, required separator, trimming, and a wrong
///   key.
/// - witness: `tests::bearer_matching_follows_ninfer`
fn bearer_matches(
    authorization: &HeaderValue,
    key: &ApiKey,
) -> Verdict
{
    let Ok(authorization) = authorization.to_str()
    else {
        return Verdict::Refused;
    };
    let blank = |character: char| return character == ' ' || character == '\t';
    let value = authorization.trim_start_matches(blank);
    let Some((scheme, rest)) = value.split_at_checked(6)
    else {
        return Verdict::Refused;
    };
    if !scheme.eq_ignore_ascii_case("Bearer") || !rest.starts_with(blank) {
        return Verdict::Refused;
    }
    if rest.trim_matches(blank) == key.0 {
        return Verdict::Admitted;
    }
    return Verdict::Refused;
}

/// The gate's verdict on a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict
{
    /// It may.
    Admitted,
    /// It lacks a valid credential.
    Refused,
}

/// Whether `request` may reach its route.
///
/// # Specification
/// - requires: nothing.
/// - ensures: admitted when access is open, for `/health`, for `OPTIONS`, and
///   for a matching bearer token or `x-api-key`.
/// - provides: the gate's decision.
/// - fails: never.
/// - panics: none.
fn admission(
    access: &Access,
    request: &Request,
) -> Verdict
{
    let Access::Key(ref key) = *access
    else {
        return Verdict::Admitted;
    };
    if request.uri().path() == "/health" || request.method() == axum::http::Method::OPTIONS {
        return Verdict::Admitted;
    }
    let headers = request.headers();
    if let Some(value) = headers.get(header::AUTHORIZATION)
        && bearer_matches(value, key) == Verdict::Admitted
    {
        return Verdict::Admitted;
    }
    if headers.get("x-api-key").map(HeaderValue::as_bytes) == Some(key.0.as_bytes()) {
        return Verdict::Admitted;
    }
    return Verdict::Refused;
}

/// The gate every route passes: the key check, and an `x-request-id` on
/// every `/v1/` response.
///
/// # Specification
/// - requires: nothing.
/// - ensures: a refused request gets 401 `invalid_api_key`; every `/v1/`
///   response carries `x-request-id: req_` and 32 hex digits.
/// - provides: authentication and request ids.
/// - fails: never.
/// - panics: none.
async fn gate(
    State(shared): State<Arc<Shared>>,
    request: Request,
    next: Next,
) -> Response
{
    let versioned = request.uri().path().starts_with("/v1/");
    let mut response = match admission(&shared.config.access, &request) {
        | Verdict::Admitted => next.run(request).await,
        | Verdict::Refused => error_response(&ApiError {
            status: StatusCode::UNAUTHORIZED,
            kind: "invalid_request_error",
            message: String::from("missing or invalid API key"),
            param: String::new(),
            code: String::from("invalid_api_key"),
        }),
    };
    if versioned && !response.headers().contains_key("x-request-id") {
        let id = format!("req_{:016x}{:016x}", fresh_bits().0, fresh_bits().0);
        if let Ok(value) = HeaderValue::from_str(&id) {
            response.headers_mut().insert("x-request-id", value);
        }
    }
    return response;
}

/// `GET /health`.
///
/// # Specification
/// trivial.
async fn health() -> Response
{
    return json_response(StatusCode::OK, json!({"status": "ok"}).to_string());
}

/// The model object.
///
/// # Specification
/// trivial.
fn model_object(config: &ServeConfig) -> serde_json::Value
{
    return json!({
        "id": config.model.0,
        "object": "model",
        "created": unix_now().0,
        "owned_by": "infinitum",
        "max_model_len": u32::from(config.max_model_len),
    });
}

/// The error naming an unknown model.
///
/// # Specification
/// trivial.
fn model_not_found(requested: &ModelId) -> ApiError
{
    return ApiError {
        status: StatusCode::NOT_FOUND,
        kind: "invalid_request_error",
        message: format!("model '{}' not found", requested.0),
        param: String::from("model"),
        code: String::from("model_not_found"),
    };
}

/// `GET /v1/models`.
///
/// # Specification
/// trivial.
async fn models(State(shared): State<Arc<Shared>>) -> Response
{
    let list = json!({"object": "list", "data": [model_object(&shared.config)]});
    return json_response(StatusCode::OK, list.to_string());
}

/// `GET /v1/models/{id}`.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the model object for the served id; 404 `model_not_found`
///   otherwise, with no `param`, as ninfer answers.
/// - provides: model lookup.
/// - fails: never.
/// - panics: none.
async fn model(
    State(shared): State<Arc<Shared>>,
    Path(id): Path<String>,
) -> Response
{
    let id = ModelId(id);
    if id != shared.config.model {
        let mut error = model_not_found(&id);
        error.param.clear();
        return error_response(&error);
    }
    return json_response(StatusCode::OK, model_object(&shared.config).to_string());
}

/// Requests cancellation when dropped: the response went away.
#[repr(transparent)]
#[derive(Debug)]
struct CancelOnDrop(CancelToken);

impl Drop for CancelOnDrop
{
    /// Cancel.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn drop(&mut self)
    {
        self.0.request();
    }
}

/// Events of an aggregate request, which the backend does not send.
struct Aggregate;

impl ChatEvents for Aggregate
{
    /// Nothing is streamed.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn submitted(
        &mut self,
        _submission: Submission,
    )
    {
    }

    /// Nothing is streamed.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn admitted(
        &mut self,
        _admission: Admission,
    )
    {
    }

    /// Nothing is streamed.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn publish(
        &mut self,
        _channel: Channel,
        _text: DeltaText<'_>,
    )
    {
    }
}

/// Whether a streamed response has begun.
#[derive(Debug)]
enum Begun
{
    /// Not yet: the decision is still owed to the handler.
    Pending(tokio::sync::oneshot::Sender<Result<(), ApiError>>),
    /// The handler has begun the event stream.
    Streaming,
}

/// Events of a streamed request: encoded as chunks and sent to the body.
struct Streamed
{
    /// The encoder.
    encoder: ChunkStream,
    /// The body's events.
    body: tokio::sync::mpsc::UnboundedSender<String>,
    /// Whether the stream has begun.
    begun: Begun,
}

impl Streamed
{
    /// Begin the stream if it has not begun: tell the handler, and send the
    /// opening chunk.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the handler has been told to stream, once, and the opening
    ///   chunk precedes every other event.
    /// - provides: the start of the event stream.
    /// - fails: never; a handler that went away is ignored, because its
    ///   cancellation is already requested.
    /// - panics: none.
    fn begin(&mut self)
    {
        if let Begun::Pending(decision) = core::mem::replace(&mut self.begun, Begun::Streaming) {
            let _gone = decision.send(Ok(()));
            let _closed = self.body.send(self.encoder.start());
        }
    }
}

impl ChatEvents for Streamed
{
    /// Begin the stream.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn submitted(
        &mut self,
        _submission: Submission,
    )
    {
        self.begin();
    }

    /// Nothing to send: the counts arrive with the outcome.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn admitted(
        &mut self,
        _admission: Admission,
    )
    {
    }

    /// Send the delta.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the stream has begun and the delta's chunk is sent.
    /// - provides: streamed output.
    /// - fails: never; a closed body is ignored, its cancellation requested.
    /// - panics: none.
    #[inline]
    fn publish(
        &mut self,
        channel: Channel,
        text: DeltaText<'_>,
    )
    {
        self.begin();
        let chunk = self.encoder.delta(channel, text);
        let _closed = self.body.send(chunk);
    }
}

/// The next keep-alive deadline: [`HEARTBEAT`] from now, or now when that
/// is past the clock's range.
///
/// # Specification
/// trivial.
fn deadline() -> tokio::time::Instant
{
    let now = tokio::time::Instant::now();
    return now.checked_add(HEARTBEAT).unwrap_or(now);
}

/// A streamed response's body: the events, a keep-alive comment after each
/// quiet interval, and cancellation of the request when dropped.
struct EventBody
{
    /// The events.
    events: tokio::sync::mpsc::UnboundedReceiver<String>,
    /// The next keep-alive deadline.
    heartbeat: core::pin::Pin<Box<tokio::time::Sleep>>,
    /// Cancels the request when the body is dropped.
    _cancel: CancelOnDrop,
}

impl tokio_stream::Stream for EventBody
{
    type Item = Result<String, core::convert::Infallible>;

    /// The next event, or a keep-alive comment when none arrived for
    /// [`HEARTBEAT`].
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: events arrive in order and end when the backend's sender is
    ///   gone; a keep-alive comment is yielded after each quiet interval.
    /// - provides: the response body.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    fn poll_next(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Option<Self::Item>>
    {
        let body = self.get_mut();
        if let core::task::Poll::Ready(event) = body.events.poll_recv(cx) {
            body.heartbeat.as_mut().reset(deadline());
            return core::task::Poll::Ready(event.map(Ok));
        }
        if body.heartbeat.as_mut().poll(cx).is_ready() {
            body.heartbeat.as_mut().reset(deadline());
            return core::task::Poll::Ready(Some(Ok(String::from(HEARTBEAT_COMMENT))));
        }
        return core::task::Poll::Pending;
    }
}

/// Run an aggregate request.
///
/// # Specification
/// - requires: nothing.
/// - ensures: 200 with the completion, or the failure's error response; if the
///   handler is dropped first, the request is cancelled; the request's start
///   and end are logged.
/// - provides: non-streamed completions.
/// - fails: never; failures are responses.
/// - panics: none.
async fn aggregate(
    shared: Arc<Shared>,
    identity: Identity,
    request: ChatRequest,
) -> Response
{
    let cancel = CancelToken::new();
    let _guard = CancelOnDrop(cancel.clone());
    let backend = Arc::clone(&shared.backend);
    let shape = RequestShape::of(shared.next_request(), &request);
    let ran = tokio::task::spawn_blocking(move || {
        let mut aggregate = Aggregate;
        let mut events = Logged::new(&mut aggregate, shape);
        let ran = backend
            .run(&request, &mut events, &cancel)
            .map_err(ApiError::from_failure);
        events.end(ran.as_ref());
        return ran;
    })
    .await;
    return match ran {
        | Ok(Ok(outcome)) => json_response(StatusCode::OK, completion(&identity, &outcome)),
        | Ok(Err(error)) => error_response(&error),
        | Err(join) => error_response(&ApiError::internal(join.to_string())),
    };
}

/// Run a streamed request.
///
/// # Specification
/// - requires: nothing.
/// - ensures: a failure before submission is an error response; after it, a 200
///   event stream of the opening chunk, the deltas, and either the closing
///   chunks or an error event; dropping the handler or the body cancels the
///   request; the request's start and end are logged.
/// - provides: streamed completions.
/// - fails: never; failures are responses or events.
/// - panics: none.
async fn streamed(
    shared: Arc<Shared>,
    parsed: ParsedChat,
    identity: Identity,
) -> Response
{
    let cancel = CancelToken::new();
    let guard = CancelOnDrop(cancel.clone());
    let (decision, decided) = tokio::sync::oneshot::channel();
    let (body, events) = tokio::sync::mpsc::unbounded_channel();
    let backend = Arc::clone(&shared.backend);
    let request = parsed.request;
    let encoder = ChunkStream::new(identity, parsed.include_usage);
    let shape = RequestShape::of(shared.next_request(), &request);
    let _running = tokio::task::spawn_blocking(move || {
        let mut sink = Streamed {
            encoder,
            body,
            begun: Begun::Pending(decision),
        };
        let mut events = Logged::new(&mut sink, shape);
        let ran = backend
            .run(&request, &mut events, &cancel)
            .map_err(ApiError::from_failure);
        events.end(ran.as_ref());
        let terminal = match ran {
            | Err(error) => {
                if let Begun::Pending(decision) =
                    core::mem::replace(&mut sink.begun, Begun::Streaming)
                {
                    let _gone = decision.send(Err(error));
                    return;
                }
                vec![error_event(&error)]
            },
            | Ok(outcome) => {
                sink.begin();
                match sink.encoder.finish(&outcome) {
                    | Ok(events) => events,
                    | Err(error) => vec![error_event(&error)],
                }
            },
        };
        for event in terminal {
            let _closed = sink.body.send(event);
        }
    });
    return match decided.await {
        | Ok(Ok(())) => {
            let body = EventBody {
                events,
                heartbeat: Box::pin(tokio::time::sleep(HEARTBEAT)),
                _cancel: guard,
            };
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, "text/event-stream"),
                    (header::CACHE_CONTROL, "no-cache"),
                    (header::HeaderName::from_static("x-accel-buffering"), "no"),
                ],
                Body::from_stream(body),
            )
                .into_response()
        },
        | Ok(Err(error)) => error_response(&error),
        | Err(_closed) => error_response(&ApiError::internal(String::from(
            "the generation ended without a result",
        ))),
    };
}

/// `POST /v1/chat/completions`.
///
/// # Specification
/// - requires: nothing.
/// - ensures: a body over the configured cap is 413 `request_too_large` with
///   ninfer's message; a body that is not JSON is 400; a refused body is its
///   translation error; another model id is 404 `model_not_found`; otherwise
///   the request runs aggregate or streamed as it asks.
/// - provides: chat completions.
/// - fails: never; failures are responses.
/// - panics: none.
async fn chat(
    State(shared): State<Arc<Shared>>,
    body: Result<Bytes, axum::extract::rejection::BytesRejection>,
) -> Response
{
    let body = match body {
        | Ok(body) => body,
        | Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            return error_response(&ApiError {
                status: StatusCode::PAYLOAD_TOO_LARGE,
                kind: "invalid_request_error",
                message: format!(
                    "request body exceeds the configured payload limit of {} bytes",
                    shared.config.max_request.0
                ),
                param: String::new(),
                code: String::from("request_too_large"),
            });
        },
        | Err(rejection) => {
            return error_response(&ApiError::invalid(
                rejection.body_text(),
                Param::NONE,
                Code::NONE,
            ));
        },
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&body)
    else {
        return error_response(&ApiError::invalid(
            String::from("request body is not valid JSON"),
            Param::NONE,
            Code::NONE,
        ));
    };
    let parsed = match chat_request(&value, shared.config.defaults, FreshSeed(fresh_bits().0)) {
        | Ok(parsed) => parsed,
        | Err(error) => return error_response(&error),
    };
    if parsed.model != shared.config.model {
        return error_response(&model_not_found(&parsed.model));
    }
    let identity = Identity::new(parsed.model.0.clone());
    return match parsed.delivery {
        | Delivery::Aggregate => aggregate(shared, identity, parsed.request).await,
        | Delivery::Streaming => streamed(shared, parsed, identity).await,
    };
}

/// Run one short request, so the first client request does not pay for the
/// backend's first-use costs.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success the backend has generated up to four tokens for a
///   one-turn `hi` prompt, with model sampling defaults, the default thinking
///   budget `thinking_budget`, and the prefix cache neither read nor written,
///   as ninfer's server warms up; a warm-up of a quarter second or more logs
///   `warmup complete | <duration>`, as ninfer's does at its info level.
/// - provides: the start-up warm-up.
/// - fails: when the request fails.
/// - panics: none.
///
/// # Errors
/// - [`infinitum_chat::ChatFailure`]: the warm-up request failed.
///
/// # Adequacy
/// - hypothesis: L3 on the request's shape, the property that keeps the warm-up
///   out of the prefix cache and cheap.
/// - witness: `tests::warm_up_neither_reads_nor_writes_the_prefix_cache`
#[inline]
pub fn warm_up(
    backend: &dyn ChatBackend,
    thinking_budget: infinitum_chat::ThinkingBudget,
) -> Result<(), infinitum_chat::ChatFailure>
{
    let request = ChatRequest {
        prompt: infinitum_chat::Prompt {
            messages: vec![infinitum_chat::Message {
                role: infinitum_chat::Role::User,
                parts: vec![String::from("hi")],
                reasoning: String::new(),
                tool_calls: Vec::new(),
                tool_call_id: String::new(),
            }],
            tools: Vec::new(),
            template_arguments: String::from("{}"),
            thinking: infinitum_chat::Switch::ModelDefault,
            preserve_thinking: infinitum_chat::Switch::ModelDefault,
            effort: infinitum_chat::Effort::Unrequested,
            cache: infinitum_chat::PromptCache {
                markers: Vec::new(),
                structural: infinitum_chat::StructuralPrefixes::Allowed,
            },
        },
        generation: infinitum_chat::Generation {
            output_tokens: TokenCount::from(4_u32),
            sampling: infinitum_chat::Sampling::MODEL_DEFAULT,
            seed: fresh_bits().0,
            post_thinking: infinitum_chat::Sampling::MODEL_DEFAULT,
            post_thinking_seed: infinitum_chat::Seed::Inherited,
            thinking_budget,
            stops: Vec::new(),
            stop_scope: infinitum_chat::StopScope::Content,
            special_tokens: infinitum_chat::SpecialTokens::Trimmed,
            tool_name_limit: crate::parse::TOOL_NAME_LIMIT,
            prefix_reuse: infinitum_chat::PrefixReuse::Disabled,
        },
        delivery: Delivery::Aggregate,
    };
    let started = std::time::Instant::now();
    backend.run(&request, &mut Aggregate, &CancelToken::new())?;
    let elapsed = started.elapsed();
    if elapsed.as_secs_f64() >= 0.25_f64 {
        crate::oplog::emit(&crate::oplog::Record {
            severity: crate::oplog::Severity::Info,
            message: format!(
                "warmup complete | {}",
                crate::pretty::Duration(elapsed.as_secs_f64())
            ),
        });
    }
    return Ok(());
}

/// Serve the chat surface on `listener` until an interrupt or termination
/// signal asks it to stop.
///
/// # Specification
/// - requires: called within a multi-threaded tokio runtime with timers and
///   signal handling.
/// - ensures: the ready line is logged, then the four routes answer as their
///   handlers specify, behind the gate, with request bodies capped at the
///   configured size, while throughput is logged at the configured interval; on
///   SIGINT or SIGTERM (Ctrl-C off Unix) the server stops accepting, finishes
///   its open connections, logs `server stopped` and returns, as ninfer's
///   server stops on those signals.
/// - provides: `infinitum serve`'s server.
/// - fails: when the listener has no address, the signal handlers cannot be
///   installed, or accepting fails.
/// - panics: none.
///
/// # Errors
/// - [`std::io::Error`]: the server could not start or failed.
///
/// # Adequacy
/// - hypothesis: L2 — `tests::routes_answer_through_the_gate` drives the router
///   with a scripted backend and `tests::a_termination_signal_stops_the_server`
///   stops a running server; the served comparison against `ninfer-serve`
///   checks the engine path end to end.
/// - witness: `tests::routes_answer_through_the_gate`
/// - witness: `tests::a_termination_signal_stops_the_server`
#[inline]
pub async fn serve(
    listener: tokio::net::TcpListener,
    backend: Arc<dyn ChatBackend>,
    config: ServeConfig,
) -> Result<(), std::io::Error>
{
    let stop = stop_requested()?;
    crate::oplog::emit(&crate::oplog::listening(
        listener.local_addr()?,
        &config.model,
        &config.access,
    ));
    if let StatsInterval::Every(period) = config.stats {
        tokio::spawn(report_throughput(
            Arc::clone(&backend),
            period,
            crate::oplog::emit,
        ));
    }
    axum::serve(listener, router(backend, config))
        .with_graceful_shutdown(stop)
        .await?;
    crate::oplog::emit(&crate::oplog::Record {
        severity: crate::oplog::Severity::Info,
        message: String::from("server stopped"),
    });
    return Ok(());
}

/// A future that completes when SIGINT or SIGTERM arrives.
///
/// # Specification
/// - requires: called within a tokio runtime with signal handling.
/// - ensures: both handlers are installed before it returns, replacing an
///   inherited ignore, and the future completes at the first of either signal.
/// - provides: [`serve`]'s stop request.
/// - fails: when a handler cannot be installed.
/// - panics: none.
///
/// # Errors
/// - [`std::io::Error`]: a handler could not be installed.
///
/// # Adequacy
/// - hypothesis: L2 — the test sends the process SIGTERM once the server
///   answers, and the server returns.
/// - witness: `tests::a_termination_signal_stops_the_server`
#[cfg(unix)]
fn stop_requested() -> Result<impl Future<Output = ()>, std::io::Error>
{
    use tokio::signal::unix::SignalKind;
    use tokio::signal::unix::signal;

    let mut interrupt = signal(SignalKind::interrupt())?;
    let mut terminate = signal(SignalKind::terminate())?;
    return Ok(core::future::poll_fn(move |cx| {
        if interrupt.poll_recv(cx).is_ready() || terminate.poll_recv(cx).is_ready() {
            return core::task::Poll::Ready(());
        }
        return core::task::Poll::Pending;
    }));
}

/// A future that completes on Ctrl-C.
///
/// # Specification
/// - requires: called within a tokio runtime with signal handling.
/// - ensures: completes at the first Ctrl-C; never, when Ctrl-C cannot be
///   watched.
/// - provides: [`serve`]'s stop request off Unix.
/// - fails: never.
/// - panics: none.
///
/// # Errors
/// - [`std::io::Error`]: never; the signature matches the Unix form.
///
/// # Adequacy
/// - hypothesis: L1 — console events are process-wide; the Unix form carries
///   the tested stop.
/// - witness: `tests::a_termination_signal_stops_the_server`
#[cfg(not(unix))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "the Unix form installs handlers that can fail"
)]
fn stop_requested() -> Result<impl Future<Output = ()>, std::io::Error>
{
    return Ok(async {
        if tokio::signal::ctrl_c().await.is_err() {
            core::future::pending::<()>().await;
        }
    });
}

/// Write the backend's throughput every `period` through `write`, as
/// ninfer's stats reporter logs it.
///
/// # Specification
/// - requires: called within a tokio runtime with timers.
/// - ensures: each period, the counters are sampled and the
///   [`crate::oplog::throughput`] line for the interval since the last sample
///   is written when it saw activity; a late tick waits a full period from when
///   it ran, as ninfer resets a missed deadline; a failed sample writes one
///   warning and ends the reporter.
/// - provides: the periodic throughput log.
/// - fails: never; a failed sample ends the reporter, not the server.
/// - panics: none.
///
/// # Adequacy
/// - hypothesis: L3 under a paused clock on a quiet interval, an active one and
///   a failed sample.
/// - witness: `tests::throughput_reports_active_intervals_until_a_sample_fails`
async fn report_throughput<Write>(
    backend: Arc<dyn ChatBackend>,
    period: core::time::Duration,
    mut write: Write,
) where
    Write: FnMut(&crate::oplog::Record),
{
    let stopped = |failure: &infinitum_chat::ChatFailure| {
        return crate::oplog::Record {
            severity: crate::oplog::Severity::Warning,
            message: format!("throughput reporting stopped | {}", failure.message),
        };
    };
    let mut previous = match backend.counters() {
        | Ok(counters) => counters,
        | Err(failure) => return write(&stopped(&failure)),
    };
    let mut previous_time = tokio::time::Instant::now();
    let mut ticks = tokio::time::interval_at(
        previous_time.checked_add(period).unwrap_or(previous_time),
        period,
    );
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        ticks.tick().await;
        let current = match backend.counters() {
            | Ok(counters) => counters,
            | Err(failure) => return write(&stopped(&failure)),
        };
        let now = tokio::time::Instant::now();
        if let crate::oplog::Throughput::Active(record) =
            crate::oplog::throughput(&previous, &current, now.duration_since(previous_time))
        {
            write(&record);
        }
        previous = current;
        previous_time = now;
    }
}

/// The router.
///
/// # Specification
/// trivial.
fn router(
    backend: Arc<dyn ChatBackend>,
    config: ServeConfig,
) -> axum::Router
{
    let limit = config.max_request.0.get();
    let shared = Arc::new(Shared {
        backend,
        config,
        requests: core::sync::atomic::AtomicU64::new(0),
    });
    return axum::Router::new()
        .route("/health", axum::routing::get(health))
        .route("/v1/models", axum::routing::get(models))
        .route("/v1/models/{*id}", axum::routing::get(model))
        .route("/v1/chat/completions", axum::routing::post(chat))
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&shared),
            gate,
        ))
        .layer(axum::extract::DefaultBodyLimit::max(limit))
        .with_state(shared);
}

/// Tests for the gate and the routes.
#[cfg(test)]
mod tests
{
    use alloc::sync::Arc;

    use axum::body::Body;
    use axum::http::HeaderValue;
    use axum::http::Request;
    use axum::http::StatusCode;
    use infinitum_chat::Admission;
    use infinitum_chat::CancelToken;
    use infinitum_chat::Channel;
    use infinitum_chat::ChatBackend;
    use infinitum_chat::ChatEvents;
    use infinitum_chat::ChatFailure;
    use infinitum_chat::ChatOutcome;
    use infinitum_chat::ChatRequest;
    use infinitum_chat::Delivery;
    use infinitum_chat::DeltaText;
    use infinitum_chat::FailureKind;
    use infinitum_chat::Finish;
    use infinitum_chat::ModelName;
    use infinitum_chat::PrefixReuse;
    use infinitum_chat::Tally;
    use infinitum_round::TokenCount;
    use tower::ServiceExt as _;

    use super::Access;
    use super::ApiKey;
    use super::ModelId;
    use super::RequestBytes;
    use super::ServeConfig;
    use super::Verdict;
    use super::bearer_matches;
    use super::router;
    use super::warm_up;
    use crate::parse::Defaults;

    /// A backend that streams "a" then "b", or fails when the prompt's
    /// first turn says "fail".
    #[repr(transparent)]
    struct Scripted(ModelName);

    impl ChatBackend for Scripted
    {
        /// The scripted name.
        ///
        /// # Specification
        /// trivial.
        fn model_name(&self) -> &ModelName
        {
            return &self.0;
        }

        /// No activity.
        ///
        /// # Specification
        /// trivial.
        fn counters(&self) -> Result<infinitum_chat::RuntimeCounters, ChatFailure>
        {
            return Ok(infinitum_chat::RuntimeCounters::default());
        }

        /// Answer "ab", streaming it as two deltas, or fail before
        /// submission when the first turn says "fail".
        ///
        /// # Specification
        /// - requires: nothing.
        /// - ensures: a streamed request is submitted, admitted, and sent `a`
        ///   then `b`; the outcome's content is `ab`.
        /// - provides: the router tests' backend.
        /// - fails: with a context-length failure for a "fail" prompt.
        /// - panics: none.
        ///
        /// # Errors
        /// - [`ChatFailure`]: the prompt says "fail".
        fn run(
            &self,
            request: &ChatRequest,
            events: &mut dyn ChatEvents,
            _cancel: &CancelToken,
        ) -> Result<ChatOutcome, ChatFailure>
        {
            let first = request
                .prompt
                .messages
                .first()
                .and_then(|turn| return turn.parts.first());
            if first.map(String::as_str) == Some("fail") {
                return Err(ChatFailure {
                    kind: FailureKind::ContextLength,
                    message: String::from("too long"),
                });
            }
            let admission = Admission {
                prompt_tokens: TokenCount::from(3_u32),
                reused_tokens: TokenCount::ZERO,
            };
            events.submitted(infinitum_chat::Submission {
                thinking: infinitum_chat::Thinking::Closed,
                thinking_budget: infinitum_chat::ThinkingBudget::Unlimited,
            });
            events.admitted(admission);
            if request.delivery == Delivery::Streaming {
                events.publish(Channel::Content, DeltaText("a"));
                events.publish(Channel::Content, DeltaText("b"));
            }
            return Ok(ChatOutcome {
                content: String::from("ab"),
                reasoning: String::new(),
                tool_calls: Vec::new(),
                finish: Finish::StopToken,
                admission,
                generated: Vec::new(),
                reasoning_tokens: TokenCount::ZERO,
                prompt_wall: core::time::Duration::ZERO,
                generation_wall: core::time::Duration::ZERO,
                drafted: Tally(0),
                accepted: Tally(0),
                telemetry: infinitum_chat::Telemetry {
                    first_token: core::time::Duration::ZERO,
                    total: core::time::Duration::ZERO,
                    prefill: core::time::Duration::ZERO,
                    decode: core::time::Duration::ZERO,
                    queue_wait: core::time::Duration::ZERO,
                    reuse: infinitum_chat::ReusePath::Root,
                    thinking: infinitum_chat::ThinkingSpend {
                        budget: infinitum_chat::ThinkingBudget::Unlimited,
                        model_tokens: TokenCount::ZERO,
                        injected_tokens: TokenCount::ZERO,
                    },
                    accepted_per_position: Vec::new(),
                },
            });
        }
    }

    /// A router over [`Scripted`] behind key `k`.
    ///
    /// # Specification
    /// trivial.
    fn app() -> axum::Router
    {
        return router(
            Arc::new(Scripted(ModelName(String::from("m")))),
            ServeConfig {
                model: ModelId(String::from("m")),
                access: Access::Key(ApiKey(String::from("k"))),
                max_model_len: TokenCount::from(64_u32),
                defaults: Defaults {
                    output_tokens: TokenCount::from(8_u32),
                    thinking_budget: infinitum_chat::ThinkingBudget::Unlimited,
                },
                max_request: RequestBytes(core::num::NonZeroUsize::new(1024).unwrap()),
                stats: super::StatsInterval::Off,
            },
        );
    }

    /// Send a request and read the status and body.
    ///
    /// # Specification
    /// - requires: the body is UTF-8.
    /// - ensures: the router's status and whole body.
    /// - provides: every router test's round trip.
    /// - fails: never.
    /// - panics: when the router or the body fails, failing the test.
    async fn send(request: Request<Body>) -> (StatusCode, String)
    {
        let response = app().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        return (status, String::from_utf8(bytes.to_vec()).unwrap());
    }

    /// A keyed chat request with `body`.
    ///
    /// # Specification
    /// trivial.
    fn chat(body: Body) -> Request<Body>
    {
        return Request::post("/v1/chat/completions")
            .header("authorization", "Bearer k")
            .body(body)
            .unwrap();
    }

    /// A backend that succeeds only for ninfer's warm-up request.
    #[repr(transparent)]
    struct WarmUpOnly(ModelName);

    impl ChatBackend for WarmUpOnly
    {
        /// The backend's name.
        ///
        /// # Specification
        /// trivial.
        fn model_name(&self) -> &ModelName
        {
            return &self.0;
        }

        /// No activity.
        ///
        /// # Specification
        /// trivial.
        fn counters(&self) -> Result<infinitum_chat::RuntimeCounters, ChatFailure>
        {
            return Ok(infinitum_chat::RuntimeCounters::default());
        }

        /// Succeed exactly for four aggregate tokens that neither read nor
        /// write the prefix cache.
        ///
        /// # Specification
        /// - requires: nothing.
        /// - ensures: `Ok` exactly for the warm-up's shape.
        /// - provides: the warm-up test's check.
        /// - fails: for any other request.
        /// - panics: none.
        ///
        /// # Errors
        /// - [`ChatFailure`]: the request is not the warm-up's shape.
        fn run(
            &self,
            request: &ChatRequest,
            _events: &mut dyn ChatEvents,
            _cancel: &CancelToken,
        ) -> Result<ChatOutcome, ChatFailure>
        {
            let generation = &request.generation;
            if generation.output_tokens != TokenCount::from(4_u32)
                || generation.prefix_reuse != PrefixReuse::Disabled
                || request.delivery != Delivery::Aggregate
            {
                return Err(ChatFailure {
                    kind: FailureKind::InvalidPrompt,
                    message: format!("not the warm-up: {request:?}"),
                });
            }
            return Ok(ChatOutcome {
                content: String::new(),
                reasoning: String::new(),
                tool_calls: Vec::new(),
                finish: Finish::OutputLimit,
                admission: Admission {
                    prompt_tokens: TokenCount::from(9_u32),
                    reused_tokens: TokenCount::ZERO,
                },
                generated: Vec::new(),
                reasoning_tokens: TokenCount::ZERO,
                prompt_wall: core::time::Duration::ZERO,
                generation_wall: core::time::Duration::ZERO,
                drafted: Tally(0),
                accepted: Tally(0),
                telemetry: infinitum_chat::Telemetry {
                    first_token: core::time::Duration::ZERO,
                    total: core::time::Duration::ZERO,
                    prefill: core::time::Duration::ZERO,
                    decode: core::time::Duration::ZERO,
                    queue_wait: core::time::Duration::ZERO,
                    reuse: infinitum_chat::ReusePath::Root,
                    thinking: infinitum_chat::ThinkingSpend {
                        budget: infinitum_chat::ThinkingBudget::Unlimited,
                        model_tokens: TokenCount::ZERO,
                        injected_tokens: TokenCount::ZERO,
                    },
                    accepted_per_position: Vec::new(),
                },
            });
        }
    }

    #[test]
    fn warm_up_neither_reads_nor_writes_the_prefix_cache()
    {
        assert_eq!(
            warm_up(
                &WarmUpOnly(ModelName(String::from("m"))),
                infinitum_chat::ThinkingBudget::Unlimited
            ),
            Ok(()),
            "the warm-up is ninfer's: four aggregate uncached tokens"
        );
    }

    #[test]
    fn bearer_matching_follows_ninfer()
    {
        let key = ApiKey(String::from("k"));
        assert_eq!(
            bearer_matches(&HeaderValue::from_static(" \tbearer  k \t"), &key),
            Verdict::Admitted,
            "case and blanks are forgiven"
        );
        assert_eq!(
            bearer_matches(&HeaderValue::from_static("Bearerk"), &key),
            Verdict::Refused,
            "the separator is required"
        );
        assert_eq!(
            bearer_matches(&HeaderValue::from_static("Bearer j"), &key),
            Verdict::Refused,
            "a wrong key is refused"
        );
        assert_eq!(
            bearer_matches(&HeaderValue::from_static("Basic k"), &key),
            Verdict::Refused,
            "another scheme is refused"
        );
    }

    #[test]
    fn an_empty_key_leaves_the_api_open()
    {
        assert_eq!(
            Access::of(ApiKey(String::new())),
            Access::Open,
            "an empty key means no authentication, as ninfer's, never an empty credential"
        );
        assert_eq!(
            Access::of(ApiKey(String::from("k"))),
            Access::Key(ApiKey(String::from("k"))),
            "a key gates the API"
        );
    }

    #[tokio::test]
    async fn routes_answer_through_the_gate()
    {
        let (status, _body) = send(Request::get("/health").body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "health needs no key");
        let (status, body) = send(Request::get("/v1/models").body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "models need the key");
        assert!(
            body.contains("invalid_api_key"),
            "the refusal names its code"
        );
        let (status, body) = send(
            Request::get("/v1/models")
                .header("x-api-key", "k")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "x-api-key is accepted");
        assert!(
            body.contains(r#""max_model_len":64"#),
            "the ceiling is reported"
        );
        let (status, body) = send(chat(Body::from(
            r#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#,
        )))
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "another model is not found");
        assert!(body.contains("model_not_found"), "the error names its code");
        let (status, body) = send(chat(Body::from(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
        )))
        .await;
        assert_eq!(status, StatusCode::OK, "an aggregate request completes");
        assert!(
            body.contains(r#""content":"ab""#),
            "the completion carries the content"
        );
        let (status, body) = send(chat(Body::from(
            r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        )))
        .await;
        assert_eq!(status, StatusCode::OK, "a streamed request completes");
        let deltas = body
            .split("\n\n")
            .filter(|event| return event.contains(r#""delta":{"content":"#))
            .count();
        assert!(
            body.starts_with(r#"data: {"id":"chatcmpl-"#),
            "the opening chunk comes first: {body}"
        );
        assert_eq!(deltas, 2, "both deltas are streamed: {body}");
        assert!(body.ends_with("data: [DONE]\n\n"), "the stream closes");
        let (status, body) = send(chat(Body::from(
            r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"fail"}]}"#,
        )))
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "a failure before submission is an HTTP error"
        );
        assert!(
            body.contains("context_length_exceeded"),
            "the failure keeps its code"
        );
        let (status, _body) = send(chat(Body::from("{"))).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "a body that is not JSON is refused"
        );
        let (status, body) = send(chat(Body::from(" ".repeat(1024)))).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "a body at the cap is read: {body}"
        );
        let (status, body) = send(chat(Body::from(" ".repeat(1025)))).await;
        assert_eq!(
            status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "a body past the cap is refused"
        );
        assert!(
            body.contains(r#""code":"request_too_large""#) && body.contains("limit of 1024 bytes"),
            "the refusal is ninfer's: {body}"
        );
    }

    /// A backend whose counter samples are scripted, oldest first.
    struct Sampled
    {
        /// Its name.
        model: ModelName,
        /// The samples still to give.
        samples: std::sync::Mutex<Vec<Result<infinitum_chat::RuntimeCounters, ChatFailure>>>,
    }

    impl ChatBackend for Sampled
    {
        /// Its name.
        ///
        /// # Specification
        /// trivial.
        fn model_name(&self) -> &ModelName
        {
            return &self.model;
        }

        /// The next scripted sample; a failure once the script is spent.
        ///
        /// # Specification
        /// trivial.
        fn counters(&self) -> Result<infinitum_chat::RuntimeCounters, ChatFailure>
        {
            let spent = ChatFailure {
                kind: FailureKind::Internal,
                message: String::from("script spent"),
            };
            return match self.samples.lock() {
                | Ok(mut samples) if !samples.is_empty() => samples.remove(0),
                | _ => Err(spent),
            };
        }

        /// Refuse; the reporter never runs a request.
        ///
        /// # Specification
        /// trivial.
        fn run(
            &self,
            _request: &ChatRequest,
            _events: &mut dyn ChatEvents,
            _cancel: &CancelToken,
        ) -> Result<ChatOutcome, ChatFailure>
        {
            return Err(ChatFailure {
                kind: FailureKind::Internal,
                message: String::from("not a chat backend"),
            });
        }
    }

    /// A quiet interval writes nothing, an active one writes its line over the
    /// full period, and a failed sample writes one warning and ends the
    /// reporter.
    #[tokio::test(start_paused = true)]
    async fn throughput_reports_active_intervals_until_a_sample_fails()
    {
        let idle = infinitum_chat::RuntimeCounters::default();
        let busy = infinitum_chat::RuntimeCounters {
            decode_tokens: Tally(50),
            ..idle
        };
        let backend = Sampled {
            model: ModelName(String::from("m")),
            samples: std::sync::Mutex::new(vec![
                Ok(idle),
                Ok(idle),
                Ok(busy),
                Err(ChatFailure {
                    kind: FailureKind::Internal,
                    message: String::from("gone"),
                }),
            ]),
        };
        let mut lines = Vec::new();
        super::report_throughput(
            Arc::new(backend),
            core::time::Duration::from_secs(5),
            |record: &crate::oplog::Record| lines.push(record.clone()),
        )
        .await;
        assert_eq!(
            lines,
            [
                crate::oplog::Record {
                    severity: crate::oplog::Severity::Info,
                    message: String::from(
                        "throughput | 5.0s | decode 10.0 tok/s (50 tok) | running 0 | host 0.0% (0 us)"
                    ),
                },
                crate::oplog::Record {
                    severity: crate::oplog::Severity::Warning,
                    message: String::from("throughput reporting stopped | gone"),
                },
            ],
            "one active line, then the warning"
        );
    }

    /// A running server stops and returns once the process receives SIGTERM.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_termination_signal_stops_the_server()
    {
        use std::io::Read as _;
        use std::io::Write as _;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(super::serve(
            listener,
            Arc::new(Scripted(ModelName(String::from("m")))),
            ServeConfig {
                model: ModelId(String::from("m")),
                access: Access::Open,
                max_model_len: TokenCount::from(64_u32),
                defaults: Defaults {
                    output_tokens: TokenCount::from(16_u32),
                    thinking_budget: infinitum_chat::ThinkingBudget::Unlimited,
                },
                max_request: RequestBytes(core::num::NonZeroUsize::new(1024).unwrap()),
                stats: super::StatsInterval::Off,
            },
        ));
        let answered = tokio::task::spawn_blocking(move || {
            let mut stream = std::net::TcpStream::connect(address).unwrap();
            stream
                .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                .unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).unwrap();
            return response;
        })
        .await
        .unwrap();
        assert!(
            answered.starts_with("HTTP/1.1 200"),
            "the server answered: {answered}"
        );
        let sent = std::process::Command::new("kill")
            .args(["-TERM", &std::process::id().to_string()])
            .status()
            .unwrap();
        assert!(sent.success(), "kill ran");
        let stopped = tokio::time::timeout(core::time::Duration::from_secs(10), server)
            .await
            .unwrap()
            .unwrap();
        assert!(stopped.is_ok(), "the server returned cleanly");
    }
}
