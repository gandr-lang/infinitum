//! The serving-backend contract: run one request, publish as it commits, and
//! report the outcome or a classified failure.

use crate::outcome::Admission;
use crate::outcome::Channel;
use crate::outcome::ChatOutcome;
use crate::request::ChatRequest;

/// A model's name, as its artifact records it.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelName(pub String);

/// Committed text on one output channel.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeltaText<'text>(pub &'text str);

/// Whether a request's consumer asked to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cancellation
{
    /// Keep generating.
    Live,
    /// Stop at the next opportunity.
    Requested,
}

/// A request's cancellation flag, shared between its consumer and the
/// backend running it.
#[repr(transparent)]
#[derive(Debug, Clone, Default)]
pub struct CancelToken(alloc::sync::Arc<core::sync::atomic::AtomicBool>);

impl CancelToken
{
    /// A live token.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn new() -> Self
    {
        return Self::default();
    }

    /// Ask the backend to stop.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: every clone of this token reads [`Cancellation::Requested`]
    ///   from now on.
    /// - provides: the consumer's stop signal.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    pub fn request(&self)
    {
        self.0.store(true, core::sync::atomic::Ordering::Release);
    }

    /// Read the flag.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: [`Cancellation::Requested`] exactly after some clone called
    ///   [`CancelToken::request`].
    /// - provides: the backend's poll.
    /// - fails: never.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 — a fresh token is live, and a request through one
    ///   clone is seen through another.
    /// - witness: `tests::a_request_through_a_clone_is_seen`
    #[inline]
    #[must_use]
    pub fn state(&self) -> Cancellation
    {
        if self.0.load(core::sync::atomic::Ordering::Acquire) {
            return Cancellation::Requested;
        }
        return Cancellation::Live;
    }
}

/// The consumer side of a streaming request, called on the thread running
/// [`ChatBackend::run`], in commit order.
pub trait ChatEvents
{
    /// The request passed preparation and was submitted; called once, before
    /// the admission. A failure before this call rejects the request; a
    /// failure after it ends a response already begun.
    ///
    /// # Specification
    /// - requires: called at most once per request, before
    ///   [`ChatEvents::admitted`].
    /// - ensures: implementation-defined.
    /// - provides: the point a streamed response may begin.
    /// - fails: never; as for [`ChatEvents::admitted`].
    /// - panics: none.
    fn submitted(&mut self);

    /// The request was admitted with this prompt accounting; called once,
    /// before any delta.
    ///
    /// # Specification
    /// - requires: called at most once per request, after
    ///   [`ChatEvents::submitted`] and before any [`ChatEvents::publish`].
    /// - ensures: implementation-defined.
    /// - provides: the admission record.
    /// - fails: never; a consumer that cannot keep up requests cancellation
    ///   through the request's [`CancelToken`].
    /// - panics: none.
    fn admitted(
        &mut self,
        admission: Admission,
    );

    /// Committed text on one channel.
    ///
    /// # Specification
    /// - requires: `text` is non-empty and follows every earlier delta.
    /// - ensures: implementation-defined.
    /// - provides: the streamed output.
    /// - fails: never; as for [`ChatEvents::admitted`].
    /// - panics: none.
    fn publish(
        &mut self,
        channel: Channel,
        text: DeltaText<'_>,
    );
}

/// How a request failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureKind
{
    /// The prompt or its options are invalid.
    InvalidPrompt,
    /// The prompt does not fit the context ceiling.
    ContextLength,
    /// The thinking budget leaves no room for its closing control tokens.
    ThinkingBudgetCapacity,
    /// The request queue is full.
    Overloaded,
    /// The request waited past its pending deadline.
    QueueTimeout,
    /// The consumer cancelled before admission.
    Cancelled,
    /// The backend is not serving.
    Unavailable,
    /// Anything else.
    Internal,
}

/// A failed request: its kind and the backend's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatFailure
{
    /// How it failed.
    pub kind: FailureKind,
    /// The backend's message.
    pub message: String,
}

impl core::fmt::Display for ChatFailure
{
    /// Render the backend's message.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str(&self.message);
    }
}

impl core::error::Error for ChatFailure
{
}

/// A backend that runs chat requests.
pub trait ChatBackend: Send + Sync
{
    /// The model's name, as the backend's artifact records it.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the same name on every call.
    /// - provides: the default public model id.
    /// - fails: never.
    /// - panics: none.
    fn model_name(&self) -> &ModelName;

    /// Run `request` to its outcome.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: under [`crate::Delivery::Streaming`], `events` receives the
    ///   submission once preparation succeeds, then the admission and every
    ///   committed delta in order, and the concatenated deltas of each channel
    ///   are a prefix of the outcome's text on that channel; under
    ///   [`crate::Delivery::Aggregate`] `events` is not called. Once `cancel`
    ///   reads requested the backend stops at its next opportunity, finishing
    ///   [`crate::Finish::Cancelled`] or failing [`FailureKind::Cancelled`]
    ///   before admission.
    /// - provides: one generation.
    /// - fails: with the failure's kind and message.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`ChatFailure`]: the request failed, classified by [`FailureKind`].
    fn run(
        &self,
        request: &ChatRequest,
        events: &mut dyn ChatEvents,
        cancel: &CancelToken,
    ) -> Result<ChatOutcome, ChatFailure>;
}

/// Tests for the cancellation flag.
#[cfg(test)]
mod tests
{
    use super::CancelToken;
    use super::Cancellation;

    #[test]
    fn a_request_through_a_clone_is_seen()
    {
        let token = CancelToken::new();
        let consumer = token.clone();
        assert_eq!(token.state(), Cancellation::Live, "a fresh token is live");
        consumer.request();
        assert_eq!(
            token.state(),
            Cancellation::Requested,
            "the request is seen through every clone"
        );
    }
}
