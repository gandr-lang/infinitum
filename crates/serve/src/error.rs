//! The `OpenAI` error shape: an HTTP status and a body naming the message,
//! type, parameter and code.

use infinitum_chat::ChatFailure;
use infinitum_chat::FailureKind;

/// The request parameter an error names; empty names none.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Param<'name>(pub &'name str);

impl Param<'_>
{
    /// No parameter: the error body's `param` is `null`.
    pub const NONE: Param<'static> = Param("");
}

/// The machine-readable code an error carries; empty carries none.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Code<'name>(pub &'name str);

impl Code<'_>
{
    /// No code: the error body's `code` is `null`.
    pub const NONE: Code<'static> = Code("");
}

/// One API error, as the response renders it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError
{
    /// The HTTP status.
    pub status: axum::http::StatusCode,
    /// The error's `type`.
    pub kind: &'static str,
    /// The human-readable message.
    pub message: String,
    /// The offending parameter; empty renders as `null`.
    pub param: String,
    /// The machine-readable code; empty renders as `null`.
    pub code: String,
}

impl ApiError
{
    /// A 400 `invalid_request_error` naming `param` and `code`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn invalid(
        message: String,
        param: Param<'_>,
        code: Code<'_>,
    ) -> Self
    {
        return Self {
            status: axum::http::StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message,
            param: String::from(param.0),
            code: String::from(code.0),
        };
    }

    /// A 500 `internal_error` carrying `message`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn internal(message: String) -> Self
    {
        return Self {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            kind: "internal_error",
            message,
            param: String::new(),
            code: String::new(),
        };
    }

    /// The error a backend failure renders as.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: each failure class takes the status, type, parameter and code
    ///   ninfer's server gives the same class: 400 for an invalid prompt, an
    ///   over-long prompt and an unplaceable thinking budget; 429 for a full
    ///   queue; 503 for a pending timeout and an unavailable backend; 499 for a
    ///   cancellation; 500 otherwise. The message is the backend's.
    /// - provides: the one mapping from backend failures to responses.
    /// - fails: never.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 — every failure class.
    /// - witness: `tests::each_failure_class_maps_to_its_status_and_code`
    #[inline]
    #[must_use]
    pub fn from_failure(failure: ChatFailure) -> Self
    {
        let (status, kind, param, code) = match failure.kind {
            | FailureKind::InvalidPrompt => {
                (400, "invalid_request_error", "messages", "invalid_prompt")
            },
            | FailureKind::ContextLength => (
                400,
                "invalid_request_error",
                "messages",
                "context_length_exceeded",
            ),
            | FailureKind::ThinkingBudgetCapacity => (
                400,
                "invalid_request_error",
                "",
                "thinking_budget_capacity_insufficient",
            ),
            | FailureKind::Overloaded => (429, "rate_limit_error", "", "server_overloaded"),
            | FailureKind::QueueTimeout => (503, "server_error", "", "request_queue_timeout"),
            | FailureKind::Cancelled => (499, "request_cancelled", "", "client_disconnected"),
            | FailureKind::Unavailable => (503, "server_error", "", "service_unavailable"),
            | FailureKind::Internal => (500, "internal_error", "", ""),
        };
        return Self {
            status: axum::http::StatusCode::from_u16(status)
                .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
            kind,
            message: failure.message,
            param: String::from(param),
            code: String::from(code),
        };
    }

    /// The JSON body: `{"error":{"message","type","param","code"}}`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the body names the four fields in that order; an empty
    ///   parameter or code is `null`.
    /// - provides: the error body of a response or a stream's error event.
    /// - fails: never.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 — an error with and one without a parameter and code.
    /// - witness: `tests::empty_fields_render_as_null`
    #[inline]
    #[must_use]
    pub fn body(&self) -> String
    {
        let nullable = |text: &str| {
            if text.is_empty() {
                return serde_json::Value::Null;
            }
            return serde_json::Value::from(text);
        };
        let mut error = serde_json::Map::new();
        error.insert(
            String::from("message"),
            serde_json::Value::from(self.message.as_str()),
        );
        error.insert(String::from("type"), serde_json::Value::from(self.kind));
        error.insert(String::from("param"), nullable(&self.param));
        error.insert(String::from("code"), nullable(&self.code));
        let mut body = serde_json::Map::new();
        body.insert(String::from("error"), serde_json::Value::Object(error));
        return serde_json::Value::Object(body).to_string();
    }
}

/// Tests for the error mapping and rendering.
#[cfg(test)]
mod tests
{
    use infinitum_chat::ChatFailure;
    use infinitum_chat::FailureKind;

    use super::ApiError;
    use super::Code;
    use super::Param;

    #[test]
    fn each_failure_class_maps_to_its_status_and_code()
    {
        let cases = [
            (FailureKind::InvalidPrompt, 400_u16, "invalid_prompt"),
            (FailureKind::ContextLength, 400, "context_length_exceeded"),
            (
                FailureKind::ThinkingBudgetCapacity,
                400,
                "thinking_budget_capacity_insufficient",
            ),
            (FailureKind::Overloaded, 429, "server_overloaded"),
            (FailureKind::QueueTimeout, 503, "request_queue_timeout"),
            (FailureKind::Cancelled, 499, "client_disconnected"),
            (FailureKind::Unavailable, 503, "service_unavailable"),
            (FailureKind::Internal, 500, ""),
        ];
        for (kind, status, code) in cases {
            let error = ApiError::from_failure(ChatFailure {
                kind,
                message: String::from("m"),
            });
            assert_eq!(error.status.as_u16(), status, "{kind:?} takes its status");
            assert_eq!(error.code, code, "{kind:?} takes its code");
        }
    }

    #[test]
    fn empty_fields_render_as_null()
    {
        let error = ApiError::internal(String::from("boom"));
        assert_eq!(
            error.body(),
            r#"{"error":{"message":"boom","type":"internal_error","param":null,"code":null}}"#,
            "empty param and code render as null"
        );
        let invalid = ApiError::invalid(String::from("bad"), Param("stop"), Code("x"));
        assert_eq!(
            invalid.body(),
            r#"{"error":{"message":"bad","type":"invalid_request_error","param":"stop","code":"x"}}"#,
            "set param and code render as strings"
        );
    }
}
