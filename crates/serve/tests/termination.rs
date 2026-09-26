//! How the server stops on a termination signal.
//!
//! A signal reaches the whole process, not one test: it would stop, or kill,
//! every other test running beside it. So this test runs alone, in its own
//! binary, and signals that process.

#[cfg(unix)]
extern crate alloc;

#[cfg(test)]
#[cfg(unix)]
mod tests
{
    use alloc::sync::Arc;
    use std::io::Read as _;
    use std::io::Write as _;

    use infinitum_chat::CancelToken;
    use infinitum_chat::ChatBackend;
    use infinitum_chat::ChatEvents;
    use infinitum_chat::ChatFailure;
    use infinitum_chat::ChatOutcome;
    use infinitum_chat::ChatRequest;
    use infinitum_chat::FailureKind;
    use infinitum_chat::ModelName;
    use infinitum_round::TokenCount;
    use infinitum_serve::Access;
    use infinitum_serve::Defaults;
    use infinitum_serve::ModelId;
    use infinitum_serve::RequestBytes;
    use infinitum_serve::ServeConfig;
    use infinitum_serve::StatsInterval;

    /// A backend that is never asked to run: the test only reads `/health`.
    #[repr(transparent)]
    struct Idle(ModelName);

    impl ChatBackend for Idle
    {
        /// Its name.
        ///
        /// # Specification
        /// trivial.
        fn model_name(&self) -> &ModelName
        {
            return &self.0;
        }

        /// Refuse: no request is sent.
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
                message: String::from("idle"),
            });
        }

        /// No activity.
        ///
        /// # Specification
        /// trivial.
        fn counters(&self) -> Result<infinitum_chat::RuntimeCounters, ChatFailure>
        {
            return Ok(infinitum_chat::RuntimeCounters::default());
        }
    }

    /// A running server, reporting throughput, stops and returns once the
    /// process receives SIGTERM.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_termination_signal_stops_the_server()
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(infinitum_serve::serve(
            listener,
            Arc::new(Idle(ModelName(String::from("m")))),
            ServeConfig {
                model: ModelId(String::from("m")),
                access: Access::Open,
                max_model_len: TokenCount::from(64_u32),
                defaults: Defaults {
                    output_tokens: TokenCount::from(16_u32),
                    thinking_budget: infinitum_chat::ThinkingBudget::Unlimited,
                },
                max_request: RequestBytes(core::num::NonZeroUsize::new(1024).unwrap()),
                stats: StatsInterval::Every(core::time::Duration::from_millis(10)),
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
