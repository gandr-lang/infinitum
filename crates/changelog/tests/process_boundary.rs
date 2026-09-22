//! What the binary writes when it fails.
//!
//! The distinction under test is invisible from inside the crate: a `main`
//! returning `Result` prints its error with `Debug`, and one that reports the
//! failure itself prints it with `Display`. Only the built binary can say which
//! happened, so this test runs it.

#[cfg(test)]
mod tests
{
    use std::path::Path;
    use std::process::Command;

    /// A failure reaches the operator as its message, and the process exits
    /// unsuccessfully.
    ///
    /// The failure is provoked without a forge, a network or a fixture history:
    /// run outside any repository, the first command the generator issues —
    /// `git rev-parse --abbrev-ref HEAD` — fails, and its failure takes the
    /// same reporting path as every other.
    #[test]
    fn a_failure_is_reported_through_display()
    {
        let outside_any_repository = Path::new("/");
        let output = Command::new(env!("CARGO_BIN_EXE_infinitum-changelog"))
            .current_dir(outside_any_repository)
            .env("GITHUB_EVENT_NAME", "pull_request")
            .env("GIT_CEILING_DIRECTORIES", "/")
            .output()
            .expect("the built binary runs");
        let reported = String::from_utf8(output.stderr).expect("the report is UTF-8");
        assert!(
            !output.status.success(),
            "a failure exits unsuccessfully: {:?}",
            output.status
        );
        assert!(
            reported.contains("`git` failed:"),
            "the report carries the message, not the structure: {reported}"
        );
        assert!(
            !reported.contains("Command {"),
            "the report is not the derived `Debug` form: {reported}"
        );
    }
}
