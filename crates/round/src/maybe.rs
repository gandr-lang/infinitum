//! A value or the reason it is absent.
//!
//! The shape `docs/agents/rust.md` §Representation asks for where an absence
//! is not a failure: the empty arm carries a per-site reason, and nothing
//! propagates it on the error channel.

/// A present value, or the reason there is none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Maybe<Value, Reason>
{
    /// The value.
    Present(Value),
    /// No value, for this reason.
    Absent(Reason),
}
