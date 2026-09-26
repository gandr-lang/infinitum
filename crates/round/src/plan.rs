//! The planning contract: a backend turns a round graph into its own plan or
//! refuses it with the reason.
//!
//! Refusal is a result, not a fallback: a backend never runs part of a graph
//! it cannot plan on the host instead. The operator reads which backend
//! refused, and at which fragment, and chooses another graph.

use crate::fragment::DraftWidth;
use crate::graph::BuildFailure;
use crate::graph::FragmentId;
use crate::graph::RoundGraph;

/// The backends infinitum plans rounds for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendName
{
    /// ninfer on CUDA, reached through its C++ engine.
    Ninfer,
}

impl core::fmt::Display for BackendName
{
    /// Render the name as the operator spells it.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return f.write_str(match *self {
            | Self::Ninfer => "ninfer",
        });
    }
}

/// Why a backend refused a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalReason
{
    /// The graph has no drafter whose round the backend knows.
    NoKnownDrafter,
    /// The graph differs from the backend's known fusion at this fragment,
    /// and the backend has no fragment-by-fragment lowering.
    NoKnownFusion(FragmentId),
    /// The backend's fusion does not run at this draft width.
    UnsupportedWidth(DraftWidth),
    /// The backend's canonical graph could not be composed.
    Composition(BuildFailure),
}

/// A backend's refusal of a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refusal
{
    /// The backend that refused.
    backend: BackendName,
    /// Why.
    reason: RefusalReason,
}

impl Refusal
{
    /// `backend` refuses for `reason`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn new(
        backend: BackendName,
        reason: RefusalReason,
    ) -> Self
    {
        return Self { backend, reason };
    }

    /// Why.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn reason(&self) -> RefusalReason
    {
        return self.reason;
    }
}

impl core::fmt::Display for Refusal
{
    /// Render the refusal with the backend's name first.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        let backend = self.backend;
        return match self.reason {
            | RefusalReason::NoKnownDrafter => {
                write!(f, "{backend} refuses the round: it knows no drafter in it")
            },
            | RefusalReason::NoKnownFusion(at) => {
                write!(
                    f,
                    "{backend} refuses the round: it differs from its known fusion at {at}"
                )
            },
            | RefusalReason::UnsupportedWidth(width) => {
                write!(
                    f,
                    "{backend} refuses the round: draft width {width} is out of its range"
                )
            },
            | RefusalReason::Composition(ref failure) => {
                write!(f, "{backend} could not compose its own round: {failure}")
            },
        };
    }
}

impl core::error::Error for Refusal
{
}

/// A backend that plans round graphs.
pub trait Backend
{
    /// What a successful plan yields.
    type Plan;

    /// Plan `graph`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the plan runs `graph` exactly; the backend never
    ///   substitutes a different round.
    /// - provides: the one gate between a graph and a backend's programs.
    /// - fails: with a refusal naming the backend and the reason.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`Refusal`]: the backend cannot run the graph.
    fn plan(
        &self,
        graph: &RoundGraph,
    ) -> Result<Self::Plan, Refusal>;
}
