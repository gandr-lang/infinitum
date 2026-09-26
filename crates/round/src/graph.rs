//! The round graph and the builder that keeps it well formed.
//!
//! Fragments live in one arena in insertion order, and an edge names an
//! earlier fragment by its position, so every graph the builder returns is
//! acyclic by construction and two graphs built by the same steps are equal.
//! Edges carry data dependence only; typed ports (dtype, shape envelope,
//! sharding) arrive with the decomposed lowering that needs them.

use crate::fragment::FragmentKind;
use crate::fragment::Phase;
use crate::maybe::Maybe;

/// A fragment's position in its graph.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FragmentId(usize);

impl core::fmt::Display for FragmentId
{
    /// Render the position as `fragment #<n>`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return write!(f, "fragment #{}", self.0);
    }
}

/// One node: what it computes and the earlier fragments it reads.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fragment
{
    /// What the fragment computes.
    kind: FragmentKind,
    /// The fragments whose outputs it reads, each earlier than it, without
    /// repeats, in the order the component named them.
    inputs: Vec<FragmentId>,
}

impl Fragment
{
    /// What the fragment computes.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> FragmentKind
    {
        return self.kind;
    }

    /// The fragments it reads.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn inputs(&self) -> &[FragmentId]
    {
        return &self.inputs;
    }
}

/// A well-formed round: execute fragments, then commit fragments, each
/// commit fragment reading an acceptance.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RoundGraph
{
    /// The fragments in insertion order.
    fragments: Vec<Fragment>,
}

/// Why two graphs have no first difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Congruence
{
    /// The graphs are equal fragment for fragment.
    Identical,
}

impl RoundGraph
{
    /// The fragments in insertion order.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn fragments(&self) -> &[Fragment]
    {
        return &self.fragments;
    }

    /// The fragments with their ids, in insertion order.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the `n`th item is the `n`th fragment with the id naming it.
    /// - provides: the id a fragment-by-fragment planner names when it refuses.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    pub fn entries(&self) -> impl Iterator<Item = (FragmentId, &Fragment)>
    {
        return self
            .fragments
            .iter()
            .enumerate()
            .map(|(position, fragment)| return (FragmentId(position), fragment));
    }

    /// The first position at which `self` and `other` differ.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: [`Maybe::Present`] with the smallest position whose fragments
    ///   differ in kind or inputs, or, where one graph is a proper prefix of
    ///   the other, the shorter graph's length; [`Maybe::Absent`] exactly when
    ///   the graphs are equal.
    /// - provides: the fragment a planner names when a graph is not the one its
    ///   known fusion implements.
    /// - fails: never; a difference is not a failure.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 on the three shapes — equal, differing inside, one a
    ///   prefix of the other.
    /// - witness: `tests::equal_graphs_have_no_divergence`
    /// - witness: `tests::a_changed_fragment_is_the_divergence`
    /// - witness: `tests::a_prefix_diverges_at_its_end`
    #[inline]
    #[must_use]
    pub fn divergence(
        &self,
        other: &Self,
    ) -> Maybe<FragmentId, Congruence>
    {
        let mut position = 0_usize;
        let mut ours = self.fragments.iter();
        let mut theirs = other.fragments.iter();
        loop {
            match (ours.next(), theirs.next()) {
                | (None, None) => return Maybe::Absent(Congruence::Identical),
                | (Some(left), Some(right)) if left == right => {},
                | (Some(_), _) | (None, Some(_)) => return Maybe::Present(FragmentId(position)),
            }
            // The position counts fragments of a `Vec`, which cannot exceed
            // `usize::MAX` elements.
            position = position.saturating_add(1);
        }
    }
}

/// A graph the builder refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildFailure
{
    /// An input names no fragment added so far.
    UnknownInput(FragmentId),
    /// A fragment names the same input twice.
    RepeatedInput(FragmentId),
    /// An execute-phase fragment was added after the commit phase began.
    ExecuteAfterCommit(FragmentKind),
    /// A commit-phase fragment reads no acceptance, so its decision input is
    /// not the accepted length.
    CommitWithoutAcceptance(FragmentKind),
    /// The graph has no fragment in this phase.
    EmptyPhase(Phase),
}

impl core::fmt::Display for BuildFailure
{
    /// Render the refusal.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return match *self {
            | Self::UnknownInput(input) => write!(f, "{input} does not exist yet"),
            | Self::RepeatedInput(input) => write!(f, "{input} is named twice as an input"),
            | Self::ExecuteAfterCommit(kind) => {
                write!(
                    f,
                    "a {kind} runs before the commit and cannot follow a commit fragment"
                )
            },
            | Self::CommitWithoutAcceptance(kind) => {
                write!(f, "a {kind} commits without reading an acceptance")
            },
            | Self::EmptyPhase(phase) => write!(f, "the round has no {phase} fragment"),
        };
    }
}

impl core::error::Error for BuildFailure
{
}

/// Builds a [`RoundGraph`], refusing each step that would break its rules.
#[derive(Debug, Clone, Default)]
pub struct RoundBuilder
{
    /// The fragments added so far.
    fragments: Vec<Fragment>,
    /// Whether a commit fragment has been added.
    committing: Committing,
}

/// Whether the builder has entered the commit phase.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Committing
{
    /// Only execute fragments so far.
    #[default]
    NotYet,
    /// A commit fragment has been added.
    Begun,
}

impl RoundBuilder
{
    /// An empty builder.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn new() -> Self
    {
        return Self::default();
    }

    /// Add a fragment reading `inputs`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the fragment is the graph's last, and the returned
    ///   id names it; on failure the builder is unchanged.
    /// - provides: the one way a component places a fragment.
    /// - fails: when an input is not an earlier fragment or is repeated, when
    ///   an execute fragment follows a commit fragment, or when a commit
    ///   fragment reads no [`FragmentKind::Accept`].
    /// - panics: none.
    ///
    /// # Errors
    /// - [`BuildFailure::UnknownInput`], [`BuildFailure::RepeatedInput`],
    ///   [`BuildFailure::ExecuteAfterCommit`],
    ///   [`BuildFailure::CommitWithoutAcceptance`]: as named.
    ///
    /// # Adequacy
    /// - hypothesis: L3, one refusal per rule, and the canonical DFlash2
    ///   composition as the accepted case.
    /// - witness: `tests::an_input_from_the_future_is_refused`
    /// - witness: `tests::a_repeated_input_is_refused`
    /// - witness: `tests::execute_after_commit_is_refused`
    /// - witness: `tests::a_commit_without_acceptance_is_refused`
    /// - witness: `crate::component::tests::dflash2_composes_the_canonical_round`
    #[inline]
    pub fn add(
        &mut self,
        kind: FragmentKind,
        inputs: &[FragmentId],
    ) -> Result<FragmentId, BuildFailure>
    {
        let mut accepted = Accepted::No;
        for (index, &input) in inputs.iter().enumerate() {
            let Some(fragment) = self.fragments.get(input.0)
            else {
                return Err(BuildFailure::UnknownInput(input));
            };
            if inputs
                .get(.. index)
                .is_some_and(|earlier| earlier.contains(&input))
            {
                return Err(BuildFailure::RepeatedInput(input));
            }
            if matches!(fragment.kind, FragmentKind::Accept { .. }) {
                accepted = Accepted::Yes;
            }
        }
        match (kind.phase(), self.committing, accepted) {
            | (Phase::Execute, Committing::Begun, _) => {
                return Err(BuildFailure::ExecuteAfterCommit(kind));
            },
            | (Phase::Commit, _, Accepted::No) => {
                return Err(BuildFailure::CommitWithoutAcceptance(kind));
            },
            | (Phase::Commit, _, Accepted::Yes) => self.committing = Committing::Begun,
            | (Phase::Execute, Committing::NotYet, _) => {},
        }
        let id = FragmentId(self.fragments.len());
        self.fragments.push(Fragment {
            kind,
            inputs: inputs.to_vec(),
        });
        return Ok(id);
    }

    /// Finish the graph.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the graph holds every fragment added, in order.
    /// - provides: the round a planner receives.
    /// - fails: when either phase is empty.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`BuildFailure::EmptyPhase`]: no execute fragment, or no commit
    ///   fragment.
    ///
    /// # Adequacy
    /// - hypothesis: L3 on both empty phases.
    /// - witness: `tests::an_empty_phase_is_refused`
    #[inline]
    pub fn finish(self) -> Result<RoundGraph, BuildFailure>
    {
        if !self
            .fragments
            .iter()
            .any(|fragment| fragment.kind.phase() == Phase::Execute)
        {
            return Err(BuildFailure::EmptyPhase(Phase::Execute));
        }
        if self.committing == Committing::NotYet {
            return Err(BuildFailure::EmptyPhase(Phase::Commit));
        }
        return Ok(RoundGraph {
            fragments: self.fragments,
        });
    }
}

/// Whether a fragment's inputs include an acceptance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Accepted
{
    /// None of them is an acceptance.
    No,
    /// At least one is.
    Yes,
}

/// Tests for the builder's rules and graph comparison.
#[cfg(test)]
mod tests
{
    use super::BuildFailure;
    use super::Congruence;
    use super::FragmentId;
    use super::RoundBuilder;
    use crate::fragment::AcceptRule;
    use crate::fragment::FragmentKind;
    use crate::fragment::Phase;
    use crate::fragment::VerifyMask;
    use crate::maybe::Maybe;

    /// The smallest acceptance a commit may read.
    const ACCEPT: FragmentKind = FragmentKind::Accept {
        rule: AcceptRule::SparseRejection,
    };

    /// The target forward.
    const TARGET: FragmentKind = FragmentKind::TargetForward {
        mask: VerifyMask::CausalBlock,
    };

    /// A builder must not accept an input it has not produced.
    #[test]
    fn an_input_from_the_future_is_refused()
    {
        let mut builder = RoundBuilder::new();
        assert_eq!(
            builder.add(ACCEPT, &[FragmentId(0)]),
            Err(BuildFailure::UnknownInput(FragmentId(0))),
            "an empty builder has no fragment #0"
        );
        assert!(
            builder.finish().is_err(),
            "the refused fragment was not added"
        );
    }

    /// An input named twice is refused.
    #[test]
    fn a_repeated_input_is_refused()
    {
        let mut builder = RoundBuilder::new();
        let Ok(target) = builder.add(TARGET, &[])
        else {
            panic!("a target forward with no inputs is admitted");
        };
        assert_eq!(
            builder.add(ACCEPT, &[target, target]),
            Err(BuildFailure::RepeatedInput(target)),
            "the second mention is the refusal"
        );
    }

    /// Once the commit phase begins, execute fragments are refused.
    #[test]
    fn execute_after_commit_is_refused()
    {
        let mut builder = RoundBuilder::new();
        let Ok(accept) = builder.add(ACCEPT, &[])
        else {
            panic!("an acceptance with no inputs is admitted");
        };
        assert!(
            builder.add(FragmentKind::CommitKv, &[accept]).is_ok(),
            "the commit reads it"
        );
        assert_eq!(
            builder.add(TARGET, &[]),
            Err(BuildFailure::ExecuteAfterCommit(TARGET)),
            "a target forward cannot follow a commit"
        );
    }

    /// A commit must read an acceptance directly.
    #[test]
    fn a_commit_without_acceptance_is_refused()
    {
        let mut builder = RoundBuilder::new();
        let Ok(target) = builder.add(TARGET, &[])
        else {
            panic!("a target forward with no inputs is admitted");
        };
        assert_eq!(
            builder.add(FragmentKind::CommitKv, &[target]),
            Err(BuildFailure::CommitWithoutAcceptance(
                FragmentKind::CommitKv
            )),
            "the target's output is not an accepted length"
        );
    }

    /// Both phases must be present.
    #[test]
    fn an_empty_phase_is_refused()
    {
        assert_eq!(
            RoundBuilder::new().finish(),
            Err(BuildFailure::EmptyPhase(Phase::Execute)),
            "an empty graph has no execute phase"
        );
        let mut builder = RoundBuilder::new();
        assert!(
            builder.add(ACCEPT, &[]).is_ok(),
            "an acceptance alone is admitted"
        );
        assert_eq!(
            builder.finish(),
            Err(BuildFailure::EmptyPhase(Phase::Commit)),
            "a round that never commits is refused"
        );
    }

    /// Build accept then commit, optionally with a trailing publish.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the graph is an acceptance, a KV commit reading it, and, when
    ///   `publish` is `Publish::Yes`, a feature publish reading it.
    /// - provides: the minimal well-formed round the comparison tests use.
    /// - fails: never.
    /// - panics: when the builder refuses one of these steps, which fails the
    ///   calling test.
    fn round(publish: Publish) -> super::RoundGraph
    {
        let mut builder = RoundBuilder::new();
        let Ok(accept) = builder.add(ACCEPT, &[])
        else {
            panic!("an acceptance with no inputs is admitted");
        };
        assert!(
            builder.add(FragmentKind::CommitKv, &[accept]).is_ok(),
            "the commit reads it"
        );
        if publish == Publish::Yes {
            assert!(
                builder
                    .add(FragmentKind::PublishFeatures, &[accept])
                    .is_ok(),
                "the publish reads it"
            );
        }
        let Ok(graph) = builder.finish()
        else {
            panic!("both phases are present");
        };
        return graph;
    }

    /// Whether [`round`] appends a publish.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Publish
    {
        /// It stops at the commit.
        No,
        /// It appends a publish.
        Yes,
    }

    /// Two graphs built by the same steps have no divergence.
    #[test]
    fn equal_graphs_have_no_divergence()
    {
        assert_eq!(
            round(Publish::Yes).divergence(&round(Publish::Yes)),
            Maybe::Absent(Congruence::Identical),
            "the same steps build the same graph"
        );
    }

    /// A differing fragment is named by its position.
    #[test]
    fn a_changed_fragment_is_the_divergence()
    {
        let mut builder = RoundBuilder::new();
        let Ok(accept) = builder.add(ACCEPT, &[])
        else {
            panic!("an acceptance with no inputs is admitted");
        };
        assert!(
            builder
                .add(FragmentKind::PublishFeatures, &[accept])
                .is_ok(),
            "reads it"
        );
        let Ok(other) = builder.finish()
        else {
            panic!("both phases are present");
        };
        assert_eq!(
            round(Publish::No).divergence(&other),
            Maybe::Present(FragmentId(1)),
            "the second fragment differs"
        );
    }

    /// Where one graph extends the other, the divergence is the shorter one's
    /// end.
    #[test]
    fn a_prefix_diverges_at_its_end()
    {
        assert_eq!(
            round(Publish::No).divergence(&round(Publish::Yes)),
            Maybe::Present(FragmentId(2)),
            "the longer graph's extra fragment is the difference"
        );
        assert_eq!(
            round(Publish::Yes).divergence(&round(Publish::No)),
            Maybe::Present(FragmentId(2)),
            "the relation is symmetric"
        );
    }
}
