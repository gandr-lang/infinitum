//! Components: the parts of a speculation method, each emitting its
//! fragments into a round.
//!
//! Components describe; they never execute. A drafter proposes, a verifier
//! scores the proposal on the target, an acceptor turns the score into
//! licensed tokens, and a committer advances committed state by the accepted
//! length. A method is one choice of the four, and [`compose`] is the one
//! order they are emitted in.

use crate::fragment::AcceptRule;
use crate::fragment::DraftWidth;
use crate::fragment::DrafterModel;
use crate::fragment::FragmentKind;
use crate::fragment::RecurrentFold;
use crate::fragment::Selector;
use crate::fragment::VerifyMask;
use crate::graph::BuildFailure;
use crate::graph::FragmentId;
use crate::graph::RoundBuilder;
use crate::graph::RoundGraph;

/// A drafter's output: the fragment holding the proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Proposal
{
    /// The fragment that selected the proposal.
    selection: FragmentId,
}

/// A verifier's output: the fragment holding the target's scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verification
{
    /// The target forward.
    target: FragmentId,
}

/// An acceptor's output: the fragment holding the licensed tokens and the
/// accepted length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Licensed
{
    /// The acceptance.
    acceptance: FragmentId,
}

/// Proposes draft tokens.
pub trait Drafter
{
    /// Emit the drafting fragments.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the proposal names an execute fragment this call
    ///   added.
    /// - provides: the proposal a verifier and an acceptor read.
    /// - fails: with the builder's refusal.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`BuildFailure`]: the builder refused a fragment.
    fn emit(
        &self,
        round: &mut RoundBuilder,
    ) -> Result<Proposal, BuildFailure>;
}

/// Scores a proposal on the target model.
pub trait Verifier
{
    /// Emit the verification fragments.
    ///
    /// # Specification
    /// - requires: `proposal` came from this round.
    /// - ensures: on success the verification names an execute fragment this
    ///   call added.
    /// - provides: the scores an acceptor reads and the replay records a
    ///   committer folds.
    /// - fails: with the builder's refusal.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`BuildFailure`]: the builder refused a fragment.
    fn emit(
        &self,
        round: &mut RoundBuilder,
        proposal: Proposal,
    ) -> Result<Verification, BuildFailure>;
}

/// Turns a verification into licensed tokens.
pub trait Acceptor
{
    /// Emit the acceptance fragments.
    ///
    /// # Specification
    /// - requires: both arguments came from this round.
    /// - ensures: on success the output names a [`FragmentKind::Accept`] this
    ///   call added.
    /// - provides: the accepted length every commit fragment reads.
    /// - fails: with the builder's refusal.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`BuildFailure`]: the builder refused a fragment.
    fn emit(
        &self,
        round: &mut RoundBuilder,
        proposal: Proposal,
        verification: Verification,
    ) -> Result<Licensed, BuildFailure>;
}

/// Advances committed state by the accepted length.
pub trait Committer
{
    /// Emit the commit fragments.
    ///
    /// # Specification
    /// - requires: both arguments came from this round.
    /// - ensures: on success at least one commit fragment was added, each
    ///   reading the acceptance.
    /// - provides: the round's commit phase.
    /// - fails: with the builder's refusal.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`BuildFailure`]: the builder refused a fragment.
    fn emit(
        &self,
        round: &mut RoundBuilder,
        verification: Verification,
        licensed: Licensed,
    ) -> Result<(), BuildFailure>;
}

/// Compose a round from its four components.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success the graph holds the drafter's fragments, then the
///   verifier's, the acceptor's, and the committer's, each reading what the
///   earlier components returned.
/// - provides: the one composition order every method shares.
/// - fails: with the first component's refusal, or the builder's when a phase
///   is empty.
/// - panics: none.
///
/// # Errors
/// - [`BuildFailure`]: a component or the finished graph was refused.
///
/// # Adequacy
/// - hypothesis: L3 on the DFlash2 composition, compared fragment for fragment
///   with the round ninfer's maintainer reference describes.
/// - witness: `tests::dflash2_composes_the_canonical_round`
#[inline]
pub fn compose<Draft, Verify, Accept, Commit>(
    drafter: &Draft,
    verifier: &Verify,
    acceptor: &Accept,
    committer: &Commit,
) -> Result<RoundGraph, BuildFailure>
where
    Draft: Drafter,
    Verify: Verifier,
    Accept: Acceptor,
    Commit: Committer,
{
    let mut round = RoundBuilder::new();
    let proposal = drafter.emit(&mut round)?;
    let verification = verifier.emit(&mut round, proposal)?;
    let licensed = acceptor.emit(&mut round, proposal, verification)?;
    committer.emit(&mut round, verification, licensed)?;
    return round.finish();
}

/// The DFlash2 drafter at one draft width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2Drafter
{
    /// Tokens drafted per round.
    width: DraftWidth,
}

impl DFlash2Drafter
{
    /// The drafter at `width`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn new(width: DraftWidth) -> Self
    {
        return Self { width };
    }
}

impl Drafter for DFlash2Drafter
{
    /// Catch the drafter's context up, run one block forward, and select the
    /// proposal.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success three fragments were added, each reading the
    ///   previous, and the proposal names the selection.
    /// - provides: DFlash2's proposal.
    /// - fails: with the builder's refusal, which a builder still in its
    ///   execute phase does not give.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`BuildFailure`]: the builder refused a fragment.
    ///
    /// # Adequacy
    /// - hypothesis: L3 through the canonical composition.
    /// - witness: `tests::dflash2_composes_the_canonical_round`
    #[inline]
    fn emit(
        &self,
        round: &mut RoundBuilder,
    ) -> Result<Proposal, BuildFailure>
    {
        let catch_up = round.add(FragmentKind::ContextCatchUp, &[])?;
        let block = round.add(
            FragmentKind::DraftBlockForward {
                model: DrafterModel::DFlash2,
                width: self.width,
            },
            &[catch_up],
        )?;
        let selection = round.add(
            FragmentKind::ProposalSelect {
                selector: Selector::Top16Conditional,
            },
            &[block],
        )?;
        return Ok(Proposal { selection });
    }
}

/// Verification of a draft block under a causal mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalBlockVerifier;

impl Verifier for CausalBlockVerifier
{
    /// Lay out the verify inputs and run the target over them.
    ///
    /// # Specification
    /// - requires: `proposal` came from this round.
    /// - ensures: on success two fragments were added, the inputs reading the
    ///   proposal and the target forward reading the inputs.
    /// - provides: the verification of a block proposal.
    /// - fails: with the builder's refusal.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`BuildFailure`]: the builder refused a fragment.
    ///
    /// # Adequacy
    /// - hypothesis: L3 through the canonical composition.
    /// - witness: `tests::dflash2_composes_the_canonical_round`
    #[inline]
    fn emit(
        &self,
        round: &mut RoundBuilder,
        proposal: Proposal,
    ) -> Result<Verification, BuildFailure>
    {
        let inputs = round.add(FragmentKind::VerifyInputs, &[proposal.selection])?;
        let target = round.add(
            FragmentKind::TargetForward {
                mask: VerifyMask::CausalBlock,
            },
            &[inputs],
        )?;
        return Ok(Verification { target });
    }
}

/// Rejection over the sparse candidate set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseRejectionAcceptor;

impl Acceptor for SparseRejectionAcceptor
{
    /// Accept against the proposal's candidates and the target's scores.
    ///
    /// # Specification
    /// - requires: both arguments came from this round.
    /// - ensures: on success one acceptance was added, reading the proposal and
    ///   the target forward in that order.
    /// - provides: the licensed tokens and the accepted length.
    /// - fails: with the builder's refusal.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`BuildFailure`]: the builder refused a fragment.
    ///
    /// # Adequacy
    /// - hypothesis: L3 through the canonical composition.
    /// - witness: `tests::dflash2_composes_the_canonical_round`
    #[inline]
    fn emit(
        &self,
        round: &mut RoundBuilder,
        proposal: Proposal,
        verification: Verification,
    ) -> Result<Licensed, BuildFailure>
    {
        let acceptance = round.add(
            FragmentKind::Accept {
                rule: AcceptRule::SparseRejection,
            },
            &[proposal.selection, verification.target],
        )?;
        return Ok(Licensed { acceptance });
    }
}

/// The commit of a hybrid attention and gated-delta-net target: KV frontier,
/// recurrent replay, and feature publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnReplayCommitter;

impl Committer for GdnReplayCommitter
{
    /// Commit the KV frontier, fold the recurrent state, and publish the
    /// accepted features.
    ///
    /// # Specification
    /// - requires: both arguments came from this round.
    /// - ensures: on success three commit fragments were added; each reads the
    ///   acceptance, and the fold and the publication also read the target
    ///   forward whose records they consume.
    /// - provides: the commit phase.
    /// - fails: with the builder's refusal.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`BuildFailure`]: the builder refused a fragment.
    ///
    /// # Adequacy
    /// - hypothesis: L3 through the canonical composition.
    /// - witness: `tests::dflash2_composes_the_canonical_round`
    #[inline]
    fn emit(
        &self,
        round: &mut RoundBuilder,
        verification: Verification,
        licensed: Licensed,
    ) -> Result<(), BuildFailure>
    {
        round.add(FragmentKind::CommitKv, &[licensed.acceptance])?;
        round.add(
            FragmentKind::FoldRecurrent {
                fold: RecurrentFold::GdnReplay,
            },
            &[licensed.acceptance, verification.target],
        )?;
        round.add(FragmentKind::PublishFeatures, &[
            licensed.acceptance,
            verification.target,
        ])?;
        return Ok(());
    }
}

/// The DFlash2 round at `width`: its drafter, a causal block verify, sparse
/// rejection, and the replay commit.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success the graph is [`compose`] over the four DFlash2
///   components.
/// - provides: the graph a driver submits for DFlash2, and the canonical form a
///   backend's known fusion is matched against.
/// - fails: never in practice; the composition satisfies every builder rule,
///   and the result type carries the builder's refusal rather than asserting
///   that.
/// - panics: none.
///
/// # Errors
/// - [`BuildFailure`]: the builder refused the composition.
///
/// # Adequacy
/// - hypothesis: L3, fragment for fragment.
/// - witness: `tests::dflash2_composes_the_canonical_round`
#[inline]
pub fn dflash2(width: DraftWidth) -> Result<RoundGraph, BuildFailure>
{
    return compose(
        &DFlash2Drafter::new(width),
        &CausalBlockVerifier,
        &SparseRejectionAcceptor,
        &GdnReplayCommitter,
    );
}

/// Tests for the DFlash2 composition.
#[cfg(test)]
mod tests
{
    use core::num::NonZeroU32;

    use super::dflash2;
    use crate::fragment::AcceptRule;
    use crate::fragment::DraftWidth;
    use crate::fragment::DrafterModel;
    use crate::fragment::FragmentKind;
    use crate::fragment::RecurrentFold;
    use crate::fragment::Selector;
    use crate::fragment::VerifyMask;
    use crate::graph::RoundBuilder;

    /// The composition is the round of ninfer's maintainer reference
    /// (`docs/maintainer/dflash.md`, execution flow), with each fragment
    /// reading exactly what that round's step reads.
    #[test]
    fn dflash2_composes_the_canonical_round()
    {
        let width = DraftWidth::from(NonZeroU32::new(7).unwrap());
        let mut round = RoundBuilder::new();
        let catch_up = round.add(FragmentKind::ContextCatchUp, &[]).unwrap();
        let block = round
            .add(
                FragmentKind::DraftBlockForward {
                    model: DrafterModel::DFlash2,
                    width,
                },
                &[catch_up],
            )
            .unwrap();
        let selection = round
            .add(
                FragmentKind::ProposalSelect {
                    selector: Selector::Top16Conditional,
                },
                &[block],
            )
            .unwrap();
        let inputs = round.add(FragmentKind::VerifyInputs, &[selection]).unwrap();
        let target = round
            .add(
                FragmentKind::TargetForward {
                    mask: VerifyMask::CausalBlock,
                },
                &[inputs],
            )
            .unwrap();
        let acceptance = round
            .add(
                FragmentKind::Accept {
                    rule: AcceptRule::SparseRejection,
                },
                &[selection, target],
            )
            .unwrap();
        round.add(FragmentKind::CommitKv, &[acceptance]).unwrap();
        round
            .add(
                FragmentKind::FoldRecurrent {
                    fold: RecurrentFold::GdnReplay,
                },
                &[acceptance, target],
            )
            .unwrap();
        round
            .add(FragmentKind::PublishFeatures, &[acceptance, target])
            .unwrap();
        assert_eq!(
            dflash2(width),
            Ok(round.finish().unwrap()),
            "the DFlash2 round is the reference round"
        );
    }
}
