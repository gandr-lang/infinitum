//! Fragments: the typed nodes a round graph is built from.
//!
//! A fragment kind names what the node computes at the granularity of model
//! sub-blocks and round steps, not tensor primitives. Parameters that change
//! the math are part of the kind; the kind fixes the node's effect, and the
//! effect fixes its phase, so a node cannot claim a phase its effect forbids.

/// The number of tokens a drafter proposes per round, `K`.
///
/// The verify runs `K + 1` query columns: the drafts and the anchor.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DraftWidth(core::num::NonZeroU32);

impl From<core::num::NonZeroU32> for DraftWidth
{
    /// Wrap a positive width.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(width: core::num::NonZeroU32) -> Self
    {
        return Self(width);
    }
}

impl From<DraftWidth> for core::num::NonZeroU32
{
    /// Unwrap the width.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(width: DraftWidth) -> Self
    {
        return width.0;
    }
}

impl core::str::FromStr for DraftWidth
{
    type Err = core::num::ParseIntError;

    /// Parse a positive decimal width.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the width is the parsed value, at least one.
    /// - provides: the command-line spelling of a draft width.
    /// - fails: with the integer parser's error on zero, a negative value,
    ///   anything above `u32::MAX`, or a non-numeric string.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`core::num::ParseIntError`]: `text` is not a positive `u32`.
    ///
    /// # Adequacy
    /// - hypothesis: L3 at the zero boundary, the one decision the wrapper adds
    ///   to `u32` parsing.
    /// - witness: `tests::a_zero_width_is_refused`
    #[inline]
    fn from_str(text: &str) -> Result<Self, Self::Err>
    {
        return text.parse::<core::num::NonZeroU32>().map(Self);
    }
}

impl core::fmt::Display for DraftWidth
{
    /// Render the width as its decimal value.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        return core::fmt::Display::fmt(&self.0, f);
    }
}

/// The drafter a block forward runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrafterModel
{
    /// DFlash2: a block-diffusion drafter conditioned on target features.
    DFlash2,
}

/// How a proposal is selected from the drafter's hidden states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Selector
{
    /// The top sixteen candidates per position, then DFlash2's conditional
    /// selector.
    Top16Conditional,
}

/// The attention mask a target forward verifies under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VerifyMask
{
    /// A causal block over the anchor and the drafts in order.
    CausalBlock,
}

/// The rule that turns a verification into licensed tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceptRule
{
    /// Rejection sampling over the sparse candidate set; greedy when the
    /// temperature is zero.
    SparseRejection,
}

/// How recurrent state is folded forward over the accepted prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecurrentFold
{
    /// Replay the recorded gated-delta-net transitions of the accepted
    /// prefix from the committed state, with the verify's own arithmetic.
    GdnReplay,
}

/// What a fragment computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FragmentKind
{
    /// Bring the drafter's context up to the target's committed prefix.
    ContextCatchUp,
    /// Run the drafter over one block of `width` positions.
    DraftBlockForward
    {
        /// The drafter.
        model: DrafterModel,
        /// The block's draft width.
        width: DraftWidth,
    },
    /// Select the proposal and its candidate distribution.
    ProposalSelect
    {
        /// The selection rule.
        selector: Selector,
    },
    /// Lay out the verify's ids and positions.
    VerifyInputs,
    /// Run the target over the anchor and the drafts.
    TargetForward
    {
        /// The verify's mask.
        mask: VerifyMask,
    },
    /// Decide the licensed tokens and the accepted length.
    Accept
    {
        /// The acceptance rule.
        rule: AcceptRule,
    },
    /// Advance the committed KV frontier by the accepted length.
    CommitKv,
    /// Fold recurrent state over the accepted prefix.
    FoldRecurrent
    {
        /// The fold.
        fold: RecurrentFold,
    },
    /// Publish the accepted positions' target features to the next round.
    PublishFeatures,
}

/// What a fragment may change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Effect
{
    /// Round-scoped buffers only.
    Scratch,
    /// State that stays invisible until a commit.
    Provisional,
    /// Committed state, at positions the target has already committed.
    CommittedPrefixMaintenance,
    /// Committed state, advanced by the accepted length.
    Commit,
}

/// The side of a round's one host boundary a fragment runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase
{
    /// Before the host's decision: committed state is not advanced.
    Execute,
    /// After it: the accepted length is the only decision input.
    Commit,
}

impl FragmentKind
{
    /// The kind's effect.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the three commit kinds are [`Effect::Commit`], the target
    ///   forward [`Effect::Provisional`], the catch-up
    ///   [`Effect::CommittedPrefixMaintenance`], and every other kind
    ///   [`Effect::Scratch`].
    /// - provides: the effect class the phase rule and a planner read.
    /// - fails: never.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 exhaustive over the kinds.
    /// - witness: `tests::each_kind_has_its_effect_and_phase`
    #[inline]
    #[must_use]
    pub const fn effect(self) -> Effect
    {
        return match self {
            | Self::ContextCatchUp => Effect::CommittedPrefixMaintenance,
            | Self::DraftBlockForward { .. }
            | Self::ProposalSelect { .. }
            | Self::VerifyInputs
            | Self::Accept { .. } => Effect::Scratch,
            | Self::TargetForward { .. } => Effect::Provisional,
            | Self::CommitKv | Self::FoldRecurrent { .. } | Self::PublishFeatures => Effect::Commit,
        };
    }

    /// The phase the kind's effect places it in.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: [`Phase::Commit`] exactly for [`Effect::Commit`].
    /// - provides: the phase the builder orders fragments by.
    /// - fails: never.
    /// - panics: none.
    ///
    /// # Adequacy
    /// - hypothesis: L3 exhaustive over the kinds.
    /// - witness: `tests::each_kind_has_its_effect_and_phase`
    #[inline]
    #[must_use]
    pub const fn phase(self) -> Phase
    {
        return match self.effect() {
            | Effect::Commit => Phase::Commit,
            | Effect::Scratch | Effect::Provisional | Effect::CommittedPrefixMaintenance => {
                Phase::Execute
            },
        };
    }
}

impl core::fmt::Display for FragmentKind
{
    /// Render the kind by its name, without parameters.
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
            | Self::ContextCatchUp => "context catch-up",
            | Self::DraftBlockForward { .. } => "draft block forward",
            | Self::ProposalSelect { .. } => "proposal select",
            | Self::VerifyInputs => "verify inputs",
            | Self::TargetForward { .. } => "target forward",
            | Self::Accept { .. } => "accept",
            | Self::CommitKv => "KV commit",
            | Self::FoldRecurrent { .. } => "recurrent fold",
            | Self::PublishFeatures => "feature publication",
        });
    }
}

impl core::fmt::Display for Phase
{
    /// Render the phase by its name.
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
            | Self::Execute => "execute",
            | Self::Commit => "commit",
        });
    }
}

/// Tests for the kinds' effects and the width's domain.
#[cfg(test)]
mod tests
{
    use core::num::NonZeroU32;
    use core::str::FromStr as _;

    use super::AcceptRule;
    use super::DraftWidth;
    use super::DrafterModel;
    use super::Effect;
    use super::FragmentKind;
    use super::Phase;
    use super::RecurrentFold;
    use super::Selector;
    use super::VerifyMask;

    /// Every kind carries the effect and phase the round's transaction rules
    /// give it.
    #[test]
    fn each_kind_has_its_effect_and_phase()
    {
        let width = DraftWidth::from(NonZeroU32::MIN);
        let expected = [
            (
                FragmentKind::ContextCatchUp,
                Effect::CommittedPrefixMaintenance,
                Phase::Execute,
            ),
            (
                FragmentKind::DraftBlockForward {
                    model: DrafterModel::DFlash2,
                    width,
                },
                Effect::Scratch,
                Phase::Execute,
            ),
            (
                FragmentKind::ProposalSelect {
                    selector: Selector::Top16Conditional,
                },
                Effect::Scratch,
                Phase::Execute,
            ),
            (FragmentKind::VerifyInputs, Effect::Scratch, Phase::Execute),
            (
                FragmentKind::TargetForward {
                    mask: VerifyMask::CausalBlock,
                },
                Effect::Provisional,
                Phase::Execute,
            ),
            (
                FragmentKind::Accept {
                    rule: AcceptRule::SparseRejection,
                },
                Effect::Scratch,
                Phase::Execute,
            ),
            (FragmentKind::CommitKv, Effect::Commit, Phase::Commit),
            (
                FragmentKind::FoldRecurrent {
                    fold: RecurrentFold::GdnReplay,
                },
                Effect::Commit,
                Phase::Commit,
            ),
            (FragmentKind::PublishFeatures, Effect::Commit, Phase::Commit),
        ];
        for (kind, effect, phase) in expected {
            assert_eq!(kind.effect(), effect, "{kind:?} has effect {effect:?}");
            assert_eq!(kind.phase(), phase, "{kind:?} runs in {phase:?}");
        }
    }

    /// Zero is not a width; one is.
    #[test]
    fn a_zero_width_is_refused()
    {
        assert!(
            DraftWidth::from_str("0").is_err(),
            "zero drafts is not a speculative round"
        );
        assert_eq!(
            DraftWidth::from_str("1"),
            Ok(DraftWidth::from(NonZeroU32::MIN)),
            "one draft is the smallest width"
        );
    }
}
