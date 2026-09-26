//! Tenstorrent's planner: fragment by fragment, with one lowering.
//!
//! tt-mlir's TTIR-to-TTMetal path lowers the Accept fragment under sparse
//! rejection, and no other fragment yet. A round plans only when every
//! fragment lowers, so today every round is refused at its first fragment
//! other than Accept. [`Tenstorrent::plan_fragment`] lowers one fragment on
//! its own, which is how a round's acceptance reaches the device ahead of the
//! rest of the round.

use core::num::NonZeroU32;

use infinitum_round::AcceptRule;
use infinitum_round::Backend;
use infinitum_round::BackendName;
use infinitum_round::DraftWidth;
use infinitum_round::DrafterModel;
use infinitum_round::Fragment;
use infinitum_round::FragmentId;
use infinitum_round::FragmentKind;
use infinitum_round::Refusal;
use infinitum_round::RefusalReason;
use infinitum_round::RoundGraph;

/// The draft widths whose Accept program the pinned pipeline lowers. At
/// K = 4, 8, 9, 10, and 14, D2M's reblocking asserts and aborts the process
/// rather than reporting a diagnostic, so a width outside this list is refused
/// before the pipeline runs.
const LOWERED_WIDTHS: [u32; 10] = [1, 2, 3, 5, 6, 7, 11, 12, 13, 15];

/// The Tenstorrent backend, as a planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tenstorrent;

/// A lowered Accept fragment: where it sits in its round and the draft width
/// its program is lowered for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptPlan
{
    /// The fragment.
    at: FragmentId,
    /// The round's draft width.
    width: DraftWidth,
}

impl AcceptPlan
{
    /// The fragment.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn at(&self) -> FragmentId
    {
        return self.at;
    }

    /// The draft width the program is lowered for.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> DraftWidth
    {
        return self.width;
    }
}

/// How one fragment runs on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lowering
{
    /// The Accept program, under sparse rejection at temperature zero.
    Accept(AcceptPlan),
}

/// A round Tenstorrent runs: one lowering per fragment, in fragment order.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundPlan
{
    /// The lowerings.
    lowerings: Vec<Lowering>,
}

impl RoundPlan
{
    /// The lowerings, in fragment order.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn lowerings(&self) -> &[Lowering]
    {
        return &self.lowerings;
    }
}

impl Tenstorrent
{
    /// Lower the fragment `at` of `graph`, on its own.
    ///
    /// # Specification
    /// - requires: `fragment` is the fragment `at` names in `graph`.
    /// - ensures: on success the lowering is the Accept program at the width of
    ///   the graph's first DFlash2 block forward, a width the pinned pipeline
    ///   lowers.
    /// - provides: the one gate between a fragment and a device program.
    /// - fails: for any fragment other than Accept, for a graph with no DFlash2
    ///   drafter, and for a width outside the lowered list.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`Refusal`]: carries [`RefusalReason::NoLowering`],
    ///   [`RefusalReason::NoKnownDrafter`], or
    ///   [`RefusalReason::UnsupportedWidth`], as described.
    ///
    /// # Adequacy
    /// - hypothesis: L3 — the canonical round's Accept lowers at K = 15; its
    ///   first fragment is refused; K = 4, where the pipeline aborts, is
    ///   refused before it runs.
    /// - witness: `tests::the_canonical_accept_lowers`
    /// - witness: `tests::the_canonical_round_is_refused_at_its_first_fragment`
    /// - witness: `tests::a_width_the_pipeline_aborts_on_is_refused`
    #[inline]
    pub fn plan_fragment(
        self,
        graph: &RoundGraph,
        at: FragmentId,
        fragment: &Fragment,
    ) -> Result<Lowering, Refusal>
    {
        let refuse = |reason| return Refusal::new(BackendName::Tenstorrent, reason);
        return match fragment.kind() {
            | FragmentKind::Accept {
                rule: AcceptRule::SparseRejection,
            } => {
                let width = graph
                    .fragments()
                    .iter()
                    .find_map(|candidate| {
                        return match candidate.kind() {
                            | FragmentKind::DraftBlockForward {
                                model: DrafterModel::DFlash2,
                                width,
                            } => Some(width),
                            | _ => None,
                        };
                    })
                    .ok_or_else(|| return refuse(RefusalReason::NoKnownDrafter))?;
                if !LOWERED_WIDTHS.contains(&NonZeroU32::from(width).get()) {
                    return Err(refuse(RefusalReason::UnsupportedWidth(width)));
                }
                Ok(Lowering::Accept(AcceptPlan { at, width }))
            },
            | FragmentKind::ContextCatchUp
            | FragmentKind::DraftBlockForward { .. }
            | FragmentKind::ProposalSelect { .. }
            | FragmentKind::VerifyInputs
            | FragmentKind::TargetForward { .. }
            | FragmentKind::CommitKv
            | FragmentKind::FoldRecurrent { .. }
            | FragmentKind::PublishFeatures => Err(refuse(RefusalReason::NoLowering(at))),
        };
    }
}

impl Backend for Tenstorrent
{
    type Plan = RoundPlan;

    /// Plan `graph` fragment by fragment.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the plan holds every fragment's lowering in order.
    /// - provides: the whole-round gate; it refuses every round until each
    ///   fragment kind has a lowering.
    /// - fails: at the first fragment [`Tenstorrent::plan_fragment`] refuses.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`Refusal`]: the first fragment's refusal.
    ///
    /// # Adequacy
    /// - hypothesis: L2 — the canonical round is refused at its first fragment.
    /// - witness: `tests::the_canonical_round_is_refused_at_its_first_fragment`
    #[inline]
    fn plan(
        &self,
        graph: &RoundGraph,
    ) -> Result<RoundPlan, Refusal>
    {
        let lowerings = graph
            .entries()
            .map(|(at, fragment)| return self.plan_fragment(graph, at, fragment))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(RoundPlan { lowerings });
    }
}

/// Tests for the planner's gate.
#[cfg(test)]
mod tests
{
    use core::str::FromStr as _;

    use infinitum_round::Backend as _;
    use infinitum_round::DraftWidth;
    use infinitum_round::FragmentKind;
    use infinitum_round::RefusalReason;
    use infinitum_round::RoundGraph;

    use super::Lowering;
    use super::Tenstorrent;

    /// The canonical DFlash2 round at `width`.
    ///
    /// # Specification
    /// trivial.
    fn round(width: DraftWidth) -> RoundGraph
    {
        return infinitum_round::dflash2(width).unwrap();
    }

    /// The canonical round's Accept fragment lowers at K = 15.
    #[test]
    fn the_canonical_accept_lowers()
    {
        let width = DraftWidth::from_str("15").unwrap();
        let graph = round(width);
        let (at, accept) = graph
            .entries()
            .find(|&(_, fragment)| return matches!(fragment.kind(), FragmentKind::Accept { .. }))
            .unwrap();
        let Ok(Lowering::Accept(plan)) = Tenstorrent.plan_fragment(&graph, at, accept)
        else {
            panic!("the Accept fragment lowers");
        };
        assert_eq!(plan.at(), at, "the plan names its fragment");
        assert_eq!(plan.width(), width, "at the round's width");
    }

    /// The whole round is refused at fragment 0, the drafter's catch-up.
    #[test]
    fn the_canonical_round_is_refused_at_its_first_fragment()
    {
        let graph = round(DraftWidth::from_str("15").unwrap());
        let (first, _) = graph.entries().next().unwrap();
        assert_eq!(
            Tenstorrent
                .plan(&graph)
                .map_err(|refusal| return refusal.reason()),
            Err(RefusalReason::NoLowering(first)),
            "no lowering for the catch-up"
        );
    }

    /// K = 4 aborts the pinned pipeline, so the planner refuses it first.
    #[test]
    fn a_width_the_pipeline_aborts_on_is_refused()
    {
        let width = DraftWidth::from_str("4").unwrap();
        let graph = round(width);
        let (at, accept) = graph
            .entries()
            .find(|&(_, fragment)| return matches!(fragment.kind(), FragmentKind::Accept { .. }))
            .unwrap();
        assert_eq!(
            Tenstorrent
                .plan_fragment(&graph, at, accept)
                .map_err(|refusal| return refusal.reason()),
            Err(RefusalReason::UnsupportedWidth(width)),
            "refused before the pipeline runs"
        );
    }
}
