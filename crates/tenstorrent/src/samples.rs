//! Deterministic verify blocks for the differential: one per acceptance case
//! the Accept fragment must get right.
//!
//! Every block fills the valid logits with pseudo-random bfloat16 values in
//! `[-8, 8)`, plants each column's target at `+20`, and derives the drafts
//! from the case. Padding logits are set to `+64` in every block, above
//! every valid logit, so a device that fails to mask them answers wrongly on
//! every sample rather than by chance.

use core::num::NonZeroU32;

use infinitum_round::DraftWidth;
use infinitum_round::TokenId;

use crate::accept::Bf16;
use crate::accept::DraftBlock;
use crate::accept::ShapeMismatch;
use crate::accept::VerifyLogits;
use crate::accept::VocabularyLayout;

/// The acceptance case a sample exercises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleCase
{
    /// Every draft equals its target.
    FullAcceptance,
    /// The first draft differs from its target.
    FirstRejected,
    /// The drafts match up to the middle of the block, then differ.
    MidRejected,
    /// Every column's maximum is held by two ids, the lower in an earlier
    /// chunk or the same chunk; the drafts follow the lower ids.
    Tied,
}

impl SampleCase
{
    /// Every case, in the order the differential runs them.
    pub const ALL: [Self; 4] = [
        Self::FullAcceptance,
        Self::FirstRejected,
        Self::MidRejected,
        Self::Tied,
    ];
}

impl core::fmt::Display for SampleCase
{
    /// Render the case.
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
            | Self::FullAcceptance => "full acceptance",
            | Self::FirstRejected => "rejection at the first draft",
            | Self::MidRejected => "rejection mid-block",
            | Self::Tied => "ties in the argmax",
        });
    }
}

/// A sample's seed.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seed(u64);

impl From<u64> for Seed
{
    /// Wrap a seed.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(seed: u64) -> Self
    {
        return Self(seed);
    }
}

/// `SplitMix64`: a small, fixed generator, so a seed names the same block on
/// every host.
#[repr(transparent)]
#[derive(Debug, Clone)]
struct SplitMix(u64);

impl Iterator for SplitMix
{
    type Item = u64;

    /// The next value.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the `SplitMix64` step of the state.
    /// - provides: the sample stream.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    fn next(&mut self) -> Option<u64>
    {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ z.checked_shr(30).unwrap_or_default()).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ z.checked_shr(27).unwrap_or_default()).wrapping_mul(0x94D0_49BB_1331_11EB);
        return Some(z ^ z.checked_shr(31).unwrap_or_default());
    }
}

/// One verify block and its drafts.
#[derive(Debug, Clone)]
pub struct Sample
{
    /// The case it exercises.
    case: SampleCase,
    /// The logits.
    logits: VerifyLogits,
    /// The drafts.
    drafts: DraftBlock,
}

impl Sample
{
    /// Generate the block for `case` from `seed`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the logits follow `layout` for `width`; each column's
    ///   greatest valid logit is `+20`, at one id or, for [`SampleCase::Tied`],
    ///   at two; padding logits are `+64`; the drafts are the targets (the
    ///   lower tied id) up to the case's first mismatch, and a different valid
    ///   id there and after.
    /// - provides: the differential's inputs.
    /// - fails: when a count leaves its width.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`ShapeMismatch`]: the layout's counts do not fit the address width.
    #[inline]
    pub fn generate(
        case: SampleCase,
        layout: VocabularyLayout,
        width: DraftWidth,
        seed: Seed,
    ) -> Result<Self, ShapeMismatch>
    {
        let drafts_n = usize::try_from(NonZeroU32::from(width).get())
            .ok()
            .ok_or(ShapeMismatch::Width)?;
        let columns = drafts_n.checked_add(1).ok_or(ShapeMismatch::Width)?;
        let padded = usize::try_from(u32::from(layout.padded()))
            .ok()
            .ok_or(ShapeMismatch::Width)?;
        let valid = usize::try_from(u32::from(layout.size()))
            .ok()
            .ok_or(ShapeMismatch::Width)?;
        let chunk = usize::try_from(u32::from(layout.chunk()))
            .ok()
            .ok_or(ShapeMismatch::Width)?;
        let total = columns.checked_mul(padded).ok_or(ShapeMismatch::Width)?;
        let mut stream = SplitMix(seed.0);
        let mut draw = |bound: usize| -> usize {
            let value = stream.next().unwrap_or_default();
            let bound = u64::try_from(bound).unwrap_or(u64::MAX).max(1);
            return usize::try_from(value.checked_rem(bound).unwrap_or_default())
                .unwrap_or_default();
        };

        // Uniform in [-8, 8): sixteen steps per unit, exact in bfloat16.
        let mut values = Vec::with_capacity(total);
        for index in 0 .. total {
            let column_offset = index.checked_rem(padded).unwrap_or_default();
            let value = if column_offset >= valid {
                64.0_f32
            }
            else {
                let step = u16::try_from(draw(256)).unwrap_or_default();
                f32::from(step) / 16.0_f32 - 8.0_f32
            };
            values.push(Bf16::from(value));
        }

        let mut targets = Vec::with_capacity(columns);
        for column in 0 .. columns {
            let row_start = column.checked_mul(padded).ok_or(ShapeMismatch::Width)?;
            let target = draw(valid);
            if let Some(slot) = values.get_mut(row_start.saturating_add(target)) {
                *slot = Bf16::from(20.0_f32);
            }
            if case == SampleCase::Tied {
                // Alternate between a twin in the same chunk and one in a
                // later chunk, both above the target.
                let same_chunk_end = target
                    .checked_div(chunk)
                    .and_then(|c| return c.checked_add(1))
                    .and_then(|c| return c.checked_mul(chunk))
                    .unwrap_or(valid)
                    .min(valid);
                let twin = if column.checked_rem(2) == Some(0)
                    && same_chunk_end > target.saturating_add(1)
                {
                    target.saturating_add(1).saturating_add(draw(
                        same_chunk_end.saturating_sub(target).saturating_sub(1),
                    ))
                }
                else {
                    target
                        .saturating_add(1)
                        .saturating_add(draw(valid.saturating_sub(target).saturating_sub(1)))
                };
                if twin < valid
                    && let Some(slot) = values.get_mut(row_start.saturating_add(twin))
                {
                    *slot = Bf16::from(20.0_f32);
                }
            }
            targets.push(target);
        }

        let first_mismatch = match case {
            | SampleCase::FullAcceptance | SampleCase::Tied => drafts_n,
            | SampleCase::FirstRejected => 0,
            | SampleCase::MidRejected => drafts_n.checked_div(2).unwrap_or_default(),
        };
        let mut ids = Vec::with_capacity(drafts_n);
        for (column, &target) in targets.iter().take(drafts_n).enumerate() {
            let id = if column < first_mismatch {
                target
            }
            else {
                target
                    .saturating_add(1)
                    .checked_rem(valid)
                    .unwrap_or_default()
            };
            ids.push(TokenId::from(
                i32::try_from(id).ok().ok_or(ShapeMismatch::Width)?,
            ));
        }

        let logits = VerifyLogits::new(layout, width, values)?;
        let drafts = DraftBlock::new(width, ids)?;
        return Ok(Self {
            case,
            logits,
            drafts,
        });
    }

    /// The case.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn case(&self) -> SampleCase
    {
        return self.case;
    }

    /// The logits.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn logits(&self) -> &VerifyLogits
    {
        return &self.logits;
    }

    /// The drafts.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn drafts(&self) -> &DraftBlock
    {
        return &self.drafts;
    }
}

/// Tests that each case's block exercises its case under the reference.
#[cfg(test)]
mod tests
{
    use core::num::NonZeroU32;
    use core::str::FromStr as _;

    use infinitum_round::DraftWidth;
    use infinitum_round::TokenCount;

    use super::Sample;
    use super::SampleCase;
    use super::Seed;
    use crate::accept::ChunkWidth;
    use crate::accept::VocabularyLayout;
    use crate::accept::VocabularySize;
    use crate::accept::reference_accept;

    /// Each case yields the accepted length it names, on a small vocabulary.
    #[test]
    fn each_case_yields_its_accepted_length()
    {
        let layout = VocabularyLayout::new(
            VocabularySize::from(NonZeroU32::new(1000).unwrap()),
            ChunkWidth::try_from(NonZeroU32::new(64).unwrap()).unwrap(),
        )
        .unwrap();
        let width = DraftWidth::from_str("6").unwrap();
        let expected = [6, 0, 3, 6];
        for (case, accepted) in SampleCase::ALL.into_iter().zip(expected) {
            for seed in 0 .. 8 {
                let sample = Sample::generate(case, layout, width, Seed::from(seed)).unwrap();
                let answer = reference_accept(sample.logits(), sample.drafts()).unwrap();
                assert_eq!(
                    answer.accepted(),
                    TokenCount::from(accepted),
                    "{case} seed {seed}"
                );
            }
        }
    }
}
