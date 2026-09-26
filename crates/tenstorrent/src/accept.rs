//! The Accept fragment's inputs, its answer, and the host reference that
//! computes the answer the device must agree with.
//!
//! Greedy acceptance at temperature zero: each verify column's target is the
//! argmax of its logits over the valid vocabulary, the lowest id winning a
//! tie; the accepted length is the first column whose draft differs from its
//! target; the licensed tokens are the targets up to and including that
//! column, which are the accepted drafts followed by the correction or bonus.

use core::num::NonZeroU32;

use infinitum_round::DraftWidth;
use infinitum_round::TokenCount;
use infinitum_round::TokenId;

/// The widest chunk the device reduces in one row: local indices are carried
/// in bfloat16, which holds every integer up to 256 exactly.
const MAX_CHUNK: u32 = 256;

/// The most chunks a vocabulary row splits into: a chunk index travels as
/// two bfloat16 digits, `c / 32` and `c % 32`, and the first must stay below
/// 256.
const MAX_CHUNKS: u32 = 8192;

/// The served vocabulary's size.
const SERVED_SIZE: NonZeroU32 = match NonZeroU32::new(248_077) {
    | Some(size) => size,
    | None => NonZeroU32::MIN,
};

/// The served layout's chunk width.
const SERVED_CHUNK: NonZeroU32 = match NonZeroU32::new(256) {
    | Some(width) => width,
    | None => NonZeroU32::MIN,
};

/// The served layout's chunk count: `ceil(248_077 / 256)`.
const SERVED_CHUNKS: NonZeroU32 = match NonZeroU32::new(970) {
    | Some(count) => count,
    | None => NonZeroU32::MIN,
};

/// A bfloat16 value, as its bit pattern.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16(u16);

impl From<u16> for Bf16
{
    /// Wrap a bit pattern.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(bits: u16) -> Self
    {
        return Self(bits);
    }
}

impl From<Bf16> for u16
{
    /// Unwrap the bit pattern.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(value: Bf16) -> Self
    {
        return value.0;
    }
}

impl From<Bf16> for f32
{
    /// Widen to float32, which represents every bfloat16 value exactly.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the result's bits are the bfloat16 bits followed by sixteen
    ///   zero bits.
    /// - provides: the value the reference compares.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    fn from(value: Bf16) -> Self
    {
        let bits = u32::from(value.0).checked_shl(16).unwrap_or_default();
        return Self::from_bits(bits);
    }
}

impl From<f32> for Bf16
{
    /// Narrow a float32 by truncation, which is exact for every value a
    /// bfloat16 holds.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the result is the upper sixteen bits of `value`.
    /// - provides: the samples' planted values.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    fn from(value: f32) -> Self
    {
        let upper = value.to_bits().checked_shr(16).unwrap_or_default();
        return Self(u16::try_from(upper).unwrap_or_default());
    }
}

/// How many ids the vocabulary holds; ids at or past it are padding.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocabularySize(NonZeroU32);

impl From<NonZeroU32> for VocabularySize
{
    /// Wrap a size.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(size: NonZeroU32) -> Self
    {
        return Self(size);
    }
}

impl From<VocabularySize> for u32
{
    /// Unwrap the size.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(size: VocabularySize) -> Self
    {
        return size.0.get();
    }
}

/// How many logits one device row reduces.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkWidth(NonZeroU32);

impl TryFrom<NonZeroU32> for ChunkWidth
{
    type Error = LayoutFailure;

    /// Wrap a width of at most 256.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the width is `width`.
    /// - provides: a width whose local indices bfloat16 holds exactly.
    /// - fails: with [`LayoutFailure::ChunkTooWide`] above 256.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`LayoutFailure::ChunkTooWide`]: `width` exceeds 256.
    #[inline]
    fn try_from(width: NonZeroU32) -> Result<Self, Self::Error>
    {
        if width.get() > MAX_CHUNK {
            return Err(LayoutFailure::ChunkTooWide);
        }
        return Ok(Self(width));
    }
}

impl From<ChunkWidth> for u32
{
    /// Unwrap the width.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(width: ChunkWidth) -> Self
    {
        return width.0.get();
    }
}

/// How many chunks one verify column splits into.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkCount(NonZeroU32);

impl From<ChunkCount> for u32
{
    /// Unwrap the count.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(count: ChunkCount) -> Self
    {
        return count.0.get();
    }
}

/// How many logits one verify column holds, padding included.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaddedWidth(NonZeroU32);

impl From<PaddedWidth> for u32
{
    /// Unwrap the width.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(width: PaddedWidth) -> Self
    {
        return width.0.get();
    }
}

/// bfloat16 bit patterns in the order a program reads them: the form every
/// buffer takes on its way to the device.
#[repr(transparent)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeviceBits(Vec<u16>);

impl DeviceBits
{
    /// Empty the buffer, keeping its allocation.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    pub fn clear(&mut self)
    {
        self.0.clear();
    }
}

impl From<Vec<u16>> for DeviceBits
{
    /// Wrap bit patterns already in program order.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(bits: Vec<u16>) -> Self
    {
        return Self(bits);
    }
}

impl Extend<u16> for DeviceBits
{
    /// Append bit patterns.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn extend<Bits>(
        &mut self,
        iter: Bits,
    ) where
        Bits: IntoIterator<Item = u16>,
    {
        self.0.extend(iter);
    }
}

impl AsRef<[u16]> for DeviceBits
{
    /// The bit patterns.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn as_ref(&self) -> &[u16]
    {
        return &self.0;
    }
}

/// Why a vocabulary layout is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutFailure
{
    /// The chunk is wider than 256 logits.
    ChunkTooWide,
    /// The vocabulary needs more than 8192 chunks.
    TooManyChunks,
}

impl core::fmt::Display for LayoutFailure
{
    /// Render the failure.
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
            | Self::ChunkTooWide => "a chunk is at most 256 logits wide",
            | Self::TooManyChunks => "the vocabulary needs more than 8192 chunks",
        });
    }
}

impl core::error::Error for LayoutFailure
{
}

/// A vocabulary and the chunking the device reduces it in: `chunks` rows of
/// `chunk` logits per verify column, the last chunk padded past the
/// vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocabularyLayout
{
    /// The valid ids.
    size: VocabularySize,
    /// The logits per device row.
    chunk: ChunkWidth,
    /// Chunks per verify column.
    chunks: NonZeroU32,
}

impl VocabularyLayout
{
    /// Chunk `size` by `chunk`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the layout has `ceil(size / chunk)` chunks.
    /// - provides: the geometry the device program is lowered for.
    /// - fails: with [`LayoutFailure::TooManyChunks`] past 8192 chunks.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`LayoutFailure::TooManyChunks`]: the vocabulary needs more than 8192
    ///   chunks.
    ///
    /// # Adequacy
    /// - hypothesis: L3 — the served vocabulary pads to 970 chunks of 256; a
    ///   vocabulary one id past 8192 chunks is refused.
    /// - witness: `tests::the_served_vocabulary_tiles_into_970_chunks`
    /// - witness: `tests::a_vocabulary_past_8192_chunks_is_refused`
    #[inline]
    pub fn new(
        size: VocabularySize,
        chunk: ChunkWidth,
    ) -> Result<Self, LayoutFailure>
    {
        let chunks = size.0.get().div_ceil(chunk.0.get());
        if chunks > MAX_CHUNKS {
            return Err(LayoutFailure::TooManyChunks);
        }
        let chunks = NonZeroU32::new(chunks).ok_or(LayoutFailure::TooManyChunks)?;
        return Ok(Self {
            size,
            chunk,
            chunks,
        });
    }

    /// The served model's vocabulary: 248,077 ids, padded to 248,320 logits
    /// in 970 chunks of 256.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn served() -> Self
    {
        return Self {
            size: VocabularySize(SERVED_SIZE),
            chunk: ChunkWidth(SERVED_CHUNK),
            chunks: SERVED_CHUNKS,
        };
    }

    /// The valid ids.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> VocabularySize
    {
        return self.size;
    }

    /// The logits per device row.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn chunk(&self) -> ChunkWidth
    {
        return self.chunk;
    }

    /// Chunks per verify column.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn chunks(&self) -> ChunkCount
    {
        return ChunkCount(self.chunks);
    }

    /// Logits per verify column, padding included.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the result is `chunks * chunk`, which `new` bounded.
    /// - provides: the row stride of the logits block.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    #[must_use]
    pub fn padded(&self) -> PaddedWidth
    {
        return PaddedWidth(self.chunks.saturating_mul(self.chunk.0));
    }
}

/// Why a verify block's inputs are refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeMismatch
{
    /// The logits do not hold `(K + 1) * padded` values.
    Logits,
    /// The drafts do not hold `K` ids.
    Drafts,
    /// A count does not fit the platform's address width.
    Width,
}

impl core::fmt::Display for ShapeMismatch
{
    /// Render the mismatch.
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
            | Self::Logits => "the logits are not (K + 1) padded vocabulary rows",
            | Self::Drafts => "the drafts are not K ids",
            | Self::Width => "a count exceeds the address width",
        });
    }
}

impl core::error::Error for ShapeMismatch
{
}

/// The target logits over the anchor and `K` drafts: `K + 1` rows of the
/// padded vocabulary, row major. Padding logits hold any value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyLogits
{
    /// The vocabulary layout the rows follow.
    layout: VocabularyLayout,
    /// The draft width `K`.
    width: DraftWidth,
    /// `(K + 1) * padded` bfloat16 bit patterns, the form the device reads.
    bits: DeviceBits,
}

impl VerifyLogits
{
    /// Wrap `values` as `K + 1` rows of `layout`.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the rows are `values` in order.
    /// - provides: the Accept fragment's logits input.
    /// - fails: with [`ShapeMismatch::Logits`] when the count is not `(K + 1) *
    ///   padded`.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`ShapeMismatch::Logits`]: the count is wrong.
    /// - [`ShapeMismatch::Width`]: the expected count overflows `usize`.
    #[inline]
    pub fn new(
        layout: VocabularyLayout,
        width: DraftWidth,
        values: Vec<Bf16>,
    ) -> Result<Self, ShapeMismatch>
    {
        let expected = usize::from(column_count(width)?)
            .checked_mul(
                usize::try_from(u32::from(layout.padded()))
                    .ok()
                    .ok_or(ShapeMismatch::Width)?,
            )
            .ok_or(ShapeMismatch::Width)?;
        if values.len() != expected {
            return Err(ShapeMismatch::Logits);
        }
        let bits = DeviceBits(values.into_iter().map(u16::from).collect());
        return Ok(Self {
            layout,
            width,
            bits,
        });
    }

    /// The layout.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn layout(&self) -> VocabularyLayout
    {
        return self.layout;
    }

    /// The draft width.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> DraftWidth
    {
        return self.width;
    }

    /// The bfloat16 bit patterns, row major: the bytes the device reads, the
    /// `[K + 1, C * W]` block it views as `[K + 1, C, W]`.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn bits(&self) -> &DeviceBits
    {
        return &self.bits;
    }
}

/// `K + 1`, the verify block's column count, as an index count.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ColumnCount(usize);

impl From<ColumnCount> for usize
{
    /// Unwrap the count.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(count: ColumnCount) -> Self
    {
        return count.0;
    }
}

/// `K + 1`, as an index count.
///
/// # Specification
/// - requires: nothing.
/// - ensures: the result is the width plus one.
/// - provides: the verify block's column count.
/// - fails: with [`ShapeMismatch::Width`] when it does not fit `usize`.
/// - panics: none.
///
/// # Errors
/// - [`ShapeMismatch::Width`]: `K + 1` exceeds `usize`.
fn column_count(width: DraftWidth) -> Result<ColumnCount, ShapeMismatch>
{
    let drafts = usize::try_from(NonZeroU32::from(width).get())
        .ok()
        .ok_or(ShapeMismatch::Width)?;
    return drafts
        .checked_add(1)
        .map(ColumnCount)
        .ok_or(ShapeMismatch::Width);
}

/// The drafted ids, one per draft column.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftBlock
{
    /// `K` ids.
    ids: Vec<TokenId>,
}

impl DraftBlock
{
    /// Wrap `ids` as a block of `width` drafts.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the block is `ids`.
    /// - provides: the Accept fragment's drafts input.
    /// - fails: with [`ShapeMismatch::Drafts`] when the count is not `K`.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`ShapeMismatch::Drafts`]: the count is not `K`.
    /// - [`ShapeMismatch::Width`]: `K` does not fit `usize`.
    #[inline]
    pub fn new(
        width: DraftWidth,
        ids: Vec<TokenId>,
    ) -> Result<Self, ShapeMismatch>
    {
        let expected = usize::try_from(NonZeroU32::from(width).get())
            .ok()
            .ok_or(ShapeMismatch::Width)?;
        if ids.len() != expected {
            return Err(ShapeMismatch::Drafts);
        }
        return Ok(Self { ids });
    }

    /// The ids.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn ids(&self) -> &[TokenId]
    {
        return &self.ids;
    }
}

/// The Accept fragment's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Acceptance
{
    /// The accepted drafts followed by the correction or bonus token.
    licensed: Vec<TokenId>,
    /// How many drafts were accepted.
    accepted: TokenCount,
}

impl Acceptance
{
    /// The licensed tokens.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub fn licensed(&self) -> &[TokenId]
    {
        return &self.licensed;
    }

    /// The accepted length.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    #[must_use]
    pub const fn accepted(&self) -> TokenCount
    {
        return self.accepted;
    }

    /// The acceptance whose licensed ids are `licensed`: every id but the last
    /// accepted, the last the correction or bonus.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success `accepted` is one less than the count.
    /// - provides: a device's answer in the reference's form.
    /// - fails: with [`ShapeMismatch::Drafts`] when `licensed` is empty.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`ShapeMismatch::Drafts`]: nothing is licensed.
    /// - [`ShapeMismatch::Width`]: the count exceeds `u32`.
    #[inline]
    pub fn new(licensed: Vec<TokenId>) -> Result<Self, ShapeMismatch>
    {
        let accepted = licensed.len().checked_sub(1).ok_or(ShapeMismatch::Drafts)?;
        let accepted = TokenCount::from(u32::try_from(accepted).ok().ok_or(ShapeMismatch::Width)?);
        return Ok(Self { licensed, accepted });
    }

    /// Accept against `targets`: the first column whose draft differs from
    /// its target ends the prefix.
    ///
    /// # Specification
    /// - requires: `targets` holds one id per verify column.
    /// - ensures: `accepted` is the lowest `i < K` with `drafts[i] !=
    ///   targets[i]`, or `K` when there is none; `licensed` is
    ///   `targets[..=accepted]`.
    /// - provides: the prefix step, shared by the reference and by a device
    ///   program that leaves the prefix to the host.
    /// - fails: with [`ShapeMismatch::Drafts`] when `targets` is not one longer
    ///   than the drafts.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`ShapeMismatch::Drafts`]: the column counts disagree.
    /// - [`ShapeMismatch::Width`]: the accepted length exceeds `u32`.
    ///
    /// # Adequacy
    /// - hypothesis: L3 — full acceptance, rejection at the first draft, and
    ///   rejection mid-block each yield their exact length and licensed ids.
    /// - witness: `tests::every_draft_accepted_licenses_the_bonus`
    /// - witness: `tests::a_first_draft_mismatch_licenses_the_correction_alone`
    /// - witness: `tests::a_mid_block_mismatch_stops_the_prefix_there`
    #[inline]
    pub fn from_targets(
        targets: &[TokenId],
        drafts: &DraftBlock,
    ) -> Result<Self, ShapeMismatch>
    {
        if targets.len() != drafts.ids.len().saturating_add(1) {
            return Err(ShapeMismatch::Drafts);
        }
        let accepted = targets
            .iter()
            .zip(&drafts.ids)
            .take_while(|&(target, draft)| return target == draft)
            .count();
        let licensed = targets
            .iter()
            .take(accepted.saturating_add(1))
            .copied()
            .collect();
        let accepted = TokenCount::from(u32::try_from(accepted).ok().ok_or(ShapeMismatch::Width)?);
        return Ok(Self { licensed, accepted });
    }
}

impl core::fmt::Display for Acceptance
{
    /// Render the accepted length and the licensed ids.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    {
        write!(f, "{} accepted, licensed [", self.accepted)?;
        for (index, id) in self.licensed.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{id}")?;
        }
        return f.write_str("]");
    }
}

/// Each verify column's argmax over the valid vocabulary, the lowest id
/// winning a tie.
///
/// # Specification
/// - requires: no valid logit is NaN.
/// - ensures: the result holds `K + 1` ids; each is below the vocabulary size,
///   holds its column's greatest valid logit, and no lower id holds an equal
///   one; padding logits never win.
/// - provides: the target ids the device must reproduce.
/// - fails: with [`ShapeMismatch::Width`] when an id exceeds `i32`.
/// - panics: none.
///
/// # Errors
/// - [`ShapeMismatch::Width`]: an id exceeds `i32`.
///
/// # Adequacy
/// - hypothesis: L3 — a tie between two ids yields the lower; a padding logit
///   above every valid one does not win.
/// - witness: `tests::a_tie_goes_to_the_lower_id`
/// - witness: `tests::padding_never_wins`
#[inline]
pub fn reference_targets(logits: &VerifyLogits) -> Result<Vec<TokenId>, ShapeMismatch>
{
    let padded = usize::try_from(u32::from(logits.layout.padded()))
        .ok()
        .ok_or(ShapeMismatch::Width)?;
    let valid = usize::try_from(u32::from(logits.layout.size))
        .ok()
        .ok_or(ShapeMismatch::Width)?;
    let mut targets = Vec::with_capacity(usize::from(column_count(logits.width)?));
    for row in logits.bits.as_ref().chunks_exact(padded) {
        let mut best = (0_usize, f32::NEG_INFINITY);
        for (id, &bits) in row.iter().take(valid).enumerate() {
            let value = f32::from(Bf16::from(bits));
            if value > best.1 {
                best = (id, value);
            }
        }
        let id = i32::try_from(best.0).ok().ok_or(ShapeMismatch::Width)?;
        targets.push(TokenId::from(id));
    }
    return Ok(targets);
}

/// The host reference for the whole fragment.
///
/// # Specification
/// - requires: no valid logit is NaN.
/// - ensures: the result is [`Acceptance::from_targets`] over
///   [`reference_targets`].
/// - provides: the answer the device is compared against.
/// - fails: as the two steps do.
/// - panics: none.
///
/// # Errors
/// - [`ShapeMismatch`]: the drafts and logits disagree on `K`, or a count
///   leaves its width.
#[inline]
pub fn reference_accept(
    logits: &VerifyLogits,
    drafts: &DraftBlock,
) -> Result<Acceptance, ShapeMismatch>
{
    let targets = reference_targets(logits)?;
    return Acceptance::from_targets(&targets, drafts);
}

/// Tests for the reference and the layout.
#[cfg(test)]
mod tests
{
    use core::num::NonZeroU32;
    use core::str::FromStr as _;

    use infinitum_round::DraftWidth;
    use infinitum_round::TokenCount;
    use infinitum_round::TokenId;

    use super::Acceptance;
    use super::Bf16;
    use super::ChunkWidth;
    use super::DraftBlock;
    use super::LayoutFailure;
    use super::VerifyLogits;
    use super::VocabularyLayout;
    use super::VocabularySize;
    use super::reference_targets;

    /// Ids from integers.
    macro_rules! ids {
        ($($id:expr),* $(,)?) => {
            [$(TokenId::from($id)),*]
        };
    }

    /// One verify column's eight logits.
    #[repr(transparent)]
    struct Row([f32; 8]);

    /// A two-draft block.
    ///
    /// # Specification
    /// trivial.
    fn drafts(ids: &[TokenId]) -> DraftBlock
    {
        return DraftBlock::new(DraftWidth::from_str("2").unwrap(), ids.to_vec()).unwrap();
    }

    /// A one-draft verify block over a vocabulary of 5 in chunks of 4.
    ///
    /// # Specification
    /// trivial.
    fn logits(rows: &[Row; 2]) -> VerifyLogits
    {
        let layout = VocabularyLayout::new(
            VocabularySize::from(NonZeroU32::new(5).unwrap()),
            ChunkWidth::try_from(NonZeroU32::new(4).unwrap()).unwrap(),
        )
        .unwrap();
        let values = rows
            .iter()
            .flat_map(|row| return row.0)
            .map(Bf16::from)
            .collect();
        return VerifyLogits::new(layout, DraftWidth::from_str("1").unwrap(), values).unwrap();
    }

    /// The served vocabulary pads to exactly 970 chunks of 256.
    #[test]
    fn the_served_vocabulary_tiles_into_970_chunks()
    {
        let layout = VocabularyLayout::new(
            VocabularySize::from(NonZeroU32::new(248_077).unwrap()),
            ChunkWidth::try_from(NonZeroU32::new(256).unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            layout,
            VocabularyLayout::served(),
            "the constant is the computed layout"
        );
        assert_eq!(u32::from(layout.padded()), 248_320, "970 chunks of 256");
    }

    /// Past 8192 chunks the chunk index's high digit leaves bfloat16's exact
    /// range.
    #[test]
    fn a_vocabulary_past_8192_chunks_is_refused()
    {
        let chunk = ChunkWidth::try_from(NonZeroU32::new(256).unwrap()).unwrap();
        let at = VocabularySize::from(NonZeroU32::new(0x0020_0000).unwrap());
        let past = VocabularySize::from(NonZeroU32::new(0x0020_0001).unwrap());
        assert!(
            VocabularyLayout::new(at, chunk).is_ok(),
            "8192 chunks of 256 fit"
        );
        assert_eq!(
            VocabularyLayout::new(past, chunk),
            Err(LayoutFailure::TooManyChunks),
            "one more id needs chunk 8192"
        );
    }

    /// All drafts match: the bonus column is licensed after them.
    #[test]
    fn every_draft_accepted_licenses_the_bonus()
    {
        let answer =
            Acceptance::from_targets(&ids![4_i32, 7_i32, 9_i32], &drafts(&ids![4_i32, 7_i32]))
                .unwrap();
        assert_eq!(
            answer.accepted(),
            TokenCount::from(2),
            "both drafts accepted"
        );
        assert_eq!(
            answer.licensed(),
            ids![4_i32, 7_i32, 9_i32].as_slice(),
            "drafts then bonus"
        );
    }

    /// The first draft differs: only the correction is licensed.
    #[test]
    fn a_first_draft_mismatch_licenses_the_correction_alone()
    {
        let answer =
            Acceptance::from_targets(&ids![4_i32, 7_i32, 9_i32], &drafts(&ids![5_i32, 7_i32]))
                .unwrap();
        assert_eq!(answer.accepted(), TokenCount::from(0), "nothing accepted");
        assert_eq!(
            answer.licensed(),
            ids![4_i32].as_slice(),
            "the correction alone"
        );
    }

    /// A later match does not resume the prefix.
    #[test]
    fn a_mid_block_mismatch_stops_the_prefix_there()
    {
        let targets = ids![4_i32, 7_i32, 9_i32, 2_i32];
        let block = DraftBlock::new(
            DraftWidth::from_str("3").unwrap(),
            ids![4_i32, 8_i32, 9_i32].to_vec(),
        )
        .unwrap();
        let answer = Acceptance::from_targets(&targets, &block).unwrap();
        assert_eq!(answer.accepted(), TokenCount::from(1), "one draft accepted");
        assert_eq!(
            answer.licensed(),
            ids![4_i32, 7_i32].as_slice(),
            "the draft then the correction"
        );
    }

    /// Equal maxima at ids 1 and 3: id 1 is the target.
    #[test]
    fn a_tie_goes_to_the_lower_id()
    {
        let block = logits(&[
            Row([0.0, 2.0, 1.0, 2.0, 0.5, 0.0, 0.0, 0.0]),
            Row([0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0]),
        ]);
        assert_eq!(
            reference_targets(&block).unwrap(),
            ids![1_i32, 4_i32],
            "the lower of two equal maxima"
        );
    }

    /// Ids 5 to 7 are padding; their larger logits never win.
    #[test]
    fn padding_never_wins()
    {
        let block = logits(&[
            Row([0.0, 1.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0]),
            Row([f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.25, 9.0, 0.0, 0.0]),
        ]);
        assert_eq!(
            reference_targets(&block).unwrap(),
            ids![1_i32, 4_i32],
            "valid ids only"
        );
    }
}
