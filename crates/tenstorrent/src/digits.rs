//! Ids as the device carries them: three small integers in bfloat16.
//!
//! The lowering's elementwise float32 path rounds its operands to TF32, so an
//! id past 2^11 cannot cross the device as one float. Every index the device
//! touches is instead an integer below 1024, exact in bfloat16: an id `i` in
//! chunks of `W` travels as the chunk's high digit `(i / W) / 32`, its low
//! digit `(i / W) % 32`, and the local index `i % W`.

use core::num::NonZeroU32;

use infinitum_round::DraftWidth;
use infinitum_round::TokenId;

use crate::accept::Bf16;
use crate::accept::DeviceBits;
use crate::accept::DraftBlock;
use crate::accept::ShapeMismatch;
use crate::accept::VocabularyLayout;

/// The base of a chunk index's two digits.
const CHUNK_BASE: u32 = 32;

/// The high digit of the bonus column's draft: above every target's high
/// digit, which stays below 256, so the bonus column never matches.
const SENTINEL_HI: u16 = 512;

/// A small non-negative integer, exact in bfloat16.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Digit(u16);

impl From<Digit> for Bf16
{
    /// The digit's bfloat16 value.
    ///
    /// # Specification
    /// - requires: the digit is at most 256, or a power of two; every digit
    ///   this module builds is.
    /// - ensures: the result is the digit's value exactly.
    /// - provides: the device's form of a digit.
    /// - fails: never.
    /// - panics: none.
    #[inline]
    fn from(digit: Digit) -> Self
    {
        return Self::from(f32::from(digit.0));
    }
}

impl From<Digit> for u16
{
    /// The digit's value.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(digit: Digit) -> Self
    {
        return digit.0;
    }
}

impl TryFrom<Bf16> for Digit
{
    type Error = DigitFailure;

    /// Read a bfloat16 value as a digit.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: on success the digit is the value, which was a non-negative
    ///   integer below 2^16.
    /// - provides: the device's digits, decoded without allocation.
    /// - fails: with [`DigitFailure::NotIntegral`] for a negative value, a
    ///   fraction, an infinity, a NaN, or a value from 2^16 up.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`DigitFailure::NotIntegral`]: as described.
    ///
    /// # Adequacy
    /// - hypothesis: L3 — every integer the device returns decodes to itself; a
    ///   fraction is refused.
    /// - witness: `tests::every_small_integer_round_trips`
    /// - witness: `tests::a_fraction_is_not_a_digit`
    #[inline]
    fn try_from(value: Bf16) -> Result<Self, Self::Error>
    {
        let bits = u16::from(value);
        if bits.trailing_zeros() >= 15 {
            return Ok(Self(0));
        }
        if bits & 0x8000 != 0 {
            return Err(DigitFailure::NotIntegral);
        }
        let exponent = bits.checked_shr(7).unwrap_or_default() & 0xff;
        let power = exponent.checked_sub(127).ok_or(DigitFailure::NotIntegral)?;
        if power > 15 {
            return Err(DigitFailure::NotIntegral);
        }
        let significand = u32::from(bits & 0x7f | 0x80);
        let value = if power >= 7 {
            significand
                .checked_shl(u32::from(power).saturating_sub(7))
                .unwrap_or_default()
        }
        else {
            let shift = 7_u32.saturating_sub(u32::from(power));
            let fraction = significand
                & 1_u32
                    .checked_shl(shift)
                    .unwrap_or_default()
                    .saturating_sub(1);
            if fraction != 0 {
                return Err(DigitFailure::NotIntegral);
            }
            significand.checked_shr(shift).unwrap_or_default()
        };
        return u16::try_from(value)
            .ok()
            .map(Self)
            .ok_or(DigitFailure::NotIntegral);
    }
}

/// Why the device's digits are not an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigitFailure
{
    /// A value is not a non-negative integer below 2^16.
    NotIntegral,
    /// The digits name no valid id of the vocabulary.
    OutOfRange,
}

impl core::fmt::Display for DigitFailure
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
            | Self::NotIntegral => "a device value is not a small non-negative integer",
            | Self::OutOfRange => "the device's digits name no vocabulary id",
        });
    }
}

impl core::error::Error for DigitFailure
{
}

/// An id's three digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdDigits
{
    /// The chunk index divided by 32.
    pub hi: Digit,
    /// The chunk index modulo 32.
    pub lo: Digit,
    /// The index within the chunk.
    pub local: Digit,
}

impl IdDigits
{
    /// All three digits zero: what the device writes past the accepted
    /// length.
    pub const ZERO: Self = Self {
        hi: Digit(0),
        lo: Digit(0),
        local: Digit(0),
    };
}

impl TryFrom<u32> for Digit
{
    type Error = DigitFailure;

    /// A small `u32` as a digit.
    ///
    /// # Specification
    /// - requires: nothing.
    /// - ensures: the digit is `value` when it fits `u16`.
    /// - provides: the one narrowing digits go through.
    /// - fails: with [`DigitFailure::OutOfRange`] otherwise.
    /// - panics: none.
    ///
    /// # Errors
    /// - [`DigitFailure::OutOfRange`]: `value` exceeds `u16`.
    #[inline]
    fn try_from(value: u32) -> Result<Self, Self::Error>
    {
        return u16::try_from(value)
            .ok()
            .map(Self)
            .ok_or(DigitFailure::OutOfRange);
    }
}

/// Split `id` into its digits under `layout`.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success `compose(layout, split(layout, id)) == id`.
/// - provides: the drafts' digits.
/// - fails: with [`DigitFailure::OutOfRange`] for a negative id or one past the
///   vocabulary.
/// - panics: none.
///
/// # Errors
/// - [`DigitFailure::OutOfRange`]: `id` is not a vocabulary id.
///
/// # Adequacy
/// - hypothesis: L3 — ids at chunk boundaries and the last valid id round trip;
///   the first id past the vocabulary is refused.
/// - witness: `tests::boundary_ids_round_trip`
#[inline]
pub fn split(
    layout: VocabularyLayout,
    id: TokenId,
) -> Result<IdDigits, DigitFailure>
{
    let id = u32::try_from(i32::from(id))
        .ok()
        .ok_or(DigitFailure::OutOfRange)?;
    if id >= u32::from(layout.size()) {
        return Err(DigitFailure::OutOfRange);
    }
    let width = u32::from(layout.chunk());
    let chunk = id.checked_div(width).ok_or(DigitFailure::OutOfRange)?;
    return Ok(IdDigits {
        hi: Digit::try_from(
            chunk
                .checked_div(CHUNK_BASE)
                .ok_or(DigitFailure::OutOfRange)?,
        )?,
        lo: Digit::try_from(
            chunk
                .checked_rem(CHUNK_BASE)
                .ok_or(DigitFailure::OutOfRange)?,
        )?,
        local: Digit::try_from(id.checked_rem(width).ok_or(DigitFailure::OutOfRange)?)?,
    });
}

/// The id `digits` name under `layout`.
///
/// # Specification
/// - requires: nothing.
/// - ensures: on success the id is `(hi * 32 + lo) * W + local`, below the
///   vocabulary size, with `lo < 32` and `local < W`.
/// - provides: the device's targets as ids.
/// - fails: with [`DigitFailure::OutOfRange`] when a digit is out of its range
///   or the id is not a vocabulary id.
/// - panics: none.
///
/// # Errors
/// - [`DigitFailure::OutOfRange`]: as described.
#[inline]
pub fn compose(
    layout: VocabularyLayout,
    digits: IdDigits,
) -> Result<TokenId, DigitFailure>
{
    let width = u32::from(layout.chunk());
    let (hi, lo, local) = (
        u32::from(digits.hi.0),
        u32::from(digits.lo.0),
        u32::from(digits.local.0),
    );
    if lo >= CHUNK_BASE || local >= width {
        return Err(DigitFailure::OutOfRange);
    }
    let id = hi
        .checked_mul(CHUNK_BASE)
        .and_then(|chunk| return chunk.checked_add(lo))
        .and_then(|chunk| return chunk.checked_mul(width))
        .and_then(|base| return base.checked_add(local))
        .ok_or(DigitFailure::OutOfRange)?;
    if id >= u32::from(layout.size()) {
        return Err(DigitFailure::OutOfRange);
    }
    return i32::try_from(id)
        .ok()
        .map(TokenId::from)
        .ok_or(DigitFailure::OutOfRange);
}

/// The constant planes a program reads on every call, in its input order:
/// the pad and local planes over `[K + 1, C, W]`, then the chunk index's high
/// and low digits over `[K + 1, C]`.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `pad` is one exactly at ids past the vocabulary; `local` is `id %
///   W`; `hi` and `lo` are the chunk index's digits; every row repeats the
///   first.
/// - provides: the constants the program reads rather than computes, since this
///   lowering's in-program `arrange` over a rank-3 dimension is wrong.
/// - fails: with [`ShapeMismatch::Width`] when a count overflows.
/// - panics: none.
///
/// # Errors
/// - [`ShapeMismatch::Width`]: a count overflows.
#[inline]
pub fn constant_planes(
    layout: VocabularyLayout,
    width: DraftWidth,
) -> Result<DeviceBits, ShapeMismatch>
{
    let columns = NonZeroU32::from(width)
        .get()
        .checked_add(1)
        .ok_or(ShapeMismatch::Width)?;
    let chunk = u32::from(layout.chunk());
    let chunks = u32::from(layout.chunks());
    let padded = u32::from(layout.padded());
    let size = u32::from(layout.size());
    let total = padded
        .checked_mul(2)
        .and_then(|planes| return planes.checked_add(chunks.checked_mul(2)?))
        .and_then(|row| return row.checked_mul(columns))
        .ok_or(ShapeMismatch::Width)?;
    let mut planes = Vec::with_capacity(usize::try_from(total).ok().ok_or(ShapeMismatch::Width)?);
    let bits = |value: u32| -> Result<u16, ShapeMismatch> {
        let digit = Digit::try_from(value).ok().ok_or(ShapeMismatch::Width)?;
        return Ok(u16::from(Bf16::from(digit)));
    };
    for _ in 0 .. columns {
        for id in 0 .. padded {
            planes.push(bits(u32::from(id >= size))?);
        }
    }
    for _ in 0 .. columns {
        for id in 0 .. padded {
            planes.push(bits(id.checked_rem(chunk).ok_or(ShapeMismatch::Width)?)?);
        }
    }
    for _ in 0 .. columns {
        for index in 0 .. chunks {
            planes.push(bits(
                index.checked_div(CHUNK_BASE).ok_or(ShapeMismatch::Width)?,
            )?);
        }
    }
    for _ in 0 .. columns {
        for index in 0 .. chunks {
            planes.push(bits(
                index.checked_rem(CHUNK_BASE).ok_or(ShapeMismatch::Width)?,
            )?);
        }
    }
    return Ok(DeviceBits::from(planes));
}

/// Fill `tail` with one call's per-call inputs.
///
/// In the program's order: the drafts' high, low, and local digits, each with
/// the bonus column's sentinel last; then each column's position, the `K`
/// fill, and zeros; all `[K + 1, 1]`.
///
/// # Specification
/// - requires: nothing.
/// - ensures: `tail` holds `6 * (K + 1)` values; the sentinel's high digit
///   exceeds every target's.
/// - provides: the drafts the device compares its targets with.
/// - fails: with [`DigitFailure::OutOfRange`] when a draft is not a vocabulary
///   id.
/// - panics: none.
///
/// # Errors
/// - [`DigitFailure::OutOfRange`]: a draft is not a vocabulary id, or `K`
///   exceeds a digit.
#[inline]
pub fn fill_tail(
    layout: VocabularyLayout,
    drafts: &DraftBlock,
    tail: &mut DeviceBits,
) -> Result<(), DigitFailure>
{
    tail.clear();
    let bits = |digit: Digit| return u16::from(Bf16::from(digit));
    let sentinel = IdDigits {
        hi: Digit(SENTINEL_HI),
        lo: Digit(0),
        local: Digit(0),
    };
    let mut split_all = Vec::with_capacity(drafts.ids().len().saturating_add(1));
    for &id in drafts.ids() {
        split_all.push(split(layout, id)?);
    }
    split_all.push(sentinel);
    tail.extend(split_all.iter().map(|digits| return bits(digits.hi)));
    tail.extend(split_all.iter().map(|digits| return bits(digits.lo)));
    tail.extend(split_all.iter().map(|digits| return bits(digits.local)));
    let columns = u32::try_from(split_all.len())
        .ok()
        .ok_or(DigitFailure::OutOfRange)?;
    for position in 0 .. columns {
        tail.extend(core::iter::once(bits(Digit::try_from(position)?)));
    }
    let drafted = Digit::try_from(columns.saturating_sub(1))?;
    tail.extend(core::iter::repeat_n(bits(drafted), split_all.len()));
    tail.extend(core::iter::repeat_n(0_u16, split_all.len()));
    return Ok(());
}

#[cfg(test)]
mod tests
{
    use infinitum_round::TokenId;

    use super::Digit;
    use super::DigitFailure;
    use super::compose;
    use super::split;
    use crate::accept::Bf16;
    use crate::accept::VocabularyLayout;

    /// Every integer the device can return round trips through bfloat16.
    #[test]
    fn every_small_integer_round_trips()
    {
        for value in 0_u16 ..= 256_u16 {
            let bits = Bf16::from(Digit(value));
            assert_eq!(
                Digit::try_from(bits),
                Ok(Digit(value)),
                "{value} decodes to itself"
            );
        }
        for value in [512_u16, 1024_u16] {
            assert_eq!(
                Digit::try_from(Bf16::from(Digit(value))),
                Ok(Digit(value)),
                "{value} decodes to itself"
            );
        }
    }

    /// 2.5 is not a digit, nor is -1.
    #[test]
    fn a_fraction_is_not_a_digit()
    {
        assert_eq!(
            Digit::try_from(Bf16::from(2.5_f32)),
            Err(DigitFailure::NotIntegral),
            "2.5"
        );
        assert_eq!(
            Digit::try_from(Bf16::from(-1.0_f32)),
            Err(DigitFailure::NotIntegral),
            "-1"
        );
    }

    /// The first and last ids, ids either side of a chunk and of a digit
    /// boundary, and the last valid id round trip; the first padding id does
    /// not split.
    #[test]
    fn boundary_ids_round_trip()
    {
        let layout = VocabularyLayout::served();
        for id in [
            0_i32,
            255_i32,
            256_i32,
            8191_i32,
            8192_i32,
            248_063_i32,
            248_064_i32,
            248_076_i32,
        ] {
            let digits = split(layout, TokenId::from(id)).unwrap();
            assert_eq!(
                compose(layout, digits),
                Ok(TokenId::from(id)),
                "{id} round trips"
            );
        }
        assert_eq!(
            split(layout, TokenId::from(248_077_i32)),
            Err(DigitFailure::OutOfRange),
            "padding"
        );
    }
}
