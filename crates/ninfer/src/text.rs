//! Text across the Engine: the raw text a prompt is encoded from, and the
//! bytes generated ids render as.

/// Text encoded as given: no chat template and no special token is added.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawText<'text>(&'text str);

impl<'text> From<&'text str> for RawText<'text>
{
    /// Borrow the text.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(text: &'text str) -> Self
    {
        return Self(text);
    }
}

impl AsRef<str> for RawText<'_>
{
    /// The text.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn as_ref(&self) -> &str
    {
        return self.0;
    }
}

/// Bytes rendered from token ids: exactly what the ids encode, not UTF-8
/// validated, since a generation budget can end inside a multi-byte
/// character.
#[repr(transparent)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderedBytes(Vec<u8>);

impl From<Vec<u8>> for RenderedBytes
{
    /// Take the bytes.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn from(bytes: Vec<u8>) -> Self
    {
        return Self(bytes);
    }
}

impl AsRef<[u8]> for RenderedBytes
{
    /// The bytes.
    ///
    /// # Specification
    /// trivial.
    #[inline]
    fn as_ref(&self) -> &[u8]
    {
        return &self.0;
    }
}
