use std::{borrow::Cow, slice};

use ndarray::{Array, ArrayView, Dimension};

pub fn to_contiguous<'a, T: Clone, D>(arr: ArrayView<'a, T, D>) -> Array<T, D>
where
    D: Dimension,
{
    arr.as_standard_layout().to_owned()
}

pub fn to_contiguous2<'a, T: Clone, D>(arr: Array<T, D>) -> Array<T, D>
where
    D: Dimension,
{
    if arr.is_standard_layout() {
        return arr;
    } else {
        arr.as_standard_layout().to_owned()
    }
}

pub fn as_slice<'a, T: Clone, D>(arr: ArrayView<'a, T, D>) -> Cow<'a, [T]>
where
    D: Dimension,
{
    if arr.is_standard_layout() {
        Cow::Borrowed(unsafe { slice::from_raw_parts(arr.as_ptr(), arr.len()) })
    } else {
        let (vec, offset) = arr
            .as_standard_layout()
            .to_owned()
            .into_raw_vec_and_offset();
        assert_eq!(offset.unwrap_or_default(), 0);
        Cow::Owned(vec)
    }
}
