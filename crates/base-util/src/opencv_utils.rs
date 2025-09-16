use std::{borrow::Cow, mem};

use opencv::core::{Mat, MatTraitConst as _};

pub fn to_continous(value: Mat) -> Mat {
    if value.is_continuous() {
        value
    } else {
        // allow:clone[to_contiguous]
        value.clone()
    }
}

pub fn to_continous2(value: &Mat) -> Cow<Mat> {
    if value.is_continuous() {
        Cow::Borrowed(value)
    } else {
        // allow:clone[to_contiguous]
        Cow::Owned(value.clone())
    }
}

pub fn to_continuous_inplace(s: &mut Mat) -> *const u8 {
    if s.is_continuous() {
        let data_ptr = s.data();
        data_ptr
    } else {
        // allow:clone[to_contiguous]
        let mut new = s.clone();
        mem::swap(s, &mut new);
        let data_ptr = s.data();
        data_ptr
    }
}

pub fn as_slice<T>(s: &mut Mat) -> &[T] {
    let len = s.total() * s.channels() as usize;
    let data_ptr = to_continuous_inplace(s) as *const T;
    let slice = unsafe { std::slice::from_raw_parts(data_ptr, len) };
    slice
}
