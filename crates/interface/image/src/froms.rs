use image::{DynamicImage, GrayImage};
use ndarray::{Array2, Array3};
use opencv::core::{Mat, MatTraitConst as _};

use crate::{DimType, Mask, RawImage};

impl From<Array3<f32>> for RawImage {
    fn from(input: Array3<f32>) -> Self {
        Self::from(input.mapv(|v| {
            let clamped = v.clamp(0.0, 1.0);
            (clamped * 255.0) as u8
        }))
    }
}

impl TryFrom<Mat> for RawImage {
    type Error = opencv::Error;

    fn try_from(value: Mat) -> Result<Self, Self::Error> {
        let resized = if value.is_continuous() {
            value
        } else {
            // allow:clone[to_contiguous]
            value.clone()
        };

        let size = resized.size()?;
        let rows = size.height as usize;
        let cols = size.width as usize;
        let channels = resized.channels() as usize;

        let total_len = rows * cols * channels;
        //TODO: take data instead of copy
        let data: &[u8] = unsafe { std::slice::from_raw_parts(resized.data(), total_len) };

        Ok(Self {
            data: data.to_vec(),
            width: cols as DimType,
            height: rows as DimType,
            channels: channels as u8,
        })
    }
}

impl From<DynamicImage> for RawImage {
    fn from(value: DynamicImage) -> Self {
        Self {
            width: value.width() as DimType,
            height: value.height() as DimType,
            channels: 3,
            data: value.into_rgb8().as_raw().to_vec(),
        }
    }
}

impl From<Mask> for RawImage {
    fn from(mask: Mask) -> Self {
        RawImage {
            data: mask.data.into_iter().flat_map(|v| vec![v, v, v]).collect(),
            width: mask.width,
            height: mask.height,
            channels: 3,
        }
    }
}

impl From<Array2<u8>> for RawImage {
    fn from(mask: Array2<u8>) -> Self {
        let (height, width) = mask.dim();
        let rgb = mask
            .broadcast((mask.shape()[0], mask.shape()[1], 3))
            .unwrap()
            .to_owned();
        assert!(rgb.is_standard_layout());
        let (v, offset) = rgb.into_raw_vec_and_offset();
        assert_eq!(offset.unwrap_or_default(), 0);

        RawImage {
            data: v,
            width: width as u16,
            height: height as u16,
            channels: 3,
        }
    }
}

impl From<Mat> for Mask {
    fn from(value: Mat) -> Self {
        let resized = if value.is_continuous() {
            value
        } else {
            // allow:clone[to_contiguous]
            value.clone()
        };

        let size = resized.size().unwrap();
        let rows = size.height as usize;
        let cols = size.width as usize;
        let channels = resized.channels() as usize;

        assert_eq!(channels, 1);

        let total_len = rows * cols * channels;
        //TODO: take data instead of cloning
        let data: &[u8] = unsafe { std::slice::from_raw_parts(resized.data(), total_len) };

        Self {
            data: data.to_vec(),
            width: cols as DimType,
            height: rows as DimType,
        }
    }
}

impl From<GrayImage> for Mask {
    fn from(value: GrayImage) -> Self {
        let (w, h) = value.dimensions();
        Mask {
            width: w as DimType,
            height: h as DimType,
            data: value.into_raw(),
        }
    }
}

impl From<Array2<u8>> for Mask {
    fn from(mut mask: Array2<u8>) -> Self {
        let (height, width) = mask.dim();
        if !mask.is_standard_layout() {
            mask = mask.as_standard_layout().to_owned();
        }

        let (v, offset) = mask.into_raw_vec_and_offset();
        assert_eq!(offset.unwrap_or_default(), 0);

        Mask {
            data: v,
            width: width as u16,
            height: height as u16,
        }
    }
}

impl From<Array3<u8>> for RawImage {
    fn from(mut value: Array3<u8>) -> Self {
        let (height, width, channels) = value.dim();

        if !value.is_standard_layout() {
            value = value.as_standard_layout().to_owned();
        }
        let (v, offset) = value.into_raw_vec_and_offset();
        assert_eq!(offset.unwrap_or_default(), 0);

        RawImage {
            data: v,
            width: width as u16,
            height: height as u16,
            channels: channels as u8,
        }
    }
}
