use crate::{DimType, ImageOp, Mask, MaskView, RawImage, RawImageCow, RawImageView};

#[derive(Default)]
pub struct DummyImageProcessor;

impl DummyImageProcessor {
    pub fn new() -> Self {
        DummyImageProcessor
    }
}

impl ImageOp for DummyImageProcessor {
    fn invert(&self, image: super::RawImage) -> super::RawImage {
        image
    }

    fn add_border_wh<'a>(
        &self,
        img: RawImageView<'a>,
        width: super::DimType,
        height: super::DimType,
    ) -> RawImageCow<'a> {
        if img.width > width && img.height > height {
            return RawImageCow::Borrowed(img);
        }
        RawImageCow::Owned(super::RawImage {
            data: vec![0; width as usize * height as usize * 3],
            width,
            height,
            channels: 3,
        })
    }

    fn add_border_center(&self, _: super::RawImage, width: super::DimType) -> super::RawImage {
        super::RawImage {
            data: vec![0; width as usize * width as usize * 3],
            width,
            height: width,
            channels: 3,
        }
    }

    fn remove_border(
        &self,
        _: super::RawImageView,
        width: super::DimType,
        height: super::DimType,
    ) -> super::RawImage {
        super::RawImage {
            data: vec![0; width as usize * height as usize * 3],
            width,
            height,
            channels: 3,
        }
    }

    fn remove_border_center(
        &self,
        _: super::RawImage,
        width: super::DimType,
        height: super::DimType,
    ) -> super::RawImage {
        super::RawImage {
            data: vec![0; width as usize * height as usize * 3],
            width,
            height,
            channels: 3,
        }
    }

    fn rotate_right(&self, mut image: super::RawImageView) -> super::RawImage {
        let temp = image.height;
        image.height = image.width;
        image.width = temp;
        RawImage {
            data: vec![0; image.data.len()],
            width: image.width,
            height: image.height,
            channels: image.channels,
        }
    }

    fn rotate_left(&self, mut image: super::RawImage) -> super::RawImage {
        let temp = image.height;
        image.height = image.width;
        image.width = temp;
        image
    }

    fn rotate_left_mask(&self, mut mask: Mask) -> Mask {
        let temp = mask.height;
        mask.height = mask.width;
        mask.width = temp;
        mask
    }

    fn gamma_correction(&self, image: super::RawImageView) -> super::RawImage {
        RawImage {
            data: vec![0; image.data.len()],
            width: image.width,
            height: image.height,
            channels: image.channels,
        }
    }

    fn histogram_equalization(&self, image: super::RawImage) -> super::RawImage {
        image
    }

    fn resize(
        &self,
        _: RawImageView,
        width: super::DimType,
        height: super::DimType,
        _: super::Interpolation,
    ) -> anyhow::Result<super::RawImage> {
        Ok(super::RawImage {
            data: vec![0; width as usize * height as usize * 3],
            width,
            height,
            channels: 3,
        })
    }

    fn resize_mask(
        &self,
        _: MaskView,
        width: usize,
        height: usize,
        _: super::Interpolation,
    ) -> anyhow::Result<Mask> {
        Ok(Mask {
            width: width as DimType,
            height: height as DimType,
            data: vec![0; width * height],
        })
    }

    fn remove_border_mask(&self, _: Mask, width: super::DimType, height: super::DimType) -> Mask {
        Mask {
            data: vec![0; width as usize * height as usize],
            width,
            height,
        }
    }

    fn transpose(&self, image: super::RawImageView) -> super::RawImage {
        super::RawImage {
            data: vec![0; image.width as usize * image.height as usize],
            width: image.height,
            height: image.width,
            channels: image.channels,
        }
    }

    fn bgr_to_rgb(&self, img: super::RawImage) -> super::RawImage {
        img
    }

    fn add_border_center_wh<'a>(
        &self,
        _: crate::RawImageView<'a>,
        width: DimType,
        _: DimType,
    ) -> crate::RawImageCow<'a> {
        RawImageCow::Owned(super::RawImage {
            data: vec![0; width as usize * width as usize * 3],
            width,
            height: width,
            channels: 3,
        })
    }

    fn mask_func(&self, mask1: Mask, _: Mask, _: fn(u8, u8) -> u8) -> Mask {
        mask1
    }
}
