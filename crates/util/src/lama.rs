use std::sync::Arc;

use interface_image::{DimType, ImageOp, Mask, RawImage, RawImageCow, RawImageView};

pub fn resize_keep_aspect(
    img: RawImageView,
    size: u16,
    img_processor: &Arc<dyn ImageOp + Send + Sync>,
) -> anyhow::Result<RawImage> {
    let ratio = size as f64 / img.width.max(img.height) as f64;
    let new_width = img.width as f64 * ratio;
    let new_height = img.height as f64 * ratio;

    img_processor.resize(
        img,
        new_width as DimType,
        new_height as DimType,
        interface_image::Interpolation::BilinearExact,
    )
}

pub fn resize_keep_aspect_mask(
    img: Mask,
    size: u16,
    img_processor: &Arc<dyn ImageOp + Send + Sync>,
) -> anyhow::Result<Mask> {
    let ratio = size as f64 / img.width.max(img.height) as f64;
    let new_width = img.width as f64 * ratio;
    let new_height = img.height as f64 * ratio;

    img_processor.resize_mask(
        img.view(),
        new_width as usize,
        new_height as usize,
        interface_image::Interpolation::BilinearExact,
    )
}

pub fn lama_resize_image<'a>(
    image: RawImageView<'a>,
    mut mask: Mask,
    inpainting_size: u16,
    img_processor: &Arc<dyn ImageOp + Send + Sync>,
) -> anyhow::Result<(RawImageCow<'a>, Mask)> {
    let w = image.width;
    let h = image.height;
    let mut image = RawImageCow::Borrowed(image);
    if w.max(h) > inpainting_size {
        image = RawImageCow::Owned(resize_keep_aspect(
            image.view(),
            inpainting_size,
            img_processor,
        )?);
        mask = resize_keep_aspect_mask(mask, inpainting_size, img_processor)?;
    }
    Ok((image, mask))
}

pub fn lama_add_border(
    mut image: RawImage,
    mut mask: Mask,
    img_processor: &Arc<dyn ImageOp + Send + Sync>,
) -> (RawImage, Mask, u16, u16) {
    let w = image.width;
    let h = image.height;
    let pad_size = 8;
    let new_h = if h % pad_size != 0 {
        (pad_size - (h % pad_size)) + h
    } else {
        h
    };
    let new_w = if w % pad_size != 0 {
        (pad_size - (w % pad_size)) + w
    } else {
        w
    };

    if new_h != h || new_w != w {
        let temp = img_processor.add_border_wh(image.view(), new_w, new_h);
        if let RawImageCow::Owned(o) = temp {
            image = o;
        }

        let mut m = RawImage {
            data: mask.data,
            width: mask.width,
            height: mask.height,
            channels: 1,
        };
        if let RawImageCow::Owned(o) = img_processor.add_border_wh(m.view(), new_w, new_h) {
            m = o;
        }
        mask = Mask {
            data: m.data,
            width: m.width,
            height: m.height,
        };
    }
    (image, mask, new_w, new_h)
}
