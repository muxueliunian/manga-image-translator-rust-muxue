use interface_image::{ImageOp, Mask, RawImage};
use interface_model::Model;

pub trait Inpainter: Model {
    type Options;

    /// Will inpaint into image. This will change the whole image. A cutout of the image still needs to happen afterwards
    fn inpaint(
        &mut self,
        image: RawImage,
        mask: Mask,
        options: Self::Options,
        img_processor: &Box<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<RawImage>;
}

pub fn remove_mask_area(mut image: RawImage, mask: &Mask) -> RawImage {
    colorize_mask_area(image, mask, [0, 0, 0])
}

pub fn colorize_mask_area(mut image: RawImage, mask: &Mask, color: [u8; 3]) -> RawImage {
    assert_eq!(mask.height, image.height, "Invalid mask height");
    assert_eq!(mask.width, image.width, "Invalid mask width");
    assert_eq!(mask.data.len() * 3, image.data.len(), "Invalid mask size");
    let (r, g, b) = (color[0], color[1], color[2]);
    unsafe {
        let mask_data = mask.data.as_ptr();
        let image_data = image.data.as_mut_ptr();

        for i in 0..mask.data.len() {
            if *mask_data.add(i) > 127 {
                let offset = i * 3;
                *image_data.add(offset) = r;
                *image_data.add(offset + 1) = g;
                *image_data.add(offset + 2) = b;
            }
        }
    }
    image
}
