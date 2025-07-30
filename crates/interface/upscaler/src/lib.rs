use interface_image::{ImageOp, RawImage};

pub trait Upscaler {
    fn upscale(
        &mut self,
        image: &RawImage,
        patch_size: Option<usize>,
        padding: usize,
        img_processor: &Box<dyn ImageOp + Send + Sync>,
    ) -> Result<RawImage, base_util::error::Error>;
}
