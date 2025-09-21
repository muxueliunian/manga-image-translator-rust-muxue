use image::{DynamicImage, GrayImage};
use interface_image::RawImage;
use log::info;

use crate::{execute::ImageProcessor, settings::UpscalerSettings, setup::Models};

impl Models {
    pub fn run_upscaler(
        &mut self,
        mut rgb_img: RawImage,
        mut alpha: Option<Vec<u8>>,
        config: UpscalerSettings,
        ip: &ImageProcessor,
    ) -> anyhow::Result<(RawImage, Option<Vec<u8>>)> {
        if let Some(upscaler) = config.upscaler {
            info!("Run Upscaler: {:?}", upscaler);
            let (h, w) = (rgb_img.height, rgb_img.width);

            rgb_img = self.get_upscaler(upscaler).upscale(
                &rgb_img,
                config.patch_size,
                config.padding,
                &ip,
            )?;
            let (ha, wa) = (rgb_img.height, rgb_img.width);
            if let Some(a) = alpha {
                let alpha_image = DynamicImage::from(
                    GrayImage::from_raw(w as u32, h as u32, a)
                        .ok_or(anyhow::anyhow!("not a valid gray image"))?,
                );
                let alpha_image = alpha_image.resize_exact(
                    wa as u32,
                    ha as u32,
                    image::imageops::FilterType::Lanczos3,
                );
                let data = match alpha_image {
                    DynamicImage::ImageLuma8(img) => img.into_raw(),
                    _ => unreachable!("the output from upscaling should be a gray image"),
                };
                alpha = Some(data);
            }
        }
        Ok((rgb_img, alpha))
    }
}
