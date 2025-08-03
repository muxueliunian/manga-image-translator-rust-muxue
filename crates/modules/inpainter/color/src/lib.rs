use std::collections::HashMap;

use interface_inpainter::{colorize_mask_area, Inpainter};
use interface_model::Model;

pub struct ColorInpainter {
    color: [u8; 3],
    loaded: bool,
}

impl Default for ColorInpainter {
    fn default() -> Self {
        Self {
            color: [255, 255, 255],
            loaded: true,
        }
    }
}

impl ColorInpainter {
    pub fn new(color: [u8; 3]) -> Self {
        Self {
            color,
            loaded: true,
        }
    }
}

impl Model for ColorInpainter {
    fn name(&self) -> &'static str {
        "color"
    }

    fn kind(&self) -> &'static str {
        "inpainter"
    }

    fn models(&self) -> std::collections::HashMap<&'static str, interface_model::ModelSource> {
        HashMap::new()
    }

    fn unload(&mut self) {
        self.loaded = false;
    }

    fn loaded_(&self) -> bool {
        self.loaded
    }

    fn reload_(&mut self) -> Result<(), interface_model::ModelLoadError> {
        self.loaded = true;
        Ok(())
    }
}

impl Inpainter for ColorInpainter {
    type Options = ();

    fn inpaint(
        &mut self,
        image: interface_image::RawImage,
        mask: interface_image::Mask,
        _: Self::Options,
        _: &Box<dyn interface_image::ImageOp + Send + Sync>,
    ) -> anyhow::Result<interface_image::RawImage> {
        Ok(colorize_mask_area(image, &mask, self.color))
    }
}

#[cfg(test)]
mod tests {
    use interface_image::{CpuImageProcessor, ImageOp, Mask, RawImage};
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_inpaint() {
        let img = RawImage::new("./imgs/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png")
            .expect("Failed to load image");
        let img = RawImage::from(img);
        let img_processor =
            Box::new(CpuImageProcessor::default()) as Box<dyn ImageOp + Send + Sync>;
        let mask: Array2<u8> = ndarray_npy::read_npy("../lama_large/mask.npy").unwrap();
        let mask = Mask::from(mask);
        let mut inp = ColorInpainter::default();
        let v = inp
            .inpaint(img, mask, Default::default(), &img_processor)
            .unwrap();
        v.to_image().unwrap().save("inpainted.png").unwrap()
    }
}
