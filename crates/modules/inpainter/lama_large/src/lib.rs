use std::{ops::Deref, sync::Arc};

use base_util::onnx::{new_session, Providers};
use interface_image::{ImageOp, RawImageCow};
use interface_inpainter::{Inpainter, InpainterOptions};
use interface_model::{impl_model_load_helpers, Model, ModelLoad, ModelSource};
use maplit::hashmap;
use ndarray::{ArrayView4, Axis};
use ort::{inputs, session::Session, value::Tensor};
use util::lama::{lama_add_border, lama_resize_image};

pub struct LamaLargeInpainter {
    model: Option<Session>,
    providers: Arc<Vec<Providers>>,
}

impl LamaLargeInpainter {
    pub fn new(providers: Arc<Vec<Providers>>) -> Self {
        Self {
            model: None,
            providers,
        }
    }
}

impl ModelLoad for LamaLargeInpainter {
    type T = Session;

    fn loaded(&self) -> bool {
        self.model.is_some()
    }

    fn get_model(&mut self) -> Option<&mut Self::T> {
        self.model.as_mut()
    }

    fn reload(&mut self) -> anyhow::Result<&mut Self::T> {
        self.model = Some(new_session(
            self.download_model("model", "model.onnx")?,
            &self.providers,
        )?);
        Ok(self.model.as_mut().unwrap())
    }
}
impl Model for LamaLargeInpainter {
    impl_model_load_helpers!("inpainter", "lama_large");

    fn models(&self) -> std::collections::HashMap<&'static str, interface_model::ModelSource> {
        hashmap! {"model" => ModelSource{ url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/lama_large_512px/model.onnx", hash: "###" }}
    }

    fn unload(&mut self) {
        self.model = None;
    }
}

impl Inpainter for LamaLargeInpainter {
    fn inpaint(
        &mut self,
        image: &Arc<interface_image::RawImage>,
        mask: interface_image::Mask,
        options: InpainterOptions,
        img_processor: &Arc<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<interface_image::RawImage> {
        let ho = image.height;
        let wo = image.width;
        let (image, mask) =
            lama_resize_image(image.view(), mask, options.inpainting_size, img_processor)?;
        let mut image = image.to_owned();
        let h = image.height;
        let w = image.width;
        image = interface_inpainter::remove_mask_area(image, &mask);

        let (image, mask, new_w, new_h) = lama_add_border(image, mask, img_processor);
        let mask = mask
            .as_nd()?
            .mapv(|v| if v >= 127 { 1.0f32 } else { 0.0f32 })
            .insert_axis(Axis(0))
            .insert_axis(Axis(0));
        let image = image
            .as_ndarray()
            .unwrap()
            .permuted_axes((2, 0, 1))
            .mapv(|v| v as f32 / 255.0)
            .insert_axis(Axis(0));
        let image = Tensor::from_array(image)?;
        let mask = Tensor::from_array(mask)?;
        let model = self.load()?;
        let out = model.run(inputs! {"image"=> image, "mask"=> mask})?;
        let out: ArrayView4<f32> = out[0].try_extract_array()?.into_dimensionality()?;
        let img_inpainted = out
            .remove_axis(Axis(0))
            .permuted_axes((1, 2, 0))
            .mapv(|v| (v * 255.0) as u8);
        let mut img_inpainted = RawImageCow::from(img_inpainted.view());
        if new_h != h || new_w != w {
            img_inpainted =
                RawImageCow::Owned(img_processor.remove_border(img_inpainted.view(), w, h));
        }
        if h != ho || w != wo {
            img_inpainted = RawImageCow::Owned(img_processor.resize(
                img_inpainted.view(),
                wo,
                ho,
                interface_image::Interpolation::Bicubic,
            )?);
        }
        Ok(img_inpainted.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use interface_image::{CpuImageProcessor, Mask, RawImage};
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_inpaint() {
        let img = RawImage::new("./imgs/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png")
            .expect("Failed to load image");
        let img = RawImage::from(img);
        let img_processor =
            Arc::new(CpuImageProcessor::default()) as Arc<dyn ImageOp + Send + Sync>;
        let mask: Array2<u8> = ndarray_npy::read_npy("mask.npy").unwrap();
        let mask = Mask::from(mask);
        let mut inp = LamaLargeInpainter::new(Default::default());
        let v = inp
            .inpaint(&Arc::new(img), mask, Default::default(), &img_processor)
            .unwrap();
        v.to_image().unwrap().save("inpainted.png").unwrap()
    }
}
