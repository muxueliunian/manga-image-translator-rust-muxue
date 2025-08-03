mod mpe;
use base_util::onnx::{new_session, Providers};
use interface_image::{ImageOp, RawImage};
use interface_inpainter::Inpainter;
use interface_model::{impl_model_load_helpers, Model, ModelLoad, ModelSource};
use maplit::hashmap;
use ndarray::{ArrayView4, Axis};
use ort::{inputs, session::Session, value::Tensor};
use util::lama::{lama_add_border, lama_resize_image};

pub struct LamaLargeInpainter {
    model: Option<Session>,
    providers: Vec<Providers>,
}

impl LamaLargeInpainter {
    pub fn new(providers: Vec<Providers>) -> Self {
        Self {
            model: None,
            providers,
        }
    }
}

pub struct InpainterOptions {
    inpainting_size: u16,
}

impl Default for InpainterOptions {
    fn default() -> Self {
        Self {
            inpainting_size: 2048,
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

    fn reload(&mut self) -> Result<&mut Self::T, interface_model::ModelLoadError> {
        self.model = Some(new_session(
            self.download_model("model", "model.onnx")?,
            self.providers.clone(),
        )?);
        Ok(self.model.as_mut().unwrap())
    }
}
impl Model for LamaLargeInpainter {
    impl_model_load_helpers!("inpainter", "lama_mpe");

    fn models(&self) -> std::collections::HashMap<&'static str, interface_model::ModelSource> {
        hashmap! {"model" => ModelSource{ url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/lama_mpe/model.onnx", hash: "###" }}
    }

    fn unload(&mut self) {
        self.model = None;
    }
}

impl Inpainter for LamaLargeInpainter {
    type Options = InpainterOptions;

    fn inpaint(
        &mut self,
        image: interface_image::RawImage,
        mask: interface_image::Mask,
        options: Self::Options,
        img_processor: &Box<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<interface_image::RawImage> {
        let ho = image.height;
        let wo = image.width;
        let (mut image, mask) =
            lama_resize_image(image, mask, options.inpainting_size, img_processor);
        let h = image.height;
        let w = image.width;
        image = interface_inpainter::remove_mask_area(image, &mask);

        let (image, mask, new_w, new_h) = lama_add_border(image, mask, img_processor);
        let (rel_pos, direct) = mpe::load_masked_position_encoding(mask.clone(), img_processor);
        let mask = mask
            .as_nd()
            .mapv(|v| if v >= 127 { 1.0f32 } else { 0.0f32 })
            .insert_axis(Axis(0))
            .insert_axis(Axis(0));
        let image = image
            .to_ndarray()
            .unwrap()
            .permuted_axes((2, 0, 1))
            .mapv(|v| v as f32 / 255.0)
            .insert_axis(Axis(0));
        let image = Tensor::from_array(image)?;
        let mask = Tensor::from_array(mask)?;
        let rel_pos = Tensor::from_array(rel_pos.insert_axis(Axis(0)))?;
        let direct = Tensor::from_array(direct.insert_axis(Axis(0)))?;

        let model = self.load()?;
        let out = model.run(
            inputs! {"image"=> image, "mask"=> mask, "rel_pos" => rel_pos, "direct" => direct},
        )?;
        let out: ArrayView4<f32> = out[0].try_extract_array()?.into_dimensionality()?;
        let img_inpainted = out
            .remove_axis(Axis(0))
            .permuted_axes((1, 2, 0))
            .mapv(|v| (v * 255.0) as u8);
        let mut img_inpainted = RawImage::from(img_inpainted);
        if new_h != h || new_w != w {
            img_inpainted = img_processor.remove_border(img_inpainted, w, h);
        }
        if h != ho || w != wo {
            img_inpainted = img_processor.resize(
                img_inpainted,
                wo,
                ho,
                interface_image::Interpolation::Bicubic,
            );
        }
        Ok(img_inpainted)
    }
}

#[cfg(test)]
mod tests {
    use interface_image::{CpuImageProcessor, Mask};
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
        let mut inp = LamaLargeInpainter::new(vec![]);
        let v = inp
            .inpaint(img, mask, Default::default(), &img_processor)
            .unwrap();
        v.to_image().unwrap().save("inpainted.png").unwrap()
    }
}
