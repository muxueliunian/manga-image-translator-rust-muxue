use std::{cmp::Ordering, path::PathBuf};

use base_util::{error::PreProcessingError, onnx::new_session};
use image::{DynamicImage, GenericImageView, RgbImage};
use interface_detector::textlines::Quadrilateral;
use interface_image::{ImageOp, Mask};
use interface_model::{
    impl_model_load_helpers, CreateData, Model, ModelLoad, ModelLoadError, ModelSource,
};
use interface_ocr::QuadrilateralInfo;
use maplit::hashmap;
use ndarray::{s, stack, Array2, Array4, Axis};
use ort::{inputs, session::Session, value::Tensor};

pub struct MangaOCRModels {
    enc: Session,
    dec: Session,
    vocab: Vec<String>,
}

impl MangaOCRModels {
    fn new(
        enc: PathBuf,
        dec: PathBuf,
        vocab: PathBuf,
        providers: Vec<base_util::onnx::Providers>,
    ) -> Result<Self, ModelLoadError> {
        let enc = new_session(enc, providers.clone())?;
        let dec = new_session(dec, providers)?;

        let vocab = std::fs::read_to_string(vocab)?
            .lines()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        Ok(Self { enc, dec, vocab })
    }
}

pub struct MangaOCR {
    models: Option<MangaOCRModels>,
    db: CreateData,
    max_length: usize,
}

impl MangaOCR {
    pub fn new(db: CreateData, max_length: usize) -> Self {
        Self {
            models: None,
            db,
            max_length,
        }
    }
}

impl ModelLoad for MangaOCR {
    type T = MangaOCRModels;
    fn loaded(&self) -> bool {
        self.models.is_some()
    }

    fn reload(&mut self) -> Result<&mut Self::T, interface_model::ModelLoadError> {
        let enc = self.download_model("enc", "encoder_model.onnx")?;
        let dec = self.download_model("dec", "decoder_model.onnx")?;
        let voc = self.download_model("vocab", "vocab.txt")?;
        self.models = Some(MangaOCRModels::new(
            enc,
            dec,
            voc,
            self.db.providers.clone(),
        )?);
        Ok(self.models.as_mut().unwrap())
    }

    fn get_model(&mut self) -> Option<&mut Self::T> {
        self.models.as_mut()
    }
}

impl Model for MangaOCR {
    impl_model_load_helpers!("ocr", "manga-ocr");

    fn models(&self) -> std::collections::HashMap<&'static str, interface_model::ModelSource> {
        hashmap! {
            "enc" => ModelSource { url: "https://huggingface.co/mayocream/manga-ocr-onnx/resolve/main/encoder_model.onnx?download=true", hash: "###" },
            "dec" => ModelSource { url: "https://huggingface.co/mayocream/manga-ocr-onnx/resolve/main/decoder_model.onnx?download=true", hash: "###" },
            "vocab" => ModelSource { url: "https://huggingface.co/mayocream/manga-ocr-onnx/resolve/main/vocab.txt?download=true", hash: "###" },
        }
    }

    fn unload(&mut self) {
        self.models = None;
    }
}

impl interface_ocr::Ocr for MangaOCR {
    fn detect(
        &mut self,
        image: &interface_image::RawImage,
        areas: &[Quadrilateral],
        img_processor: &Box<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<Vec<interface_ocr::QuadrilateralInfo>> {
        let mut texts = vec![];
        let grayscale = DynamicImage::from(
            RgbImage::from_raw(image.width as u32, image.height as u32, image.data.clone())
                .unwrap(),
        )
        .to_luma8();
        for area in areas {
            let bbox = area.aabb();
            let view = grayscale.view(bbox.x as u32, bbox.y as u32, bbox.w as u32, bbox.h as u32);
            let img = Mask::from(view.to_image());
            texts.push(self.detect_patch(img, area.clone(), img_processor)?);
        }

        Ok(texts)
    }

    fn detect_patch(
        &mut self,
        image: Mask,
        area: Quadrilateral,
        img_processor: &Box<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<QuadrilateralInfo> {
        self.load()?;
        let sos_idnex = 2;
        let eos_index = 3;
        let special_tokens = 5;
        let pre = preprocessor(image, img_processor)?;

        let t = Tensor::from_array(pre)?;
        let models = self.models.as_mut().expect("loaded");

        let out = models.enc.run(inputs! {"pixel_values" => t})?;
        let hs = &out[0];

        let mut token_ids: Vec<i64> = vec![sos_idnex];
        for _ in 0..self.max_length {
            let input = Array2::from_shape_vec((1, token_ids.len()), token_ids.clone())?;
            let t = Tensor::from_array(input)?;

            let out = models.dec.run(inputs! {
                "encoder_hidden_states" => hs,
                "input_ids" => t,
            })?;
            let logits = out["logits"].try_extract_array::<f32>()?;
            let v = logits.slice(s![0, -1, ..]);
            let token_id = v
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|v| v.0)
                .unwrap_or_default();

            token_ids.push(token_id as i64);
            if token_id as i64 == eos_index {
                break;
            }
        }
        Ok(QuadrilateralInfo {
            text: token_ids
                .into_iter()
                .filter(|&id| id >= special_tokens)
                .map(|v| models.vocab[v as usize].to_owned())
                .collect::<String>(),
            fg: None,
            bg: None,
            pos: area.clone(),
        })
    }
}

fn preprocessor(
    img: Mask,
    img_processor: &Box<dyn ImageOp + Send + Sync>,
) -> Result<Array4<f32>, PreProcessingError> {
    //"resample": 2,"size": 224
    let resized =
        img_processor.resize_mask(img, 224, 224, interface_image::Interpolation::Bilinear);
    let img = resized
        .as_nd()
        .mapv(|pixel| pixel as f32 / 255.0 * 2.0 - 1.0);
    Ok(stack(Axis(0), &[img.view(), img.view(), img.view()])?.insert_axis(Axis(0)))
}

#[cfg(test)]
mod tests {
    use interface_detector::textlines::Quadrilateral;
    use interface_image::{CpuImageProcessor, ImageOp, RawImage};
    use interface_model::CreateData;
    use interface_ocr::Ocr as _;

    use crate::MangaOCR;

    #[test]
    fn ocr_test() {
        let img = RawImage::new("./imgs/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png")
            .expect("Failed to load image");
        let mut mocr = MangaOCR::new(CreateData::all(), 255);
        let inp = vec![
            Quadrilateral::new(vec![(208, 4), (246, 4), (246, 192), (208, 192)], 1.0),
            Quadrilateral::new(vec![(76, 1788), (128, 1788), (128, 1930), (76, 1930)], 1.0),
        ];
        let ip = Box::new(CpuImageProcessor::default()) as Box<dyn ImageOp + Send + Sync>;
        let v = mocr.detect(&img, &inp, &ip).unwrap();
        assert_eq!(v[0].text, "そうだなあ・・・");
        assert_eq!(v.len(), 2);
    }
}
