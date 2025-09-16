use std::{cmp::Ordering, path::PathBuf, sync::Arc};

use base_util::onnx::{new_session, Providers};
use image::{DynamicImage, GenericImageView, RgbImage};
use interface_detector::textlines::Quadrilateral;
use interface_image::{ImageOp, Mask};
use interface_model::{impl_model_load_helpers, Model, ModelLoad, ModelSource};
use interface_ocr::QuadrilateralInfo;
use maplit::hashmap;
use ndarray::{s, stack, Array4, ArrayView2, Axis};
use ort::{
    inputs,
    session::{RunOptions, Session},
    value::Tensor,
};
use parking_lot::Mutex;
use tokio::task::spawn_blocking;

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
        providers: &[Providers],
    ) -> anyhow::Result<Self> {
        let enc = new_session(enc, providers)?;
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
    providers: Arc<Vec<Providers>>,
    max_length: usize,
}

impl MangaOCR {
    pub fn new(providers: Arc<Vec<Providers>>, max_length: usize) -> Self {
        Self {
            models: None,
            providers,
            max_length,
        }
    }
}

impl ModelLoad for MangaOCR {
    type T = MangaOCRModels;
    fn loaded(&self) -> bool {
        self.models.is_some()
    }

    fn reload(&mut self) -> anyhow::Result<&mut Self::T> {
        let enc = self.download_model("enc", "encoder_model.onnx")?;
        let dec = self.download_model("dec", "decoder_model.onnx")?;
        let voc = self.download_model("vocab", "vocab.txt")?;
        self.models = Some(MangaOCRModels::new(enc, dec, voc, &self.providers)?);
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

impl MangaOCR {
    async fn detect_patch(
        &mut self,
        image: Mask,
        area: Arc<Mutex<Quadrilateral>>,
        img_processor: &Arc<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<QuadrilateralInfo> {
        self.load()?;
        let sos_idnex = 2;
        let eos_index = 3;
        let special_tokens = 5;
        // allow:clone[arc]
        let pre = preprocessor(image, img_processor.clone()).await?;

        let t = Tensor::from_array(pre)?;
        let models = self.models.as_mut().expect("loaded");
        let run_options = RunOptions::new()?;

        let out = models
            .enc
            .run_async(inputs! {"pixel_values" => t}, &run_options)?
            .await?;
        let hs = &out[0];

        let mut token_ids: Vec<i64> = vec![sos_idnex];
        for _ in 0..self.max_length {
            let input = ArrayView2::from_shape((1, token_ids.len()), &token_ids)?;
            let t = Tensor::from_array(input.to_owned())?;

            let out = models
                .dec
                .run_async(
                    inputs! {
                        "encoder_hidden_states" => hs,
                        "input_ids" => t,
                    },
                    &run_options,
                )?
                .await?;
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
            // allow:clone[arc]
            pos: area.clone(),
            prob: 1.0,
        })
    }
}
#[async_trait::async_trait]
impl interface_ocr::Ocr for MangaOCR {
    async fn detect(
        &mut self,
        image: &Arc<interface_image::RawImage>,
        areas: &[Arc<Mutex<Quadrilateral>>],
        img_processor: &Arc<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<Vec<interface_ocr::QuadrilateralInfo>> {
        let mut texts = vec![];
        // allow:clone[arc]
        let image = image.clone();
        let grayscale = spawn_blocking(move || {
            Arc::new(
                DynamicImage::from(
                    RgbImage::from_raw(image.width as u32, image.height as u32, image.data.clone())
                        .unwrap(),
                )
                .to_luma8(),
            )
        })
        .await?;
        for (_, area) in areas.iter().enumerate() {
            let bbox = area.lock().aabb();
            // allow:clone[arc]
            let grayscale = grayscale.clone();
            let img = tokio::task::spawn_blocking(move || {
                let view =
                    grayscale.view(bbox.x as u32, bbox.y as u32, bbox.w as u32, bbox.h as u32);
                Mask::from(view.to_image())
            })
            .await?;

            // allow:clone[arc]
            texts.push(self.detect_patch(img, area.clone(), img_processor).await?);
        }

        Ok(texts)
    }
}

async fn preprocessor(
    img: Mask,
    img_processor: Arc<dyn ImageOp + Send + Sync>,
) -> anyhow::Result<Array4<f32>> {
    //"resample": 2,"size": 224
    spawn_blocking(move || {
        let resized = img_processor.resize_mask(
            img.view(),
            224,
            224,
            interface_image::Interpolation::Bilinear,
        )?;
        let img = resized
            .as_nd()?
            .mapv(|pixel| pixel as f32 / 255.0 * 2.0 - 1.0);
        Ok(stack(Axis(0), &[img.view(), img.view(), img.view()])?.insert_axis(Axis(0)))
    })
    .await
    .unwrap()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use base_util::onnx::all_providers;
    use interface_detector::textlines::Quadrilateral;
    use interface_image::{CpuImageProcessor, ImageOp, RawImage};
    use interface_ocr::Ocr as _;
    use parking_lot::Mutex;

    use crate::MangaOCR;

    #[tokio::test]
    async fn ocr_test() {
        let img = RawImage::new("./imgs/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png")
            .expect("Failed to load image");
        let mut mocr = MangaOCR::new(Arc::new(all_providers()), 255);
        let inp = vec![
            Arc::new(Mutex::new(Quadrilateral::new(
                vec![(208, 4), (246, 4), (246, 192), (208, 192)],
                1.0,
            ))),
            Arc::new(Mutex::new(Quadrilateral::new(
                vec![(76, 1788), (128, 1788), (128, 1930), (76, 1930)],
                1.0,
            ))),
        ];
        let ip = Arc::new(CpuImageProcessor::default()) as Arc<dyn ImageOp + Send + Sync>;
        let mut v = mocr.detect(&Arc::new(img), &inp, &ip).await.unwrap();
        v.sort_by_key(|a| a.text.len());
        assert_eq!(v[0].pos.lock().pts()[0].x, 76);
        assert_eq!(v[0].text, "ふふっ、");
        assert_eq!(v[1].text, "そうだなあ・・・");
        assert_eq!(v.len(), 2);
    }
}
