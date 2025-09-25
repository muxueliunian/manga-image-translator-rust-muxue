use std::{collections::HashMap, sync::Arc};

use image::{DynamicImage, GenericImageView as _, RgbImage};
use interface_detector::textlines::Quadrilateral;
use interface_image::{ImageOp, Mask, RawImage};
use interface_model::{impl_model_load_helpers, Model, ModelLoad};
use interface_ocr::{Ocr, QuadrilateralInfo};
use parking_lot::Mutex;
use uni_ocr::{Language, OcrEngine, OcrOptions, OcrProvider};
use util::spawn_blocking;

#[derive(Default)]
pub struct TesseractOCR {
    model: Option<OcrEngine>,
}

impl TesseractOCR {}

impl ModelLoad for TesseractOCR {
    type T = OcrEngine;

    fn loaded(&self) -> bool {
        self.model.is_some()
    }

    fn get_model(&mut self) -> Option<&mut Self::T> {
        self.model.as_mut()
    }

    fn reload(&mut self) -> anyhow::Result<&mut Self::T> {
        let engine = OcrEngine::new(OcrProvider::Tesseract)
            .unwrap()
            .with_options(OcrOptions::default().languages(vec![
                Language::Chinese,
                Language::Japanese,
                Language::Korean,
                Language::English,
            ]));
        self.model = Some(engine);
        Ok(self.model.as_mut().unwrap())
    }
}

impl Model for TesseractOCR {
    impl_model_load_helpers!("ocr", "tesseract");

    fn models(&self) -> std::collections::HashMap<&'static str, interface_model::ModelSource> {
        HashMap::new()
    }

    fn unload(&mut self) {
        self.model = None;
    }
}

#[async_trait::async_trait]
impl Ocr for TesseractOCR {
    async fn detect(
        &mut self,
        image: &RawImage,
        areas: &[Arc<Mutex<Quadrilateral>>],
        options: interface_ocr::OcrOptions,
        img_processor: &Arc<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<Vec<interface_ocr::QuadrilateralInfo>> {
        let mut texts = vec![];

        let grayscale =
            spawn_blocking!(|| Ok::<_, anyhow::Error>(image.clone().to_image()?.to_luma8()))??;

        for (i, area) in areas.into_iter().enumerate() {
            let bbox = area.lock().aabb();
            let img = spawn_blocking!(|| {
                let view =
                    grayscale.view(bbox.x as u32, bbox.y as u32, bbox.w as u32, bbox.h as u32);
                Mask::from(view.to_image())
            })?;
            if let Some(v) = &options.debug_path {
                img.clone()
                    .to_image()?
                    .save(v.join(format!("patch_{i}_0.png")))?
            }

            // allow:clone[arc]
            texts.push(self.detect_patch(img, area.clone(), img_processor).await?);
        }

        Ok(texts)
    }
}

impl TesseractOCR {
    async fn detect_patch(
        &mut self,
        sliced_image: interface_image::Mask,
        area: Arc<Mutex<interface_detector::textlines::Quadrilateral>>,
        _: &Arc<dyn interface_image::ImageOp + Send + Sync>,
    ) -> anyhow::Result<interface_ocr::QuadrilateralInfo> {
        let model = self.load()?;
        let image = tokio::task::spawn_blocking(move || {
            image::DynamicImage::from(sliced_image.to_image().unwrap())
        })
        .await?;

        let (result, _, prob) = model.recognize_image(&image).await?;
        Ok(QuadrilateralInfo {
            text: result,
            fg: None,
            bg: None,
            pos: area,
            prob: prob.unwrap_or(1.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use interface_detector::textlines::Quadrilateral;
    use interface_image::{CpuImageProcessor, ImageOp, RawImage};
    use interface_ocr::Ocr as _;
    use parking_lot::Mutex;

    use crate::TesseractOCR;

    #[tokio::test]
    async fn ocr_test() {
        let img = RawImage::new("./imgs/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png")
            .expect("Failed to load image");
        let mut mocr = TesseractOCR::default();
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
        let v = mocr
            .detect(&Arc::new(img), &inp, Default::default(), &ip)
            .await
            .unwrap();
        assert_eq!(v[0].text, "そうだなあ・・・");
        assert_eq!(v.len(), 2);
    }
}
