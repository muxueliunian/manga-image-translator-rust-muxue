use std::{collections::HashMap, sync::Arc};

use image::{DynamicImage, GenericImageView as _, RgbImage};
use interface_detector::textlines::Quadrilateral;
use interface_image::{ImageOp, Mask, RawImage};
use interface_model::{impl_model_load_helpers, Model, ModelLoad};
use interface_ocr::{Ocr, QuadrilateralInfo};
use parking_lot::Mutex;
use tokio::task::spawn_blocking;
use uni_ocr::{Language, OcrEngine, OcrOptions, OcrProvider};

#[derive(Default)]
pub struct NativeOCR {
    model: Option<OcrEngine>,
}

impl NativeOCR {}

impl ModelLoad for NativeOCR {
    type T = OcrEngine;

    fn loaded(&self) -> bool {
        self.model.is_some()
    }

    fn get_model(&mut self) -> Option<&mut Self::T> {
        self.model.as_mut()
    }

    fn reload(&mut self) -> anyhow::Result<&mut Self::T> {
        let engine = OcrEngine::new(OcrProvider::Auto).unwrap().with_options(
            OcrOptions::default().languages(vec![
                Language::Chinese,
                Language::Japanese,
                Language::Korean,
                Language::English,
            ]),
        );
        self.model = Some(engine);
        Ok(self.model.as_mut().unwrap())
    }
}

impl Model for NativeOCR {
    impl_model_load_helpers!("ocr", "native");

    fn models(&self) -> std::collections::HashMap<&'static str, interface_model::ModelSource> {
        HashMap::new()
    }

    fn unload(&mut self) {
        self.model = None;
    }
}

#[async_trait::async_trait]
impl Ocr for NativeOCR {
    async fn detect(
        &mut self,
        image: &Arc<RawImage>,
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

        for area in areas {
            let bbox = area.lock().aabb();
            // allow:clone[arc]
            let grayscale = grayscale.clone();
            let img = spawn_blocking(move || {
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

impl NativeOCR {
    async fn detect_patch(
        &mut self,
        sliced_image: interface_image::Mask,
        area: Arc<Mutex<Quadrilateral>>,
        _: &Arc<dyn interface_image::ImageOp + Send + Sync>,
    ) -> anyhow::Result<interface_ocr::QuadrilateralInfo> {
        let model = self.load()?;
        let image =
            spawn_blocking(move || image::DynamicImage::from(sliced_image.to_image().unwrap()))
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

    use crate::NativeOCR;

    #[tokio::test]
    async fn ocr_test() {
        let img = RawImage::new("./imgs/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png")
            .expect("Failed to load image");
        let mut mocr = NativeOCR::default();
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
        let v = mocr.detect(&Arc::new(img), &inp, &ip).await.unwrap();
        assert_eq!(v[0].text, "そうだなあ・・・");
        assert_eq!(v.len(), 2);
    }
}
