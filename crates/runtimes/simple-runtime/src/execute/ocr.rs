use std::{fs::create_dir_all, path::PathBuf, sync::Arc};

use interface_detector::textlines::Quadrilateral;
use interface_image::RawImage;
use interface_ocr::{OcrOptions, QuadrilateralInfo};
use log::info;
use parking_lot::Mutex;

use crate::{execute::ImageProcessor, settings::OCRSettings, setup::Models};

impl Models {
    pub async fn run_ocr(
        &mut self,
        img: &RawImage,
        areas: &[Arc<Mutex<Quadrilateral>>],
        config: &OCRSettings,
        debug_path: &Option<PathBuf>,
        ip: &ImageProcessor,
    ) -> anyhow::Result<Vec<QuadrilateralInfo>> {
        let debug_path = if let Some(debug_path) = debug_path {
            let p = debug_path.join("ocr_patches");
            create_dir_all(&p)?;
            Some(p)
        } else {
            None
        };
        info!("Run OCR: {:?}", config.ocr);
        let textlines = self
            .get_ocr(config.ocr)
            .detect(img, areas, OcrOptions { debug_path }, ip)
            .await?;
        Ok(textlines)
    }
}
