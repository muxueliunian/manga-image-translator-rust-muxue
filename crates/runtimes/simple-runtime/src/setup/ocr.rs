use std::{collections::HashMap, sync::Arc};

use base_util::onnx::all_providers;
use strum::IntoEnumIterator;

use crate::settings::OCR;

pub type OcrType = Box<dyn interface_ocr::Ocr + Send + Sync>;

pub struct OCRs(HashMap<OCR, OcrType>);
impl OCRs {
    pub fn get(&mut self, ocr: OCR) -> &mut OcrType {
        self.0.get_mut(&ocr).expect("Upscaler not registered")
    }
    pub fn new() -> Self {
        let mut items = HashMap::new();
        let providers = Arc::new(all_providers());
        for key in OCR::iter() {
            let ocr = match key {
                OCR::MangaOcr => {
                    // allow:clone
                    Box::new(manga_ocr::MangaOCR::new(providers.clone(), 256)) as OcrType
                }
                OCR::Native => Box::new(native::NativeOCR::default()) as OcrType,
                OCR::Tesseract => Box::new(tesseract::TesseractOCR::default()) as OcrType,
                // allow:clone
                OCR::Ctc48px => Box::new(ctc_48px::Ctc48pxOcr::new(providers.clone())) as OcrType,
            };
            items.insert(key, ocr);
        }
        OCRs(items)
    }
}
