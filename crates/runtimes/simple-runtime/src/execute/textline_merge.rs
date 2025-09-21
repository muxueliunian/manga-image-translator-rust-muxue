use interface_ocr::QuadrilateralInfo;
use log::info;
use textline_merge::TextBlock;

use crate::{
    settings::{OCRSettings, TranslatorSettings},
    setup::Models,
};

impl Models {
    pub fn run_textline_merge(
        &self,
        textlines: &[QuadrilateralInfo],
        width: u16,
        height: u16,
        config: &OCRSettings,
        config2: &TranslatorSettings,
    ) -> anyhow::Result<Vec<TextBlock>> {
        assert!(!textlines.is_empty());
        info!("Run Textline Merge");
        textline_merge::dispatch_main(
            textlines,
            width,
            height,
            config.post_processing.min_text_length,
            config.post_processing.prob,
            config2.filter_lang.iter().map(|v| v.0).collect(),
            &config.post_processing.filter_text,
            &self.lang_detector,
        )
    }
}
