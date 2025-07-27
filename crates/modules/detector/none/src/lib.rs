use std::{collections::HashMap, vec};

use base_util::error::ModelLoadError;
use interface_detector::{textlines::Quadrilateral, Detector};
use interface_image::{ImageOp, Mask, RawImage};
use interface_model::{CreateData, Model, ModelSource};

pub struct NoneDetector {
    loaded: bool,
}

impl NoneDetector {
    pub fn new(_: CreateData) -> Box<Self> {
        Box::new(NoneDetector { loaded: true })
    }
}

impl Model for NoneDetector {
    fn name(&self) -> &'static str {
        "none"
    }

    fn kind(&self) -> &'static str {
        "detector"
    }

    fn models(&self) -> std::collections::HashMap<&'static str, ModelSource> {
        HashMap::new()
    }

    fn loaded(&self) -> bool {
        self.loaded
    }

    fn unload(&mut self) {
        self.loaded = false;
    }

    fn load(&mut self) -> Result<(), ModelLoadError> {
        self.loaded = true;
        Ok(())
    }
}

impl Detector for NoneDetector {
    fn infer(
        &mut self,
        img: RawImage,
        _: &[u8],
        _: &Box<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<(Vec<Quadrilateral>, Mask)> {
        Ok((
            vec![],
            Mask {
                width: img.width,
                height: img.height,
                data: Vec::new(),
            },
        ))
    }
}

#[cfg(test)]
mod tests {

    use interface_detector::{Detector as _, PreprocessorOptions};
    use interface_image::{CpuImageProcessor, ImageOp, RawImage};
    use interface_model::{CreateData, Model as _};

    use crate::NoneDetector;

    #[test]
    fn load_unload() {
        let mut data = NoneDetector::new(CreateData::all());
        data.load().expect("failed to load model");
        data.unload();
    }

    #[test]
    fn run() {
        let mut data = NoneDetector::new(CreateData::all());
        let cpu_image_processor =
            Box::new(CpuImageProcessor::default()) as Box<dyn ImageOp + Send + Sync>;
        data.load().expect("Failed to load data");
        data.detect(
            &RawImage::new("./imgs/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.border.png")
                .expect("Failed to load image"),
            PreprocessorOptions::default(),
            &[],
            &cpu_image_processor,
        )
        .expect("failed to detect");
    }
}
