use std::collections::HashMap;

use base_util::error::ModelLoadError;
use interface_detector::{textlines::Quadrilateral, Detector};
use interface_image::{ImageOp, Mask, RawImage};
use interface_model::{Model, ModelSource};

pub struct PythonDetector {}

impl Model for PythonDetector {
    fn name(&self) -> &'static str {
        "python-module"
    }

    fn kind(&self) -> &'static str {
        "detector"
    }

    fn models(&self) -> std::collections::HashMap<&'static str, ModelSource> {
        HashMap::new()
    }

    fn loaded(&self) -> bool {
        todo!()
    }

    fn unload(&mut self) {
        todo!()
    }

    fn load(&mut self) -> Result<(), ModelLoadError> {
        todo!()
    }
}

impl Detector for PythonDetector {
    fn infer(
        &mut self,
        img: RawImage,
        options: &[u8],
        img_processor: &Box<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<(Vec<Quadrilateral>, Mask)> {
        todo!()
    }
}
