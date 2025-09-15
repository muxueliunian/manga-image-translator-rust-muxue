use std::sync::Arc;

use base_util::onnx::all_providers;
use dbnet::DbNetDetector;
use interface_detector::DefaultOptions;
use interface_detector::Detector;
use interface_detector::PreprocessorOptions;
use interface_image::{CpuImageProcessor, ImageOp, RawImage};
use interface_model::ModelLoad;

fn main() {
    env_logger::init();
    let mut data = DbNetDetector::new(Arc::new(all_providers()), false);
    let cpu_image_processor =
        Arc::new(CpuImageProcessor::default()) as Arc<dyn ImageOp + Send + Sync>;
    data.load().expect("Failed to load data");
    let img = RawImage::new("./imgs/01_1-optimized.png").expect("Failed to load image");
    let (_, _) = data
        .detect(
            &img,
            PreprocessorOptions::default(),
            DefaultOptions::default(),
            &cpu_image_processor,
        )
        .expect("failed to detect");
}
