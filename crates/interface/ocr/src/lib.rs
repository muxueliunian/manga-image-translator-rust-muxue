use std::sync::Arc;

use interface_detector::textlines::Quadrilateral;
use interface_image::{ImageOp, RawImage};
use parking_lot::Mutex;

#[async_trait::async_trait]
pub trait Ocr {
    async fn detect(
        &mut self,
        image: &Arc<RawImage>,
        areas: &[Arc<Mutex<Quadrilateral>>],
        img_processor: &Arc<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<Vec<QuadrilateralInfo>>;
}

#[derive(Debug, Clone)]
pub struct QuadrilateralInfo {
    pub text: String,
    pub fg: Option<[u8; 3]>,
    pub bg: Option<[u8; 3]>,
    pub pos: Arc<Mutex<Quadrilateral>>,
    pub prob: f64,
}
