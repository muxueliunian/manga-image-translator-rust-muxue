use std::sync::Arc;

use interface_detector::textlines::Quadrilateral;
use interface_image::{ImageOp, Mask, RawImage};
use parking_lot::Mutex;

#[async_trait::async_trait]
pub trait Ocr {
    async fn detect(
        &mut self,
        image: &Arc<RawImage>,
        areas: &[Arc<Mutex<Quadrilateral>>],
        img_processor: &Arc<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<Vec<QuadrilateralInfo>>;

    /// image is already the sliced image
    async fn detect_patch(
        &mut self,
        sliced_image: Mask,
        area: Arc<Mutex<Quadrilateral>>,
        img_processor: &Arc<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<QuadrilateralInfo>;
}

#[derive(Debug, Clone)]
pub struct QuadrilateralInfo {
    pub text: String,
    pub fg: Option<[u8; 3]>,
    pub bg: Option<[u8; 3]>,
    pub pos: Arc<Mutex<Quadrilateral>>,
    pub prob: f64,
}
