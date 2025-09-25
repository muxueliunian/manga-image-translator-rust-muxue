use std::{path::PathBuf, sync::Arc};

use interface_detector::textlines::Quadrilateral;
use interface_image::{ImageOp, RawImage};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

#[derive(Default)]
pub struct OcrOptions {
    pub debug_path: Option<PathBuf>,
}

#[async_trait::async_trait]
pub trait Ocr {
    async fn detect(
        &mut self,
        image: &RawImage,
        areas: &[Arc<Mutex<Quadrilateral>>],
        options: OcrOptions,
        img_processor: &Arc<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<Vec<QuadrilateralInfo>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadrilateralInfo {
    pub text: String,
    pub fg: Option<[u8; 3]>,
    pub bg: Option<[u8; 3]>,
    #[serde(with = "mutex_arc")]
    pub pos: Arc<Mutex<Quadrilateral>>,
    pub prob: f64,
}

mod mutex_arc {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(val: &Arc<Mutex<Quadrilateral>>, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        val.lock().serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Arc<Mutex<Quadrilateral>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner = Quadrilateral::deserialize(d)?;
        Ok(Arc::new(Mutex::new(inner)))
    }
}
