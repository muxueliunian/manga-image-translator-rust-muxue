use interface_detector::textlines::Quadrilateral;
use interface_image::RawImage;

pub trait Ocr {
    type Options;
    fn detect(
        &self,
        image: &RawImage,
        areas: &[Quadrilateral],
        options: Self::Options,
    ) -> anyhow::Result<Vec<QuadrilateralInfo>>;
}

pub struct QuadrilateralInfo {
    pub text: String,
    pub fg: Option<[u8; 3]>,
    pub bg: Option<[u8; 3]>,
    pub pos: Quadrilateral,
}
