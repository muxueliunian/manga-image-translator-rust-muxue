use interface_image::RawImage;
use interface_ocr::QuadrilateralInfo;

pub trait Renderer {
    type Options;
    type Output;
    fn render(
        &self,
        image: RawImage,
        translations: QuadrilateralInfo,
        options: Self::Options,
    ) -> anyhow::Result<Self::Output>;
}
