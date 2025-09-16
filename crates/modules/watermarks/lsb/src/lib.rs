use image::{DynamicImage, RgbaImage};
use stego::LSBStego;

pub trait Watermark<I, O> {
    fn custom_text(&self) -> Option<&String>;
    fn text(&self) -> String {
        let name = format!("manga-image-translator {}", env!("CARGO_PKG_VERSION"));
        let url = "https://github.com/frederik-uni/manga-image-translator-rust";
        let custom = self
            .custom_text()
            .map(|v| format!("\n{v}"))
            .unwrap_or_default();
        format!("{name}\n{url}{custom}")
    }
    fn apply_watermark(&mut self, img: I) -> Result<O, String>;
    fn read_watermark(&mut self, img: I) -> Result<String, String>;
}

pub struct LSBWatermark {
    text: Option<String>,
}

impl Watermark<DynamicImage, RgbaImage> for LSBWatermark {
    fn apply_watermark(&mut self, img: DynamicImage) -> Result<RgbaImage, String> {
        let mut stego = LSBStego::new(img);
        let img = stego.encode_text(self.text());
        Ok(img)
    }

    fn read_watermark(&mut self, img: DynamicImage) -> Result<String, String> {
        let mut stego = LSBStego::new(img);
        Ok(stego.decode_text())
    }

    fn custom_text(&self) -> Option<&String> {
        self.text.as_ref()
    }
}
