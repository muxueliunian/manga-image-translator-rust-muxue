use std::fmt::Display;

use base_util::{
    error::PreProcessingError,
    onnx::{new_session, Providers},
};
use interface_image::{combine_patches_m, generate_patches, ImageOp, RawImage};
use interface_model::{impl_model_load_helpers, Model, ModelLoad, ModelSource};
use interface_upscaler::Upscaler;
use maplit::hashmap;
use ndarray::{stack, Array3, Array4, ArrayView, ArrayView4, ArrayViewD, Axis, Dimension};
use ort::{inputs, session::Session, value::Tensor};

pub struct Waifu2xUpscaler {
    model: Option<Session>,
    model_kind: Waifu2xModels,
    max_batch_size: usize,
    providers: Vec<Providers>,
}

impl Display for Waifu2xModels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Waifu2xModels::CuNetArt { noise } => format!(
                    "cunet-art-2x{}",
                    noise.map(|v| format!("-noise{v}")).unwrap_or_default()
                ),
                Waifu2xModels::SwinUnetArt { x4, noise } => format!(
                    "swin_unet-art-{}x{}",
                    if *x4 { "4" } else { "2" },
                    noise.map(|v| format!("-noise{v}")).unwrap_or_default()
                ),
                Waifu2xModels::SwinUnetArtScan { x4, noise } => format!(
                    "swin_unet-art_scan-{}x{}",
                    if *x4 { "4" } else { "2" },
                    noise.map(|v| format!("-noise{v}")).unwrap_or_default()
                ),
                Waifu2xModels::SwinUnetArtPhoto { x4, noise } => format!(
                    "swin_unet-photo-{}x{}",
                    if *x4 { "4" } else { "2" },
                    noise.map(|v| format!("-noise{v}")).unwrap_or_default()
                ),
            }
        )
    }
}

pub enum Waifu2xModels {
    CuNetArt { noise: Option<u8> },
    SwinUnetArt { x4: bool, noise: Option<u8> },
    SwinUnetArtScan { x4: bool, noise: Option<u8> },
    SwinUnetArtPhoto { x4: bool, noise: Option<u8> },
}

impl Waifu2xUpscaler {
    pub fn new(model: Waifu2xModels, max_batch_size: usize, providers: Vec<Providers>) -> Self {
        Self {
            model: None,
            model_kind: model,
            max_batch_size,
            providers,
        }
    }
}

impl ModelLoad for Waifu2xUpscaler {
    type T = Session;
    fn loaded(&self) -> bool {
        self.model.is_some()
    }

    fn reload(&mut self) -> Result<&mut Session, interface_model::ModelLoadError> {
        let model = self.model_kind.to_string();
        let path = self.download_model(&model, &format!("{model}.onnx"))?;
        let session = new_session(path, self.providers.clone())?;
        self.model = Some(session);
        Ok(self.model.as_mut().expect("Set model before"))
    }

    fn get_model(&mut self) -> Option<&mut Self::T> {
        self.model.as_mut()
    }
}

impl Model for Waifu2xUpscaler {
    impl_model_load_helpers!("upscaler", "waifu2x");
    fn models(&self) -> std::collections::HashMap<&'static str, interface_model::ModelSource> {
        hashmap! {
            "cunet-art-2x-noise0" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/cunet-art-2x-noise0.onnx", hash: "###" },
            "cunet-art-2x-noise1" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/cunet-art-2x-noise1.onnx", hash: "###" },
            "cunet-art-2x-noise2" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/cunet-art-2x-noise2.onnx", hash: "###" },
            "cunet-art-2x-noise3" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/cunet-art-2x-noise3.onnx", hash: "###" },
            "cunet-art-2x" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/cunet-art-2x.onnx", hash: "###" },
            "swin_unet-art-2x-noise0" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art-2x-noise0.onnx", hash: "###" },
            "swin_unet-art-2x-noise1" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art-2x-noise1.onnx", hash: "###" },
            "swin_unet-art-2x-noise2" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art-2x-noise2.onnx", hash: "###" },
            "swin_unet-art-2x-noise3" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art-2x-noise3.onnx", hash: "###" },
            "swin_unet-art-2x" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art-2x.onnx", hash: "###" },
            "swin_unet-art-4x-noise0" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art-4x-noise0.onnx", hash: "###" },
            "swin_unet-art-4x-noise1" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art-4x-noise1.onnx", hash: "###" },
            "swin_unet-art-4x-noise2" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art-4x-noise2.onnx", hash: "###" },
            "swin_unet-art-4x-noise3" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art-4x-noise3.onnx", hash: "###" },
            "swin_unet-art-4x" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art-4x.onnx", hash: "###" },
            "swin_unet-art_scan-2x-noise0" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art_scan-2x-noise0.onnx", hash: "###" },
            "swin_unet-art_scan-2x-noise1" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art_scan-2x-noise1.onnx", hash: "###" },
            "swin_unet-art_scan-2x-noise2" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art_scan-2x-noise2.onnx", hash: "###" },
            "swin_unet-art_scan-2x-noise3" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art_scan-2x-noise3.onnx", hash: "###" },
            "swin_unet-art_scan-2x" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art_scan-2x.onnx", hash: "###" },
            "swin_unet-art_scan-4x-noise0" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art_scan-4x-noise0.onnx", hash: "###" },
            "swin_unet-art_scan-4x-noise1" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art_scan-4x-noise1.onnx", hash: "###" },
            "swin_unet-art_scan-4x-noise2" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art_scan-4x-noise2.onnx", hash: "###" },
            "swin_unet-art_scan-4x-noise3" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art_scan-4x-noise3.onnx", hash: "###" },
            "swin_unet-art_scan-4x" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-art_scan-4x.onnx", hash: "###" },
            "swin_unet-photo-2x-noise0" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-photo-2x-noise0.onnx", hash: "###" },
            "swin_unet-photo-2x-noise1" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-photo-2x-noise1.onnx", hash: "###" },
            "swin_unet-photo-2x-noise2" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-photo-2x-noise2.onnx", hash: "###" },
            "swin_unet-photo-2x-noise3" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-photo-2x-noise3.onnx", hash: "###" },
            "swin_unet-photo-2x" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-photo-2x.onnx", hash: "###" },
            "swin_unet-photo-4x-noise0" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-photo-4x-noise0.onnx", hash: "###" },
            "swin_unet-photo-4x-noise1" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-photo-4x-noise1.onnx", hash: "###" },
            "swin_unet-photo-4x-noise2" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-photo-4x-noise2.onnx", hash: "###" },
            "swin_unet-photo-4x-noise3" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-photo-4x-noise3.onnx", hash: "###" },
            "swin_unet-photo-4x" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/waifu2x-20250502/swin_unet-photo-4x.onnx", hash: "###" },
        }
    }

    fn unload(&mut self) {
        self.model = None;
    }
}

#[cfg(test)]
mod tests {
    use base_util::onnx::all_providers;
    use interface_image::{CpuImageProcessor, ImageOp, RawImage};
    use interface_upscaler::Upscaler as _;

    use crate::{Waifu2xModels, Waifu2xUpscaler};

    #[test]
    fn test_upscaler() {
        let mut upscaler = Waifu2xUpscaler::new(
            Waifu2xModels::CuNetArt { noise: Some(3) },
            5,
            all_providers(),
        );
        let image = RawImage::url(
            "https://github.com/xinntao/Real-ESRGAN/blob/master/inputs/0014.jpg?raw=true",
        )
        .expect("Failed to load image");
        let w = image.width;
        let img_processor = Box::new(CpuImageProcessor::default());
        let upscaled = upscaler
            .upscale(
                &image,
                None,
                0,
                &(img_processor as Box<dyn ImageOp + Send + Sync>),
            )
            .expect("Failed to upscale image");
        upscaled
            .clone()
            .to_image()
            .unwrap()
            .save("upscaled.png")
            .unwrap();
        assert_eq!(upscaled.width, w * 2);
        assert_eq!(upscaled.height, w * 2);
    }

    #[test]
    fn test_upscaler_patches() {
        let mut upscaler = Waifu2xUpscaler::new(
            Waifu2xModels::CuNetArt { noise: Some(3) },
            5,
            all_providers(),
        );
        let image = RawImage::url(
            "https://github.com/xinntao/Real-ESRGAN/blob/master/inputs/0014.jpg?raw=true",
        )
        .expect("Failed to load image");
        let w = image.width;
        let img_processor = Box::new(CpuImageProcessor::default());
        let upscaled = upscaler
            .upscale(
                &image,
                Some(100),
                0,
                &(img_processor as Box<dyn ImageOp + Send + Sync>),
            )
            .expect("Failed to upscale image");
        assert_eq!(upscaled.width, w * 2);
        assert_eq!(upscaled.height, w * 2);
    }
}

pub fn join_batches<A: Clone, Dim: Dimension>(
    patches: Vec<ArrayView<'_, A, Dim>>,
    max: usize,
) -> Result<
    Vec<ndarray::ArrayBase<ndarray::OwnedRepr<A>, <Dim as Dimension>::Larger>>,
    ndarray::ShapeError,
> {
    let v = patches
        .chunks(max)
        .map(|v| stack(Axis(0), v))
        .collect::<Result<Vec<_>, ndarray::ShapeError>>()?;
    Ok(v)
}

fn pre_process(
    image: &RawImage,
    patch_size: Option<usize>,
    padding: usize,
    max_batch_size: usize,
    img_processor: &Box<dyn ImageOp + Send + Sync>,
) -> Result<Vec<Array4<f32>>, PreProcessingError> {
    let pad_x = (8 - image.width % 8) % 8;
    let pad_y = (8 - image.height % 8) % 8;
    let w = image.width;
    let h = image.height;
    let imgs = match patch_size {
        Some(v) => generate_patches(image.clone(), v, 18 + padding),
        None => vec![img_processor.add_border_center_wh(
            img_processor.add_border_wh(image.clone(), w + pad_x, h + pad_y),
            w + pad_x + 18 * 2,
            h + pad_y + 18 * 2,
        )],
    };

    let patches = imgs
        .into_iter()
        .map(|v| {
            v.to_ndarray()
                .unwrap()
                .mapv(|v| v as f32 / 255.0)
                .permuted_axes((2, 0, 1))
        })
        .collect::<Vec<_>>();
    Ok(join_batches(
        patches.iter().map(|v| v.view()).collect(),
        max_batch_size,
    )?)
}

fn process(
    model: &mut Session,
    batches: Vec<Array4<f32>>,
) -> Result<Vec<Array3<u8>>, base_util::error::ProcessingError> {
    let mut processed_patches = vec![];
    for batch in batches {
        let t = Tensor::from_array(batch)?;
        let out = model.run(inputs! {"x"=>t})?;
        let img: ArrayViewD<f32> = out[0].try_extract_array()?;
        let img: ArrayView4<f32> = img.into_dimensionality()?;
        for img in img.outer_iter() {
            let img = img.permuted_axes((1, 2, 0)).mapv(|v| (v * 255.0) as u8);
            processed_patches.push(img);
        }
    }
    Ok(processed_patches)
}

impl Upscaler for Waifu2xUpscaler {
    fn upscale(
        &mut self,
        image: &RawImage,
        patch_size: Option<usize>,
        padding: usize,
        img_processor: &Box<dyn ImageOp + Send + Sync>,
    ) -> Result<RawImage, base_util::error::Error> {
        let max_batch_size = self.max_batch_size;
        let w = image.width;
        let h = image.height;

        let model = self.load()?;
        let batches = pre_process(image, patch_size, padding, max_batch_size, img_processor)?;
        let mut patches = process(model, batches)?
            .into_iter()
            .map(RawImage::from)
            .collect::<Vec<_>>();
        let ps = patches[0].width;
        let out = match patch_size {
            Some(_) => combine_patches_m(patches, w * 2, h * 2, ps as usize, padding * 2),
            None => img_processor.remove_border(patches.remove(0), w * 2, h * 2),
        };

        Ok(out)
    }
}
