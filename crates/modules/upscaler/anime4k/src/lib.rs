use std::{fmt::Display, sync::Arc};

use base_util::onnx::{new_session, Providers};
use half::f16;
use interface_image::RawImage;
use interface_model::{impl_model_load_helpers, Model, ModelLoad};
use interface_upscaler::Upscaler;
use maplit::hashmap;
use ndarray::{ArrayView3, ArrayViewD, Axis};
use ort::{inputs, session::Session, value::Tensor};

pub struct Anime4KUpscaler {
    model: Option<Session>,
    model_kind: Anime4KModel,
    providers: Arc<Vec<Providers>>,
}

impl Anime4KUpscaler {
    pub fn new(model_kind: Anime4KModel, providers: Arc<Vec<Providers>>) -> Self {
        Anime4KUpscaler {
            model: None,
            model_kind,
            providers,
        }
    }
}

pub enum Anime4KModel {
    X4UUL,
    X4UL,
    X3VL,
    X3L,
    X2M,
    X2S,
}

impl Display for Anime4KModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Anime4KModel::X4UUL => write!(f, "2x_UUL"),
            Anime4KModel::X4UL => write!(f, "4x_UL"),
            Anime4KModel::X3VL => write!(f, "3x_VL"),
            Anime4KModel::X3L => write!(f, "3x_L"),
            Anime4KModel::X2M => write!(f, "2x_M"),
            Anime4KModel::X2S => write!(f, "2x_S"),
        }
    }
}

impl ModelLoad for Anime4KUpscaler {
    type T = Session;
    fn loaded(&self) -> bool {
        self.model.is_some()
    }

    fn reload(&mut self) -> anyhow::Result<&mut Session> {
        let model = self.model_kind.to_string();
        let path = self.download_model(&model, &format!("{model}.onnx"))?;
        let session = new_session(path, &self.providers)?;
        self.model = Some(session);
        Ok(self.model.as_mut().expect("Set model before"))
    }

    fn get_model(&mut self) -> Option<&mut Self::T> {
        self.model.as_mut()
    }
}

impl Model for Anime4KUpscaler {
    impl_model_load_helpers!("upscaler", "waifu2x");

    fn models(&self) -> std::collections::HashMap<&'static str, interface_model::ModelSource> {
        hashmap! {
            "2x_S" => interface_model::ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/anime4k/2x_S", hash: "###" },
            "2x_M" => interface_model::ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/anime4k/2x_M", hash: "###" },
            "3x_L" => interface_model::ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/anime4k/3x_L", hash: "###" },
            "3x_VL" => interface_model::ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/anime4k/3x_VL", hash: "###" },
            "4x_UL" => interface_model::ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/anime4k/4x_UL", hash: "###" },
            "4x_UUL" =>interface_model::ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/anime4k/4x_UUL", hash: "###" }
        }
    }

    fn unload(&mut self) {
        self.model = None;
    }
}

impl Upscaler for Anime4KUpscaler {
    fn upscale(
        &mut self,
        image: &RawImage,
        _: Option<usize>,
        _: usize,
        _: &Arc<dyn interface_image::ImageOp + Send + Sync>,
    ) -> anyhow::Result<RawImage> {
        let image = image
            .as_ndarray()
            .unwrap()
            .mapv(|v| f16::from_f32(v as f32 / 255.0))
            .permuted_axes((2, 0, 1))
            .insert_axis(Axis(0));
        let t = Tensor::from_array(image).unwrap();
        let model = self.load()?;
        let out = model.run(inputs! {"input"=>t}).unwrap();
        let out: ArrayViewD<f16> = out[0].try_extract_array().unwrap().remove_axis(Axis(0));
        let out: ArrayView3<f16> = out.into_dimensionality().unwrap();
        let out = out
            .mapv(|v| (v.to_f32() * 255.0) as u8)
            .permuted_axes((1, 2, 0));
        let out = RawImage::from(out);
        Ok(out)
    }
}
