use std::{collections::HashMap, sync::Arc};

use base_util::onnx::all_providers;
use esrgan::EsrGanModel;
use strum::IntoEnumIterator;
use waifu2x::Waifu2xModels;

use crate::settings::Upscaler;
pub type UpscalerType = Box<dyn interface_upscaler::Upscaler + Send + Sync>;

pub struct Upscalers(HashMap<Upscaler, UpscalerType>);
impl Upscalers {
    pub fn get(&mut self, upscaler: Upscaler) -> &mut UpscalerType {
        self.0.get_mut(&upscaler).expect("Upscaler not registered")
    }
    pub fn new(max_batch_size: usize, fast: bool) -> Self {
        let mut items = HashMap::new();
        let providers = Arc::new(all_providers());
        //TODO: add more anime4k models
        for key in Upscaler::iter() {
            let upscaler = match key {
                Upscaler::Waifu2xCuNetArt(noise) => Box::new(waifu2x::Waifu2xUpscaler::new(
                    Waifu2xModels::CuNetArt { noise },
                    max_batch_size,
                    // allow:clone
                    providers.clone(),
                )) as UpscalerType,
                Upscaler::Waifu2xSwinUnetArt4x(noise) => Box::new(waifu2x::Waifu2xUpscaler::new(
                    Waifu2xModels::SwinUnetArt { x4: true, noise },
                    max_batch_size,
                    // allow:clone
                    providers.clone(),
                )) as UpscalerType,
                Upscaler::Waifu2xSwinUnetArt2x(noise) => Box::new(waifu2x::Waifu2xUpscaler::new(
                    Waifu2xModels::SwinUnetArt { x4: false, noise },
                    max_batch_size,
                    // allow:clone
                    providers.clone(),
                )) as UpscalerType,
                Upscaler::Anime4k => Box::new(anime4k::Anime4KUpscaler::new(
                    anime4k::Anime4KModel::X2S,
                    // allow:clone
                    providers.clone(),
                )),
                Upscaler::Esrgan2x => Box::new(esrgan::EsrGan::new(
                    EsrGanModel::X2Plus { f32: fast },
                    max_batch_size,
                    // allow:clone
                    providers.clone(),
                )) as UpscalerType,
                Upscaler::Esrgan4x => Box::new(esrgan::EsrGan::new(
                    EsrGanModel::X4Plus { f32: fast },
                    max_batch_size,
                    // allow:clone
                    providers.clone(),
                )) as UpscalerType,
                Upscaler::EsrganAnime4x => Box::new(esrgan::EsrGan::new(
                    EsrGanModel::X4PlusAnime6B { f32: fast },
                    max_batch_size,
                    // allow:clone
                    providers.clone(),
                )) as UpscalerType,
            };
            items.insert(key, upscaler);
        }
        Upscalers(items)
    }
}
