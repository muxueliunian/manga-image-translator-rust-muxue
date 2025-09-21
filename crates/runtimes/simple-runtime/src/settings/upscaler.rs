use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

#[derive(
    Serialize, Deserialize, Default, EnumIter, Hash, PartialEq, Eq, Copy, Clone, JsonSchema, Debug,
)]
pub enum Upscaler {
    Esrgan2x,
    #[default]
    Esrgan4x,
    EsrganAnime4x,
    Waifu2xCuNetArt(Option<u8>),
    Waifu2xSwinUnetArt2x(Option<u8>),
    Waifu2xSwinUnetArt4x(Option<u8>),
    Anime4k,
}

#[derive(Serialize, Deserialize, Default, Copy, Clone, JsonSchema)]
#[serde(default)]
pub struct UpscalerSettings {
    pub upscaler: Option<Upscaler>,
    pub patch_size: Option<usize>,
    pub padding: usize,
}
