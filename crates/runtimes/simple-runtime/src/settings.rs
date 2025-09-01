use std::collections::{HashMap, HashSet};

use interface_detector::{DefaultOptions, PreprocessorOptions};
use interface_translator::Language;
use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
/// Settings for the simple runtime
pub struct Settings {
    /// Settings for the upscaler module
    pub upscaler: UpscalerSettings,

    /// Settings for the detector module
    pub detector: DetectorSettings,

    /// Settings for the OCR module
    pub ocr: OCRSettings,

    /// Settings for the inpainter module
    pub inpainter: InpainterSettings,

    /// Settings for the translator module
    pub translator: TranslatorSettings,

    /// Settings for the render module
    pub render: RenderSettings,
}

#[derive(Serialize, Deserialize, Default)]
pub struct TranslatorSettings {
    pub translator: Translator,
    /// Filters out languages that should not be translated
    pub filter_lang: Vec<String>,
    pub pre_dict: Option<String>,
}

// #[derive(Deserialize)]
// #[serde(untagged)]
pub enum Target {
    Single(SingleOrMultiple),
    Selective(HashMap<Option<Language>, SingleOrMultiple>),
}

impl Target {
    pub fn validate(&self) -> Option<&'static str> {
        match self {
            Target::Single(_) => None,
            Target::Selective(hash_map) => {
                if hash_map.get(&None).is_none() {
                    return Some("no default");
                };
                for mut key in hash_map.keys().cloned() {
                    let mut keys_used = HashSet::new();
                    loop {
                        let value = hash_map.get(&key);
                        let value = match value {
                            Some(v) => v,
                            None => return None,
                        };
                        let v = keys_used.insert(key);
                        if !v {
                            return Some("loop detected");
                        }
                        let next = match value {
                            SingleOrMultiple::Single(translation) => translation.target,
                            SingleOrMultiple::Multiple(translations) => {
                                if translations.is_empty() {
                                    return Some("empty array");
                                }
                                translations
                                    .last()
                                    .expect("translations should not be empty")
                                    .target
                            }
                        };
                        key = Some(next);
                    }
                }
                None
            }
        }
    }
}

#[derive(Hash, Eq, PartialEq)]
// #[serde(untagged)]
pub enum SingleOrMultiple {
    Single(Translation),
    Multiple(Vec<Translation>),
}

#[derive(Hash, Eq, PartialEq)]
pub struct Translation {
    translator: Translator,
    target: Language,
}

#[derive(Serialize, Deserialize, Default, EnumIter, Hash, PartialEq, Eq, Copy, Clone)]
pub enum OCR {
    #[default]
    MangaOcr,
    Native,
    Tesseract,
    Ctc48px,
}
#[derive(Serialize, Deserialize, Default, EnumIter, Hash, PartialEq, Eq, Copy, Clone)]
pub enum Translator {
    JParaCrawlSmall,
    JParaCrawlBase,
    JParaCrawlLarge,
    Baidu,
    Caiyun,
    Deepl,
    Google,
    M2M100Small,
    M2M100Large,
    MBart,
    MyMemory,
    NLLBSmallDistilled,
    NLLBBase,
    NLLBLarge,
    Papago,
    #[default]
    Sugoi,
    Youdao,
}

#[derive(Serialize, Deserialize, Default, EnumIter, Hash, PartialEq, Eq, Copy, Clone)]
pub enum Detector {
    #[default]
    DBNet,
    // DBNetConvNext,
    Paddle,
    Ctd,
}

#[derive(Serialize, Deserialize, Default, EnumIter, Hash, PartialEq, Eq, Copy, Clone)]
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
#[derive(Serialize, Deserialize, Default, EnumIter, Hash, PartialEq, Eq, Copy, Clone)]
pub enum Inpainter {
    #[default]
    LamaAot,
    LamaLarge,
    LamaMpe,
}

#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DetectorSettings {
    /// Text detector used for creating a text mask from an image, DO NOT use craft for manga, it\'s not designed for it
    pub detector: Detector,
    /// General Options to apply before detection
    pub preprocessor: PreprocessorOptions,
    /// Detector specific options
    pub options: DefaultOptions,
    // todo: skip mask generation
}
#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
pub struct OCRSettings {
    /// Optical character recognition (OCR) model to use
    pub ocr: OCR,
    /// Minimum text length of a text region
    pub min_text_length: usize,
    /// Filter regions by their text with a regex. Example usage: '.*badtext.*'
    /// todo: regex
    pub filter_text: Vec<String>,
    /// Minimum probability of a text region to be considered valid. If None, uses the model default
    /// todo: not used
    prob: Option<f32>,
    /// Use bbox merge when Manga OCR inference.
    /// todo: not used
    use_mocr_merge: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct InpainterSettings {
    /// Inpainting model to use
    pub inpainter: Inpainter,
    /// Size of image used for inpainting (too large will result in OOM)
    pub inpainting_size: u16,
    /// The threshold for ignoring text in non bubble areas, with valid values ranging from 1 to 50, does not ignore others. Recommendation 5 to 10. If it is too low, normal bubble areas may be ignored, and if it is too large, non bubble areas may be considered normal bubbles
    pub ignore_bubble: Option<u8>,
    /// By how much to extend the text mask to remove left-over text pixels of the original image.
    mask_dilation_offset: u32,
    /// Set the convolution kernel size of the text erasure area to completely clean up text residues"
    kernel_size: u8,
    pub furi: bool,
    /// If no ai is used for inpainting than use this color
    pub inpaint_color: [u8; 3],
    sort: Option<Sort>,
}

#[derive(Serialize, Deserialize)]
enum Sort {
    Simple,
    Advanced,
}

impl Default for InpainterSettings {
    fn default() -> Self {
        Self {
            inpainter: Default::default(),
            inpainting_size: 2048,
            ignore_bubble: None,
            sort: None,
            kernel_size: 3,
            mask_dilation_offset: 20,
            inpaint_color: [255; 3],
            furi: false,
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RenderSettings {}

#[derive(Serialize, Deserialize, Default, Copy, Clone)]
#[serde(default)]
pub struct UpscalerSettings {
    pub upscaler: Option<Upscaler>,
    pub patch_size: Option<usize>,
    pub padding: usize,
}
