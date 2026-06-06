use std::{
    collections::{HashMap, HashSet},
    slice,
};

use interface_translator::{Language, LanguageWrapper};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

#[derive(
    Serialize, Deserialize, Default, EnumIter, Hash, PartialEq, Eq, Copy, Clone, JsonSchema, Debug,
)]
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
    OpenAICompatible,
    #[default]
    Sugoi,
    Youdao,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct TranslatorSettings {
    pub target: Target,
    /// Filters out languages that should not be translated
    pub filter_lang: Vec<LanguageWrapper>,
    pub pre_dict: Option<String>,
    pub post_dict: Option<String>,
    pub openai_compatible: OpenAICompatibleSettings,
}

impl Default for TranslatorSettings {
    fn default() -> Self {
        Self {
            target: Target::Single(SingleOrMultiple::Single(Translation {
                translator: Translator::default(),
                target: LanguageWrapper(Language::English),
            })),
            filter_lang: vec![],
            pre_dict: None,
            post_dict: None,
            openai_compatible: OpenAICompatibleSettings::default(),
        }
    }
}

#[derive(Serialize, Deserialize, Default, Copy, Clone, JsonSchema, Debug, PartialEq, Eq)]
pub enum ProviderPreset {
    #[default]
    Custom,
    OpenAI,
    DeepSeek,
    OpenRouter,
    SiliconFlow,
    DashScope,
    Moonshot,
    Zhipu,
}

impl ProviderPreset {
    pub fn base_url(self) -> Option<&'static str> {
        match self {
            ProviderPreset::Custom => None,
            ProviderPreset::OpenAI => Some("https://api.openai.com/v1"),
            ProviderPreset::DeepSeek => Some("https://api.deepseek.com/v1"),
            ProviderPreset::OpenRouter => Some("https://openrouter.ai/api/v1"),
            ProviderPreset::SiliconFlow => Some("https://api.siliconflow.cn/v1"),
            ProviderPreset::DashScope => Some("https://dashscope.aliyuncs.com/compatible-mode/v1"),
            ProviderPreset::Moonshot => Some("https://api.moonshot.cn/v1"),
            ProviderPreset::Zhipu => Some("https://open.bigmodel.cn/api/paas/v4"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ProviderPreset::Custom => "Custom",
            ProviderPreset::OpenAI => "OpenAI",
            ProviderPreset::DeepSeek => "DeepSeek",
            ProviderPreset::OpenRouter => "OpenRouter",
            ProviderPreset::SiliconFlow => "SiliconFlow",
            ProviderPreset::DashScope => "DashScope",
            ProviderPreset::Moonshot => "Moonshot/Kimi",
            ProviderPreset::Zhipu => "Zhipu GLM",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, JsonSchema, Debug, PartialEq)]
#[serde(default)]
pub struct OpenAICompatibleSettings {
    pub provider_preset: ProviderPreset,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub system_prompt: String,
    pub user_prompt_template: String,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub timeout_secs: u64,
}

impl OpenAICompatibleSettings {
    pub fn resolved_base_url(&self) -> Option<&str> {
        if self.base_url.trim().is_empty() {
            self.provider_preset.base_url()
        } else {
            Some(self.base_url.trim())
        }
    }
}

impl Default for OpenAICompatibleSettings {
    fn default() -> Self {
        Self {
            provider_preset: ProviderPreset::Custom,
            base_url: String::new(),
            api_key: String::new(),
            model: String::new(),
            system_prompt: "You are a professional manga translator. Translate faithfully and preserve meaning, tone, and line breaks where appropriate. Return only the numbered translations.".to_string(),
            user_prompt_template: "Translate the following text from {source_language} to {target_language}.\nReturn exactly one translated line for each input line, preserving the numeric labels like [1]. Do not add explanations.\n\n{texts}".to_string(),
            temperature: Some(0.2),
            top_p: None,
            timeout_secs: 60,
        }
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Target {
    Single(SingleOrMultiple),
    Selective(HashMap<Option<LanguageWrapper>, SingleOrMultiple>),
}

impl Default for Target {
    fn default() -> Self {
        Target::Single(SingleOrMultiple::Single(Translation {
            translator: Translator::default(),
            target: LanguageWrapper(Language::English),
        }))
    }
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

#[derive(Hash, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum SingleOrMultiple {
    Single(Translation),
    Multiple(Vec<Translation>),
}

impl SingleOrMultiple {
    pub fn as_slice(&self) -> &[Translation] {
        match self {
            SingleOrMultiple::Single(t) => slice::from_ref(t),
            SingleOrMultiple::Multiple(ts) => ts,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Translation {
    pub translator: Translator,
    pub target: LanguageWrapper,
}

#[cfg(test)]
mod tests {
    use super::{OpenAICompatibleSettings, ProviderPreset};

    #[test]
    fn provider_preset_supplies_base_url_when_custom_url_empty() {
        let settings = OpenAICompatibleSettings {
            provider_preset: ProviderPreset::OpenAI,
            ..Default::default()
        };

        assert_eq!(
            settings.resolved_base_url(),
            Some("https://api.openai.com/v1")
        );
    }

    #[test]
    fn custom_base_url_overrides_provider_preset() {
        let settings = OpenAICompatibleSettings {
            provider_preset: ProviderPreset::OpenAI,
            base_url: " https://example.test/v1 ".to_string(),
            ..Default::default()
        };

        assert_eq!(
            settings.resolved_base_url(),
            Some("https://example.test/v1")
        );
    }
}
