use std::time::Duration;

use anyhow::{anyhow, bail};
use interface_translator::Detector;
use interface_translator::TranslationListOutput;
use log::info;
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use textline_merge::TextBlock;

use crate::{
    settings::{OpenAICompatibleSettings, Target, Translation, Translator, TranslatorSettings},
    setup::Models,
};

impl Models {
    pub async fn run_translators(
        &mut self,
        textblocks: Vec<TextBlock>,
        config: &TranslatorSettings,
    ) -> anyhow::Result<Vec<TextBlock>> {
        match &config.target {
            Target::Single(items) => {
                self.run_translator_list(textblocks, items.as_slice(), config)
                    .await
            }
            Target::Selective(hash_map) => todo!("selective not implemented yet"),
        }
    }

    pub async fn run_translator_list(
        &mut self,
        mut textblocks: Vec<TextBlock>,
        translators: &[Translation],
        config: &TranslatorSettings,
    ) -> anyhow::Result<Vec<TextBlock>> {
        assert!(!textblocks.is_empty());
        let mut textblocks_use = textblocks
            .iter_mut()
            .filter(|v| !v.skip_translate)
            .collect::<Vec<_>>();

        let texts = textblocks_use
            .iter()
            .map(|v| v.text.clone())
            .collect::<Vec<_>>();
        for tb in &textblocks_use {
            assert!(tb.translations.is_empty());
        }

        let d_str = texts.join(" ");
        let lang = self.lang_detector.detect_language(&d_str);

        let mut texts = TranslationListOutput { text: texts, lang };

        for translator in translators {
            let out = self.run_translator_item(texts, translator, config).await?;
            let lang_str = out.lang.map(|v| v.to_name().unwrap()).unwrap_or("unknown");
            for (i, item) in out.text.iter().enumerate() {
                textblocks_use[i]
                    .translations
                    .insert(lang_str.to_owned(), item.to_owned());
            }
            texts = out;
        }
        let lang_str = texts
            .lang
            .map(|v| v.to_name().unwrap())
            .unwrap_or("unknown");

        for item in textblocks_use.iter_mut() {
            item.translations
                .insert("last_trans".to_owned(), lang_str.to_owned());
        }
        Ok(textblocks)
    }
    pub async fn run_translator_item(
        &mut self,
        input: TranslationListOutput,
        translator_info: &Translation,
        config: &TranslatorSettings,
    ) -> anyhow::Result<TranslationListOutput> {
        info!("Run Translator: {:?}", translator_info.translator);
        let to = translator_info.target.0;
        // TODO: set fallback language in config
        let from = input.lang.ok_or(anyhow!("Failed to detect language"))?;

        let text = if translator_info.translator == Translator::OpenAICompatible {
            translate_openai_compatible(
                &input.text,
                from.to_name().unwrap(),
                to.to_name().unwrap(),
                &config.openai_compatible,
            )
            .await?
        } else {
            let translator = self.get_translator(translator_info.translator);
            translator
                .translate_vec(&input.text, None, Some(from), &to)
                .await?
                .text
        };

        let d_str = text.join(" ");
        let lang = self.lang_detector.detect_language(&d_str);

        Ok(TranslationListOutput { text, lang })
    }
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Deserialize)]
struct ChatCompletionChoice {
    message: ChatCompletionMessage,
}

#[derive(Deserialize)]
struct ChatCompletionMessage {
    content: String,
}

async fn translate_openai_compatible(
    texts: &[String],
    source_language: &str,
    target_language: &str,
    settings: &OpenAICompatibleSettings,
) -> anyhow::Result<Vec<String>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let base_url = settings
        .resolved_base_url()
        .ok_or_else(|| anyhow!("OpenAI-compatible base_url is required"))?;
    if settings.api_key.trim().is_empty() {
        bail!("OpenAI-compatible api_key is required");
    }
    if settings.model.trim().is_empty() {
        bail!("OpenAI-compatible model is required");
    }

    let numbered_texts = texts
        .iter()
        .enumerate()
        .map(|(i, text)| format!("[{}] {}", i + 1, text))
        .collect::<Vec<_>>()
        .join("\n");
    let user_prompt = settings
        .user_prompt_template
        .replace("{source_language}", source_language)
        .replace("{target_language}", target_language)
        .replace("{texts}", &numbered_texts);

    let request = ChatCompletionRequest {
        model: settings.model.clone(),
        messages: vec![
            ChatMessage {
                role: "system",
                content: settings.system_prompt.clone(),
            },
            ChatMessage {
                role: "user",
                content: user_prompt,
            },
        ],
        temperature: settings.temperature,
        top_p: settings.top_p,
    };

    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let client = Client::builder()
        .timeout(Duration::from_secs(settings.timeout_secs.max(1)))
        .build()?;
    let response = client
        .post(url)
        .bearer_auth(settings.api_key.trim())
        .json(&request)
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        bail!("OpenAI-compatible request failed with {status}: {body}");
    }

    let response: ChatCompletionResponse = serde_json::from_str(&body)?;
    let content = response
        .choices
        .first()
        .ok_or_else(|| anyhow!("OpenAI-compatible response contained no choices"))?
        .message
        .content
        .as_str();

    parse_numbered_translations(content, texts.len())
}

fn parse_numbered_translations(content: &str, expected: usize) -> anyhow::Result<Vec<String>> {
    let re = Regex::new(r"^\s*(?:\[(\d+)\]|(\d+)[\.\):：、])\s*(.*)\s*$")?;
    let mut parsed = vec![None; expected];

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        if let Some(captures) = re.captures(line) {
            let raw_number = captures
                .get(1)
                .or_else(|| captures.get(2))
                .ok_or_else(|| anyhow!("missing translation number"))?
                .as_str();
            let number = raw_number.parse::<usize>()?;
            if number == 0 || number > expected {
                bail!("translation number {number} is outside expected range 1..={expected}");
            }
            let index = number - 1;
            if parsed[index].is_some() {
                bail!("duplicate translation number {number}");
            }
            parsed[index] = Some(
                captures
                    .get(3)
                    .map(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            );
        } else {
            bail!("unnumbered translation line: {line}");
        }
    }

    parsed
        .into_iter()
        .enumerate()
        .map(|(i, item)| item.ok_or_else(|| anyhow!("missing translation number {}", i + 1)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::parse_numbered_translations;

    #[test]
    fn parses_bracketed_numbered_translations() {
        let parsed = parse_numbered_translations("[1] Hello\n[2] World", 2).unwrap();

        assert_eq!(parsed, vec!["Hello", "World"]);
    }

    #[test]
    fn rejects_missing_numbers() {
        let error = parse_numbered_translations("[1] Hello\n[3] Later", 3)
            .unwrap_err()
            .to_string();

        assert!(error.contains("missing translation number 2"));
    }

    #[test]
    fn rejects_duplicate_numbers() {
        let error = parse_numbered_translations("[1] Hello\n[1] Again", 1)
            .unwrap_err()
            .to_string();

        assert!(error.contains("duplicate translation number 1"));
    }

    #[test]
    fn rejects_unnumbered_lines() {
        let error = parse_numbered_translations("[1] Hello\nWorld", 1)
            .unwrap_err()
            .to_string();

        assert!(error.contains("unnumbered translation line"));
    }
}
