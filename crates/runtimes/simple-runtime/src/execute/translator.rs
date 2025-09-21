use anyhow::anyhow;
use interface_translator::Detector;
use interface_translator::TranslationListOutput;
use log::info;
use textline_merge::TextBlock;

use crate::{
    settings::{Target, Translation, TranslatorSettings},
    setup::Models,
};

impl Models {
    pub async fn run_translators(
        &mut self,
        textblocks: Vec<TextBlock>,
        config: &TranslatorSettings,
    ) -> anyhow::Result<Vec<TextBlock>> {
        match &config.target {
            Target::Single(items) => self.run_translator_list(textblocks, items.as_slice()).await,
            Target::Selective(hash_map) => todo!("selective not implemented yet"),
        }
    }

    pub async fn run_translator_list(
        &mut self,
        mut textblocks: Vec<TextBlock>,
        translators: &[Translation],
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
            let out = self.run_translator_item(texts, translator).await?;
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
    ) -> anyhow::Result<TranslationListOutput> {
        info!("Run Translator: {:?}", translator_info.translator);
        let to = translator_info.target.0;
        let translator = self.get_translator(translator_info.translator);
        let translations = if translator.local() {
            // TODO: set fallback language in config
            let from = input.lang.ok_or(anyhow!("Failed to detect language"))?;
            let t = translator
                .translator_mut()
                .as_blocking()
                .unwrap()
                .translate_vec(&input.text, None, from, &to)?;
            let d_str = t.join(" ");
            let lang = self.lang_detector.detect_language(&d_str);
            TranslationListOutput { text: t, lang }
        } else {
            translator
                .translator()
                .as_async()
                .unwrap()
                .translate_vec(&input.text, None, input.lang, &to)
                .await?
        };
        Ok(translations)
    }
}
