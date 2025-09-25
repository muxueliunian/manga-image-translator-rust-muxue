mod hypo;
mod infer;

use std::{collections::HashMap, fs::read_to_string, path::PathBuf, sync::Arc};

use base_util::{
    onnx::{new_session, Providers},
    opencv_utils::to_continous2,
};
use interface_detector::textlines::Quadrilateral;
use interface_image::{ImageOp, RawImage, RawImageCow};
use interface_model::{impl_model_load_helpers, Model, ModelLoad, ModelSource};
use interface_ocr::{Ocr, OcrOptions, QuadrilateralInfo};
use maplit::hashmap;
use ndarray::{s, Array4, Axis};
use opencv::core::{Mat, MatTraitConst as _, MatTraitConstManual};
use ort::{
    session::{RunOptions, Session},
    sys::OrtRunOptions,
};
use parking_lot::Mutex;
use util::{
    average::AvgMeter, ocr, resize::get_transformed_region, spawn_blocking,
    text_direction::generate_text_direction,
};

use crate::infer::Pred;

pub struct Ocr48px {
    model: Option<((Session, Session, Session), Vec<String>)>,
    providers: Arc<Vec<Providers>>,
    max_seq_len: i32,
    max_batch_size: usize,
}

impl Ocr48px {
    pub fn new(providers: Arc<Vec<Providers>>, max_seq_len: i32, max_batch_size: usize) -> Self {
        Self {
            model: None,
            providers,
            max_batch_size,
            max_seq_len,
        }
    }
}

impl ModelLoad for Ocr48px {
    type T = ((Session, Session, Session), Vec<String>);

    fn loaded(&self) -> bool {
        self.model.is_some()
    }

    fn get_model(&mut self) -> Option<&mut Self::T> {
        self.model.as_mut()
    }

    fn reload(&mut self) -> anyhow::Result<&mut Self::T> {
        let decoder = self.download_model("decoder", "decoder.onnx")?;
        let encoder = self.download_model("encoder", "encoder.onnx")?;
        let color_pred = self.download_model("color_pred", "color_pred.onnx")?;
        let dict = self.download_model("alphabet-all-v7", "alphabet-all-v7.txt")?;
        let dict = read_to_string(dict)
            .unwrap()
            .lines()
            .map(|v| v.trim_end().to_string())
            .collect::<Vec<String>>();
        let encoder = new_session(encoder, &self.providers)?;
        let color_pred = new_session(color_pred, &self.providers)?;
        let decoder = new_session(decoder, &self.providers)?;

        self.model = Some(((encoder, decoder, color_pred), dict));
        Ok(self.model.as_mut().unwrap())
    }
}
impl Model for Ocr48px {
    impl_model_load_helpers!("ocr", "48px");

    fn models(&self) -> std::collections::HashMap<&'static str, interface_model::ModelSource> {
        hashmap! {
            "decoder" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/ocr-48px/decoder.onnx", hash: "###" },
            "encoder" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/ocr-48px/encoder.onnx", hash: "###" },
            "color_pred" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/ocr-48px/color_pred.onnx", hash: "###" },
            "alphabet-all-v7" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/ocr-48px/alphabet-all-v7.txt", hash: "###" }
        }
    }

    fn unload(&mut self) {
        self.model = None;
    }
}

fn post_process(
    texts: Vec<Pred>,
    dict: &Vec<String>,
    areas: &[Arc<parking_lot::Mutex<Quadrilateral>>],
) -> Vec<QuadrilateralInfo> {
    let mut out = Vec::with_capacity(texts.len());
    for (i, pred) in texts.into_iter().enumerate() {
        let mut cur_texts = String::new();
        let mut avgs = [AvgMeter::default(); 6];
        let pred_chars_index = pred.out_idx;
        let fg_pred = pred.fg_pred;
        assert_eq!(fg_pred.len(), pred_chars_index.len());
        let has_fg = pred
            .fg_ind_pred
            .iter()
            .map(|v| (v.1 > v.0) as u32)
            .sum::<u32>() as f64
            / pred.fg_ind_pred.len() as f64
            > 0.5;
        let has_bg = pred
            .bg_ind_pred
            .iter()
            .map(|v| (v.1 > v.0) as u32)
            .sum::<u32>() as f64
            / pred.bg_ind_pred.len() as f64
            > 0.5;
        for (chid, fg_pred, bg_pred) in pred_chars_index
            .into_iter()
            .zip(fg_pred)
            .zip(pred.bg_pred)
            .map(|((x, y), z)| (x, y, z))
        {
            let mut ch = dict[chid as usize].as_str();
            if ch == "<S>" {
                continue;
            } else if ch == "</S>" {
                break;
            } else if ch == "<SP>" {
                ch = " ";
            } else {
                avgs[0].update((fg_pred.0 * 255.0).clamp(0.0, 255.0) as i32);
                avgs[1].update((fg_pred.1 * 255.0).clamp(0.0, 255.0) as i32);
                avgs[2].update((fg_pred.2 * 255.0).clamp(0.0, 255.0) as i32);
                avgs[3].update((bg_pred.0 * 255.0).clamp(0.0, 255.0) as i32);
                avgs[4].update((bg_pred.1 * 255.0).clamp(0.0, 255.0) as i32);
                avgs[5].update((bg_pred.2 * 255.0).clamp(0.0, 255.0) as i32);
            }
            cur_texts.push_str(ch);
        }

        out.push(QuadrilateralInfo {
            text: cur_texts,
            fg: match has_fg {
                true => Some([
                    avgs[0].average() as u8,
                    avgs[1].average() as u8,
                    avgs[2].average() as u8,
                ]),
                false => None,
            },
            bg: match has_bg {
                true => Some([
                    avgs[3].average() as u8,
                    avgs[4].average() as u8,
                    avgs[5].average() as u8,
                ]),
                false => None,
            },
            // allow:clone[arc]
            pos: areas[i].clone(),
            prob: pred.prob as f64,
        });
    }

    out
}

#[async_trait::async_trait]
impl Ocr for Ocr48px {
    async fn detect(
        &mut self,
        image: &RawImage,
        areas: &[Arc<parking_lot::Mutex<Quadrilateral>>],
        options: OcrOptions,
        _: &Arc<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<Vec<QuadrilateralInfo>> {
        let mut out = vec![];
        let text_height = 48;
        let items: Vec<(
            ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>,
            Vec<i32>,
            Vec<Arc<parking_lot::lock_api::Mutex<parking_lot::RawMutex, Quadrilateral>>>,
        )> = spawn_blocking!(|| ocr::prepare(
            image,
            areas,
            text_height as u32,
            self.max_batch_size,
            &options.debug_path,
        ))??;
        let max_seq_len = self.max_seq_len;
        let ((encoder, decoder, color_pred), dict) = self.load()?;
        let dict = &*dict;
        let run_options = RunOptions::new()?;
        for (images, widths, areas) in items {
            let texts = infer::infer(
                encoder,
                decoder,
                color_pred,
                images,
                widths,
                1,
                2,
                5,
                max_seq_len,
                2,
                &run_options,
            )
            .await;
            let texts = spawn_blocking!(|| post_process(texts, dict, &areas))?;
            out.extend(texts);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use base_util::onnx::all_providers;
    use interface_detector::textlines::Quadrilateral;
    use interface_image::{CpuImageProcessor, ImageOp, RawImage};
    use interface_ocr::Ocr as _;
    use parking_lot::Mutex;

    use crate::Ocr48px;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ocr_test() {
        let img = RawImage::new("./imgs/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png")
            .expect("Failed to load image");
        let mut mocr = Ocr48px::new(Arc::new(all_providers()), 255, 16);
        let inp = vec![
            Arc::new(Mutex::new(Quadrilateral::new(
                vec![(208, 4), (246, 4), (246, 192), (208, 192)],
                1.0,
            ))),
            Arc::new(Mutex::new(Quadrilateral::new(
                vec![(76, 1788), (128, 1788), (128, 1930), (76, 1930)],
                1.0,
            ))),
        ];
        let ip = Arc::new(CpuImageProcessor::default()) as Arc<dyn ImageOp + Send + Sync>;
        let mut v = mocr
            .detect(&Arc::new(img), &inp, Default::default(), &ip)
            .await
            .unwrap();
        v.sort_by_key(|a| a.text.len());
        assert_eq!(v[0].pos.lock().pts()[0].x, 76);
        assert_eq!(v[0].text, "ふふっ、");
        assert_eq!(v[1].text, "そうだなあ‥");
        assert_eq!(v.len(), 2);
    }
}
