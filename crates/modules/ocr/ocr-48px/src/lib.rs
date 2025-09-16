mod hypo;
mod infer;

use std::{fs::read_to_string, ops::Deref, sync::Arc};

use base_util::{
    onnx::{new_session, Providers},
    opencv_utils::to_continous2,
};
use interface_detector::textlines::Quadrilateral;
use interface_image::{ImageOp, RawImage};
use interface_model::{impl_model_load_helpers, Model, ModelLoad, ModelSource};
use interface_ocr::{Ocr, QuadrilateralInfo};
use maplit::hashmap;
use ndarray::{s, Array4};
use opencv::core::{MatTraitConst as _, MatTraitConstManual};
use ort::session::Session;
use util::{
    average::AvgMeter, resize::get_transformed_region, text_direction::generate_text_direction,
};

pub struct Ocr48px {
    model: Option<((Session, Session, Session), Vec<String>)>,
    providers: Vec<Providers>,
}

impl Ocr48px {
    pub fn new(providers: Vec<Providers>) -> Self {
        Self {
            model: None,
            providers,
        }
    }
}

pub struct Config {
    max_chunk_size: usize,
    max_seq_len: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_chunk_size: 16,
            max_seq_len: 255,
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

#[async_trait::async_trait]
impl Ocr for Ocr48px {
    async fn detect(
        &mut self,
        image: &Arc<RawImage>,
        areas: &[Arc<parking_lot::Mutex<Quadrilateral>>],
        _: &Arc<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<Vec<QuadrilateralInfo>> {
        let mut out = vec![];
        let text_height = 48;
        let config = Config::default();
        let whs = areas
            .iter()
            .map(|v| {
                let aabb = v.lock().aabb();
                let w = aabb.w;
                let h = aabb.h;
                let scale = text_height as f64 / w as f64;
                (h as f64 * scale) as u32
            })
            .collect::<Vec<_>>();
        let mut perm: Vec<usize> = (0..whs.len()).collect();
        let quadrilaterals = generate_text_direction(areas.to_vec()).collect::<Vec<_>>();
        perm.sort_by_key(|&i| whs[i]);

        let img = image.deref().clone().to_image().unwrap().to_rgb8();
        //TODO: apply direction
        let region_imgs = quadrilaterals
            .iter()
            .map(|(v, _)| {
                Ok::<_, anyhow::Error>((get_transformed_region(&*v.lock(), &img, text_height)?, v))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (region_imgs, areas): (Vec<_>, Vec<_>) = region_imgs.into_iter().unzip();

        let ((encoder, decoder, color_pred), dict) = self.load()?;
        let dict = &*dict;
        for indices in perm.chunks(config.max_chunk_size) {
            let n = indices.len();
            let img_slice = indices.iter().map(|v| &region_imgs[*v]).collect::<Vec<_>>();
            let widths = img_slice.iter().map(|v| v.cols()).collect::<Vec<_>>();
            let max_width = widths.iter().max().copied().unwrap_or_default() as usize + 7;
            let text_height = text_height as usize;
            let mut region = Array4::<u8>::zeros((n, text_height, max_width, 3));
            for (i, tmp) in img_slice.iter().enumerate() {
                let tmp = to_continous2(tmp);
                let data = tmp.data_bytes().expect("to_continous used");
                let rows = tmp.rows() as usize;
                let cols = tmp.cols() as usize;
                let row_stride = tmp.step1(0).unwrap();
                for y in 0..rows.min(text_height) {
                    let row_start = y * row_stride;
                    let row_end = row_start + (cols.min(max_width) * 3); // 3 channels
                    let row_slice = &data[row_start..row_end];
                    region
                        .slice_mut(s![i, y, 0..cols.min(max_width), ..])
                        .assign(
                            &ndarray::ArrayView::from_shape((cols.min(max_width), 3), row_slice)
                                .unwrap(),
                        );
                }
            }
            let images = region
                .mapv(|v| (v as f32 - 127.5) / 127.5)
                .permuted_axes([0, 3, 1, 2]);
            let texts = infer::infer(
                encoder,
                decoder,
                color_pred,
                images,
                widths,
                1,
                2,
                5,
                config.max_seq_len,
                2,
            );
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
                    pos: areas[indices[i]].clone(),
                    prob: pred.prob as f64,
                });
            }
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

    #[tokio::test]
    async fn ocr_test() {
        let img = RawImage::new("./imgs/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png")
            .expect("Failed to load image");
        let mut mocr = Ocr48px::new(all_providers());
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
        let mut v = mocr.detect(&Arc::new(img), &inp, &ip).await.unwrap();
        v.sort_by_key(|a| a.text.len());
        assert_eq!(v[0].pos.lock().pts()[0].x, 76);
        assert_eq!(v[0].text, "ふふっ、");
        assert_eq!(v[1].text, "そうだなあ‥");
        assert_eq!(v.len(), 2);
    }
}
