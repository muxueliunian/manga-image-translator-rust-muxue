mod decode;

use std::{fs::read_to_string, ops::Deref, sync::Arc};

use base_util::onnx::{new_session, Providers};
use interface_detector::textlines::Quadrilateral;
use interface_image::{ImageOp, Mask, RawImage};
use interface_model::{impl_model_load_helpers, Model, ModelLoad, ModelSource};
use interface_ocr::{Ocr, QuadrilateralInfo};
use maplit::hashmap;
use ndarray::{s, Array4};
use opencv::core::{MatTraitConst as _, MatTraitConstManual};
use ort::session::Session;
use parking_lot::Mutex;
use util::{
    average::AvgMeter, resize::get_transformed_region, text_direction::generate_text_direction,
};

pub struct Ctc48pxOcr {
    model: Option<(Session, Vec<String>)>,
    providers: Arc<Vec<Providers>>,
}

impl Ctc48pxOcr {
    pub fn new(providers: Arc<Vec<Providers>>) -> Self {
        Self {
            model: None,
            providers,
        }
    }
}

pub struct Config {
    max_chunk_size: usize,
    ignore_bubble: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_chunk_size: 16,
            ignore_bubble: false,
        }
    }
}

impl ModelLoad for Ctc48pxOcr {
    type T = (Session, Vec<String>);

    fn loaded(&self) -> bool {
        self.model.is_some()
    }

    fn get_model(&mut self) -> Option<&mut Self::T> {
        self.model.as_mut()
    }

    fn reload(&mut self) -> anyhow::Result<&mut Self::T> {
        let model = self.download_model("model", "model.onnx")?;
        let dict = self.download_model("alphabet-all-v5", "alphabet-all-v5.txt")?;
        let dict = read_to_string(dict)
            .unwrap()
            .lines()
            .map(|v| v.trim_end().to_string())
            .collect::<Vec<String>>();
        let model = new_session(model, &self.providers)?;

        self.model = Some((model, dict));
        Ok(self.model.as_mut().unwrap())
    }
}
impl Model for Ctc48pxOcr {
    impl_model_load_helpers!("ocr", "ctc-48px");

    fn models(&self) -> std::collections::HashMap<&'static str, interface_model::ModelSource> {
        hashmap! {
            "model" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/ctc-48px/model.onnx", hash: "###" },
            "alphabet-all-v5" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/ctc-48px/alphabet-all-v5.txt", hash: "###" }
        }
    }

    fn unload(&mut self) {
        self.model = None;
    }
}

#[async_trait::async_trait]
impl Ocr for Ctc48pxOcr {
    async fn detect(
        &mut self,
        image: &Arc<RawImage>,
        areas: &[Arc<parking_lot::Mutex<Quadrilateral>>],
        _: &Arc<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<Vec<QuadrilateralInfo>> {
        //TODO: ignore bubble
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

        let (model, dict) = self.load()?;
        let dict = &*dict;
        for indices in perm.chunks(config.max_chunk_size) {
            let n = indices.len();
            let img_slice = indices.iter().map(|v| &region_imgs[*v]).collect::<Vec<_>>();
            let widths = img_slice.iter().map(|v| v.cols()).collect::<Vec<_>>();
            let max_width = widths.iter().max().copied().unwrap_or_default() as usize + 135;
            let text_height = text_height as usize;
            let mut region = Array4::<u8>::zeros((n, text_height, max_width, 3));
            for (i, tmp) in img_slice.iter().enumerate() {
                let keep_alive;
                let data = match tmp.data_bytes() {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        keep_alive = (*tmp).clone();
                        keep_alive.data_bytes().unwrap()
                    }
                };
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
            let texts = decode::decode(model, images, 0);
            for (i, single_line) in texts.into_iter().enumerate() {
                if single_line.is_empty() {
                    continue;
                }
                let mut cur_texts = String::new();
                let mut total = AvgMeter::default();
                let mut avgs = [AvgMeter::default(); 6];
                for (chid, logprob, fr, fg, fb, br, bg, bb) in single_line {
                    let mut ch = dict[chid as usize].as_str();
                    if ch == "<SP>" {
                        ch = " ";
                    } else {
                        avgs[0].update((fr * 255.0) as i32);
                        avgs[1].update((fg * 255.0) as i32);
                        avgs[2].update((fb * 255.0) as i32);
                        avgs[3].update((br * 255.0) as i32);
                        avgs[4].update((bg * 255.0) as i32);
                        avgs[5].update((bb * 255.0) as i32);
                    }
                    cur_texts.push_str(ch);
                    total.update(logprob);
                }
                let prob = total.average().exp();

                out.push(QuadrilateralInfo {
                    text: cur_texts,
                    fg: Some([
                        avgs[0].average() as u8,
                        avgs[1].average() as u8,
                        avgs[2].average() as u8,
                    ]),
                    bg: Some([
                        avgs[3].average() as u8,
                        avgs[4].average() as u8,
                        avgs[5].average() as u8,
                    ]),
                    pos: areas[indices[i]].clone(),
                    prob,
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

    use crate::Ctc48pxOcr;

    #[tokio::test]
    async fn ocr_test() {
        let img = RawImage::new("./imgs/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png")
            .expect("Failed to load image");
        let mut mocr = Ctc48pxOcr::new(Arc::new(all_providers()));
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
        assert_eq!(v[1].text, "そうだなあ…");
        assert_eq!(v[0].text, "ふふっ、");
        assert_eq!(v.len(), 2);
    }
}
