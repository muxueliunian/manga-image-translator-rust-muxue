mod refine_mask;

use base_util::{
    error::{PostProcessingError, ProcessingError},
    onnx::new_session,
};

use interface_detector::{textlines::Quadrilateral, Detector};
use interface_image::{ImageOp, Interpolation, Mask, RawImage};
use interface_model::{impl_model_load_helpers, CreateData, Model, ModelLoad, ModelSource};
use maplit::hashmap;
use ndarray::{s, stack, Array2, Array4, ArrayViewD, Axis};
use ort::{session::Session, value::Tensor};
use util::{
    dbnet::SegDetectorRepresenter,
    det_arrange::{det_rearrange_forward, shoud_rearrange},
};

pub struct CtdDetector {
    db: CreateData,
    model: Option<Session>,
}

impl CtdDetector {
    ///convnext: Different model architecture, but based on dbnet
    pub fn new(db: CreateData) -> Self {
        CtdDetector { db, model: None }
    }
}

impl ModelLoad for CtdDetector {
    type T = Session;
    fn loaded(&self) -> bool {
        self.model.is_some()
    }

    fn reload(&mut self) -> Result<&mut Self::T, base_util::error::ModelLoadError> {
        self.model = Some(new_session(
            self.download_model("model", "model.onnx")?,
            self.db.providers.clone(),
        )?);
        Ok(self.model.as_mut().unwrap())
    }

    fn get_model(&mut self) -> Option<&mut Self::T> {
        self.model.as_mut()
    }
}

impl Model for CtdDetector {
    impl_model_load_helpers!("detector", "ctd");

    fn models(&self) -> std::collections::HashMap<&'static str, ModelSource> {
        hashmap! {
            "model" => ModelSource { url: "https://github.com/frederik-uni/manga-image-translator-rust/releases/download/ctd-v1.0.0/model.onnx", hash: "c921d44fea30913a1689dcb4d28faef664dfd0c9f895146d27342e52b823ec0c" }
        }
    }
    fn unload(&mut self) {
        self.model = None;
    }
}

impl Detector for CtdDetector {
    fn infer(
        &mut self,
        img: RawImage,
        _: &[u8],
        img_processor: &Box<dyn ImageOp + Send + Sync>,
    ) -> anyhow::Result<(Vec<Quadrilateral>, Mask)> {
        let (im_w, im_h) = (img.width, img.height);
        let session = self.load()?;
        let (lines_map, mask) = match shoud_rearrange(&img, 1024) {
            true => {
                let v = |batch| det_batch_forward_ctd(session, batch);
                let (lines_map, mask) =
                    det_rearrange_forward(img.clone(), 1024, 4, v, img_processor)?;
                (lines_map, mask)
            }
            false => {
                let (img_in, _, dw, dh) =
                    preprocess_img(img.clone(), img_processor, (1024, 1024), true, false)?;
                let tensor = Tensor::from_array(img_in)?;
                let outputs = session.run(ort::inputs!["input" => tensor])?;

                let mask: ArrayViewD<f32> = outputs["mask"].try_extract_array()?;
                let lines_map: ArrayViewD<f32> = outputs["lines"].try_extract_array()?;
                let mut lines_map = lines_map.into_dimensionality::<ndarray::Ix4>()?.to_owned();
                let mut mask = mask.into_dimensionality::<ndarray::Ix4>()?.to_owned();
                let mask_shape = mask.shape().to_vec();
                let lines_shape = lines_map.shape().to_vec();
                let dh = dh as usize;
                let dw = dw as usize;
                mask = mask.slice_move(s![.., .., 0..mask_shape[2] - dh, 0..mask_shape[3] - dw]);

                lines_map = lines_map.slice_move(s![
                    ..,
                    ..,
                    0..lines_shape[2] - dh,
                    0..lines_shape[3] - dw
                ]);
                (lines_map, mask)
            }
        };

        let mask: Array2<f32> = mask
            .index_axis(ndarray::Axis(0), 0)
            .index_axis(ndarray::Axis(0), 0)
            .to_owned();

        let mask = Mask::from(mask.mapv(|v| f32::clamp(v * 255.0, 0.0, 255.0) as u8));
        let (lines, scores) =
            SegDetectorRepresenter::default().call(lines_map, false, im_w as u16, im_h as u16)?;
        let box_thresh = 0.6;
        let scores = scores
            .into_iter()
            .flatten()
            .next()
            .ok_or(PostProcessingError::Empty)?;
        let lines = lines
            .into_iter()
            .flatten()
            .next()
            .ok_or(PostProcessingError::Empty)?;
        let qu = lines
            .outer_iter()
            .zip(scores)
            .filter(|(_, score)| *score > box_thresh)
            .map(|(points, score)| {
                Quadrilateral::new(
                    points
                        .outer_iter()
                        .map(|v| (v[0], v[1]))
                        .collect::<Vec<(i64, i64)>>(),
                    score,
                )
            })
            .collect::<Vec<_>>();
        let mask =
            img_processor.resize_mask(mask, im_w as usize, im_h as usize, Interpolation::Bilinear);
        let mask_refined = refine_mask::refine_mask(img, mask, qu.clone(), false)?;

        Ok((qu, mask_refined))
    }
}
fn det_batch_forward_ctd(
    session: &mut Session,
    batch: Array4<u8>,
) -> Result<(Array4<f32>, Array4<f32>), ProcessingError> {
    let batch = batch.mapv(|v| v as f32 / 255.).permuted_axes([0, 3, 1, 2]);
    let tensor = Tensor::from_array(batch)?;
    let outputs = session.run(ort::inputs!["input" => tensor])?;

    let mask: ArrayViewD<f32> = outputs["mask"].try_extract_array()?;
    let lines: ArrayViewD<f32> = outputs["lines"].try_extract_array()?;
    Ok((
        lines.into_dimensionality::<ndarray::Ix4>()?.to_owned(),
        mask.into_dimensionality::<ndarray::Ix4>()?.to_owned(),
    ))
}

fn preprocess_img(
    mut img: RawImage,
    img_processor: &Box<dyn ImageOp + Send + Sync>,
    input_size: (u32, u32),
    bgr2rgb: bool,
    half: bool,
) -> Result<(Array4<f32>, (f64, f64), u32, u32), ProcessingError> {
    if bgr2rgb {
        img = img_processor.bgr_to_rgb(img);
    }
    let (img_in, ratio, (dw, dh)) =
        letterbox(img, img_processor, input_size, false, false, true, 64);
    let nd = img_in.to_ndarray()?;
    let nd = nd.permuted_axes([2, 0, 1]);
    let nd = nd.slice(s![..;-1, .., ..]);
    let nd = nd.mapv(|v| v as f32 / 255.0);
    let nd = stack(Axis(0), &[nd.view()])?;
    if half {
        todo!("convert to f16")
    }
    Ok((nd, ratio, dw, dh))
}

fn letterbox(
    mut im: RawImage,
    img_processor: &Box<dyn ImageOp + Send + Sync>,
    new_shape: (u32, u32),
    auto: bool,
    scale_fill: bool,
    scaleup: bool,
    stride: u32,
) -> (RawImage, (f64, f64), (u32, u32)) {
    let (w, h) = (im.width, im.height);
    let mut r = f64::min(new_shape.0 as f64 / h as f64, new_shape.1 as f64 / w as f64);
    if !scaleup {
        r = 1.0_f64.min(r);
    }
    let mut ratio = (r, r);
    let mut new_unpad = ((w as f64 * r).round() as u32, (h as f64 * r).round() as u32);
    let (mut dw, mut dh) = ((new_shape.1 - new_unpad.0), (new_shape.0 - new_unpad.1));
    if auto {
        dw = dw % stride;
        dh = dh % stride;
    } else if scale_fill {
        dw = 0;
        dh = 0;
        new_unpad = (new_shape.1, new_shape.0);
        ratio = (new_shape.1 as f64 / w as f64, new_shape.0 as f64 / h as f64);
    }

    if new_unpad.0 != w as u32 || new_unpad.1 != h as u32 {
        im = img_processor.resize(
            im,
            new_unpad.0 as u16,
            new_unpad.1 as u16,
            Interpolation::Bilinear,
        );
    }
    let im_height = im.height;
    let im_width = im.width;
    im = img_processor.add_border_wh(im, im_width + dw as u16, im_height + dh as u16);
    (im, ratio, (dw, dh))
}

#[cfg(test)]
mod tests {

    use interface_detector::{Detector as _, PreprocessorOptions};
    use interface_image::{CpuImageProcessor, ImageOp, RawImage};
    use interface_model::{CreateData, Model as _, ModelLoad as _};

    use crate::CtdDetector;

    #[test]
    fn load_unload() {
        let mut data = CtdDetector::new(CreateData::all());
        data.load().expect("failed to load model");
        data.unload();
    }

    #[test]
    fn run() {
        let mut data = CtdDetector::new(CreateData::all());
        let cpu_image_processor =
            Box::new(CpuImageProcessor::default()) as Box<dyn ImageOp + Send + Sync>;
        data.load().expect("Failed to load data");
        let img = RawImage::new("./imgs/232265329-6a560438-e887-4f7f-b6a1-a61b8648f781.png")
            .expect("Failed to load image");
        let (v, _) = data
            .detect(
                &img,
                PreprocessorOptions::default(),
                &[],
                &cpu_image_processor,
            )
            .expect("failed to detect");
        println!("{:?}", v);
    }
}
