mod detector;
mod dict;
mod inpainter;
mod mask_refinement;
mod ocr;
mod textline_merge;
mod translator;
mod upscaler;

use std::{path::PathBuf, ptr, sync::Arc};

use export::Export;
use image::DynamicImage;
use interface_image::{CpuImageProcessor, ImageOp, RawImage};

use crate::{
    debug::{bbox::render_bboxes, save_img, save_json, save_mask, textblocks::render_textblocks},
    settings::Settings,
    setup::Models,
};

pub type ImageProcessor = Arc<dyn ImageOp + Sync + Send>;

impl Models {
    pub async fn execute(
        &mut self,
        img: DynamicImage,
        config: &Settings,
        debug_path: Option<PathBuf>,
    ) -> anyhow::Result<Option<Export>> {
        let ip = Arc::new(CpuImageProcessor::default()) as ImageProcessor;
        let (img, alpha) = RawImage::rgba(img);
        let (img, alpha) = self.run_upscaler(img, alpha, config.upscaler, &ip)?;

        if let Some(debug_path) = &debug_path {
            save_json(config, &debug_path.join("0_config.json"))?;
            save_img(&img, &debug_path.join("0_input.png"))?;
        }

        let (areas, mask) = self.run_detector(&img, &config.detector, &ip)?;
        if let Some(debug_path) = &debug_path {
            save_mask(&mask, &debug_path.join("1_mask_raw.png"))?;
            save_json(&areas, &debug_path.join("1_quadrilateral.json"))?;
            render_bboxes(&img, &areas, debug_path)?;
        }
        if areas.is_empty() {
            return Ok(None);
        }

        let areas = areas.into_iter().map(to_mutex).collect::<Vec<_>>();
        let upscaled_img = Arc::new(img);

        let textlines = self
            .run_ocr(&upscaled_img, &areas, &config.ocr, &debug_path, &ip)
            .await?;

        if textlines.is_empty() {
            return Ok(None);
        }

        if let Some(debug_path) = &debug_path {
            save_json(&textlines, &debug_path.join("2_quadrilateral.json"))?;
        }

        let textblocks = self.run_textline_merge(
            &textlines,
            upscaled_img.width,
            upscaled_img.height,
            &config.ocr,
            &config.translator,
        )?;
        if textblocks.is_empty() {
            return Ok(None);
        }

        if let Some(debug_path) = &debug_path {
            save_json(&textblocks, &debug_path.join("3_textblock.json"))?;
            render_textblocks(&upscaled_img, &textblocks, debug_path)?;
        }

        let textblocks = self.run_pre_dict(textblocks, &config.translator)?;
        if let Some(debug_path) = &debug_path {
            if config.translator.pre_dict.is_some() {
                save_json(
                    &textlines,
                    &debug_path.join("3_textblock_predict_applied.json"),
                )?;
            }
        }

        let textblocks = self.run_translators(textblocks, &config.translator).await?;

        if let Some(debug_path) = &debug_path {
            save_json(
                &textblocks,
                &debug_path.join("4_textblocks_translated.json"),
            )?;
        }

        let mask_refined = Models::run_mask_refinement(
            &upscaled_img,
            &mask,
            &textblocks,
            &config.mask_refinement,
            &ip,
        )?;

        if let Some(debug_path) = &debug_path {
            save_mask(&mask_refined, &debug_path.join("4_mask_refined.png"))?;
        }

        let (inpainted, mask) =
            self.run_inpainter(&upscaled_img, mask, mask_refined, &config.inpainter, &ip)?;

        let inpainted = inpainted.add_a(mask.data);
        if let Some(debug_path) = &debug_path {
            let mut img = upscaled_img.as_ref().clone();
            img.apply_filter(&inpainted, |a, b| unsafe {
                if *b.get_unchecked(3) > 128 {
                    ptr::copy_nonoverlapping(b.as_ptr(), a.as_mut_ptr(), 3);
                }
            });
            save_img(&img, &debug_path.join("5_inpainted.png"))?;
        }

        Ok(Some(Export::new(
            match alpha {
                Some(a) => upscaled_img.as_ref().clone().add_a(a),
                None => upscaled_img.as_ref().clone(),
            }
            .to_image()?,
            inpainted.to_image()?,
            textblocks,
            None,
        )))
    }
}

fn to_mutex<T>(areas: T) -> Arc<parking_lot::Mutex<T>> {
    Arc::new(parking_lot::Mutex::new(areas))
}
