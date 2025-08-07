use ab_glyph::FontVec;
use image::{DynamicImage, GrayImage, Pixel, Rgb, RgbImage};
use imageproc::{
    drawing::{draw_line_segment_mut, draw_text_mut},
    rect::Rect,
};
use interface_image::{CpuImageProcessor, ImageOp, RawImage};
use textline_merge::TextBlock;

use crate::{settings::Settings, setup::Models};

impl Models {
    pub async fn execute(&mut self, img: DynamicImage, config: &Settings) {
        let img_processor =
            Box::new(CpuImageProcessor::default()) as Box<dyn ImageOp + Sync + Send>;

        let (mut img, mut alpha) = RawImage::rgba(img);

        if let Some(upscaler) = config.upscaler.upscaler {
            let (h, w) = (img.height, img.width);

            img = self
                .get_upscaler(upscaler)
                .upscale(
                    &img,
                    config.upscaler.patch_size,
                    config.upscaler.padding,
                    &img_processor,
                )
                .unwrap();
            let (ha, wa) = (img.height, img.width);
            if let Some(a) = alpha {
                let alpha_image =
                    DynamicImage::from(GrayImage::from_raw(w as u32, h as u32, a).unwrap());
                let alpha_image = alpha_image.resize_exact(
                    wa as u32,
                    ha as u32,
                    image::imageops::FilterType::Lanczos3,
                );
                let data = match alpha_image {
                    DynamicImage::ImageLuma8(img) => img.into_raw(),
                    _ => unreachable!("the output from upscaling should be a gray image"),
                };
                alpha = Some(data);
            }
        }

        let (areas, mask) = self
            .get_detector(config.detector.detector)
            .detect(
                &img,
                config.detector.preprocessor,
                config.detector.options,
                &img_processor,
            )
            .unwrap();
        let textlines = self
            .get_ocr(config.ocr.ocr)
            .detect(&img, &areas, &img_processor)
            .unwrap();
        let textblocks = textline_merge::dispatch_main(
            &textlines,
            img.width,
            img.height,
            config.ocr.min_text_length,
            vec![],
            &config.ocr.filter_text,
            &self.lang_detector,
        );
        visualize_textblocks(&img, &textblocks, false);
        // todo: pre-dictionary
        let translator = self.get_translator(config.translator.translator);
        let texts = textblocks
            .iter()
            .filter(|v| !v.skip_translate)
            .map(|v| v.text.clone())
            .collect::<Vec<_>>();
        let translations = if translator.local() {
            translator
                .translator_mut()
                .as_blocking()
                .unwrap()
                .translate_vec(
                    &texts,
                    None,
                    interface_translator::Language::Japanese,
                    &interface_translator::Language::English,
                )
                .unwrap()
        } else {
            translator
                .translator()
                .as_async()
                .unwrap()
                .translate_vec(&texts, None, None, &interface_translator::Language::English)
                .await
                .unwrap()
                .text
        };
        // mask refinement
        // inpainting
        // rgb => rgba
        // rendering
    }
}

fn draw_polyline_mut(canvas: &mut RgbImage, points: &[(i64, i64)], closed: bool, color: Rgb<u8>) {
    if points.len() < 2 {
        return;
    }

    let mut segments = points.windows(2).collect::<Vec<_>>();

    let t = &[points[points.len() - 1], points[0]];
    if closed {
        segments.push(t);
    }

    for segment in segments {
        let (p1, p2) = (segment[0], segment[1]);
        let p1 = (p1.0 as f32, p1.1 as f32);
        let p2 = (p2.0 as f32, p2.1 as f32);
        draw_line_segment_mut(canvas, p1, p2, color);
    }
}

fn visualize_textblocks(canvas: &RawImage, blk_list: &[TextBlock], show_panels: bool) {
    let mut img = canvas.clone().to_image().unwrap();
    if show_panels {
        //             panels_raw = get_panels_from_array(img_rgb, rtl=right_to_left)
        //             panels = [(x, y, x + w, y + h) for x, y, w, h in panels_raw]
        //             # Use the customised sorter that keeps vertically stacked panels together.
        //             panels = _sort_panels_fill(panels, right_to_left)

        //             # Draw panel boxes and order
        //             for panel_idx, (x1, y1, x2, y2) in enumerate(panels):
        //                 cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), lw)  # Magenta color for panels
        //                 # Put panel number inside the box with deep blue color for better visibility and aesthetics
        //                 cv2.putText(canvas, str(panel_idx), (x1+5, y1+60), cv2.FONT_HERSHEY_SIMPLEX,
        //                            lw/2, (200, 100, 0), max(lw-1, 1), cv2.LINE_AA)
    }
    // let font_data =
    // include_bytes!("/System/Library/Fonts/Supplemental/AmericanTypewriter.ttc").to_vec();
    // let font_vec = FontVec::try_from_vec(font_data).unwrap();
    for (i, blk) in blk_list.iter().enumerate() {
        let (bx1, by1, bx2, by2) = blk.xyxy();
        img = imageproc::drawing::draw_hollow_rect(
            &img,
            Rect::at(bx1 as i32, by1 as i32).of_size((bx2 - bx1) as u32, (by2 - by1) as u32),
            Rgb::<u8>::from_slice(&[127, 255, 127]).clone(),
        );
        for (j, line) in blk.lines.iter().enumerate() {
            // draw_text_mut(
            //     &mut img,
            //     Rgb::<u8>::from_slice(&[0, 127, 255]).clone(),
            //     line[0].0 as i32,
            //     line[0].1 as i32,
            //     20.0,
            //     &font_vec,
            //     &j.to_string(),
            // );
            draw_polyline_mut(
                &mut img,
                line,
                false,
                Rgb::<u8>::from_slice(&[255, 127, 0]).clone(),
            );
        }
        for min_rect in blk.min_rect() {
            draw_polyline_mut(
                &mut img,
                &min_rect,
                false,
                Rgb::<u8>::from_slice(&[127, 127, 0]).clone(),
            );
        }

        // draw_text_mut(
        //     &mut img,
        //     Rgb::<u8>::from_slice(&[255, 127, 127]).clone(),
        //     bx1 as i32,
        //     by1 as i32 + 2,
        //     1.0,
        //     &font_vec,
        //     &i.to_string(),
        // );
        let center = [(bx1 + bx2) / 2, (by1 + by2) / 2];

        let angle_text = format!("a: {:.2}", blk.angle);
        let x_text = format!("x: {bx1}");
        let y_text = format!("y: {by1}");

        let center_x = center[0];
        // put_text_with_outline(angle_text, center_x, center[1] - 10)
        // put_text_with_outline(x_text, center_x, center[1] + 15)
        // put_text_with_outline(y_text, center_x, center[1] + 40)
        img.save("bbox.png").unwrap();
    }
}
