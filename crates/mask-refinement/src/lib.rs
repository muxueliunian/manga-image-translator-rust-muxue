mod bubble;
mod expand;

use std::{borrow::Cow, i32, sync::Arc};

use interface_detector::textlines::{BBox, MyPoint, Quadrilateral};
use interface_image::{ImageOp, Mask, RawImage};
use opencv::{
    core::{
        bitwise_and, no_array, Mat, MatTraitConst, Point, Scalar, Vector, BORDER_CONSTANT, CV_8UC1,
    },
    imgproc::{
        bounding_rect, dilate, draw_contours, find_contours, morphology_default_border_value,
        rectangle, CHAIN_APPROX_SIMPLE, LINE_8, RETR_EXTERNAL,
    },
};
use textline_merge::TextBlock;

use crate::{
    bubble::is_ignore,
    expand::{
        expand_right_quad, expand_right_to_connect, expand_top_quad, expand_top_to_connect,
        shrink_quad_right, shrink_quad_top,
    },
};

pub enum Method {
    FitText,
    FillMask,
}

pub fn expand(furi: bool, lines: &[[MyPoint; 4]], mask: &Mask) -> Vec<Quadrilateral> {
    let mut out = Vec::with_capacity(lines.len());
    let mut lines = lines.iter().rev().peekable();
    while let Some(line) = lines.next() {
        let line_ = Quadrilateral::new2(line.to_vec(), 0.0);
        if furi {
            let peek = lines.peek();
            match line_.vertical() {
                true => out.push(Quadrilateral::new2(
                    match peek {
                        Some(n) => expand_right_to_connect(line, n),
                        None => shrink_quad_right(expand_right_quad(*line, 2.0), mask),
                    }
                    .to_vec(),
                    0.0,
                )),
                false => out.push(Quadrilateral::new2(
                    match peek {
                        Some(n) => expand_top_to_connect(line, n),
                        None => shrink_quad_top(expand_top_quad(*line, 2.0), mask),
                    }
                    .to_vec(),
                    0.0,
                )),
            }
        } else {
            out.push(line_);
        }
    }
    out
}
pub fn dispatch(
    text_regions: &[TextBlock],
    raw_img: &RawImage,
    raw_mask: &Mask,
    method: Method,
    ignore_bubble: u8,
    furi: bool,
    image_op: &Arc<dyn ImageOp + Send + Sync>,
) -> Mask {
    let raw_mask = if furi {
        Cow::Owned(image_op.resize_mask(
            raw_mask.clone(),
            raw_img.width as usize,
            raw_img.height as usize,
            interface_image::Interpolation::Nearest,
        ))
    } else {
        Cow::Borrowed(raw_mask)
    };
    let size = (raw_img.width, raw_img.height);
    // let scale_factor = ((raw_mask.height as f64 - raw_mask.height as f64 / 3.0)
    //     / raw_mask.height as f64)
    //     .clamp(0.5, 1.0);
    // img_resized = cv2.resize(raw_image, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)
    // mask_resized = cv2.resize(raw_mask, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)

    // mask_resized[mask_resized > 0] = 255
    let textlines = text_regions
        .iter()
        .flat_map(|v| expand(furi, &v.lines, raw_mask.as_ref()))
        .collect::<Vec<_>>();

    let final_mask = match method {
        Method::FitText => {
            // complete_mask(
            //     img_resized,
            //     mask_resized,
            //     textlines,
            //     dilation_offset = dilation_offset,
            //     kernel_size = kernel_size,
            // );
            todo!()
        }
        Method::FillMask => complete_mask_fill(
            (size.0 as i64, size.1 as i64),
            textlines.iter().map(|v| v.aabb()).collect::<Vec<_>>(),
        ),
    };

    if ignore_bubble < 1 || ignore_bubble > 50 {
        return Mask {
            width: size.0,
            height: size.1,
            data: final_mask,
        };
    }
    let kernel_size = (size.0.max(size.1) as f64 * 0.025) as usize;
    let ones = vec![1_u8; kernel_size * kernel_size];
    let kernel = Mat::from_slice(&ones).unwrap();
    let kernel = kernel.reshape(1, kernel_size as i32).unwrap();
    let final_mask = Mat::from_slice(&final_mask).unwrap();
    let final_mask = final_mask.reshape(1, size.1 as i32).unwrap();
    let mut new_final_mask = Mat::default();
    dilate(
        &final_mask,
        &mut new_final_mask,
        &kernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        morphology_default_border_value().unwrap(),
    )
    .unwrap();
    let mut final_mask = new_final_mask;

    let mut contours = Vector::<Vector<Point>>::new();

    find_contours(
        &final_mask,
        &mut contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE,
        Point::default(),
    )
    .unwrap();
    for cnt in contours {
        let mut temp_mask = Mat::new_rows_cols_with_default(
            size.1 as i32,
            size.0 as i32,
            CV_8UC1,
            Scalar::all(0.0),
        )
        .unwrap();
        // # rect min
        let rec = bounding_rect(&cnt).unwrap();
        rectangle(&mut temp_mask, rec, Scalar::all(255.0), -1, LINE_8, 0).unwrap();
        // get textblock
        let raw_image = raw_img.as_opencv_mat().unwrap();
        let mut textblock = Mat::default();
        bitwise_and(&raw_image, &raw_image, &mut textblock, &temp_mask).unwrap();
        if is_ignore(size, &textblock, ignore_bubble) {
            draw_contours(
                &mut final_mask,
                &vec![cnt].into_iter().collect::<Vector<Vector<Point>>>(),
                -1,
                Scalar::all(0.0),
                -1,
                LINE_8,
                &no_array(),
                i32::MAX,
                Point::default(),
            )
            .unwrap();
        }
    }
    Mask::from(final_mask)
}

fn complete_mask_fill(size: (i64, i64), aabbs: Vec<BBox>) -> Vec<u8> {
    let (width, height) = size;
    let mut mask = vec![0u8; (width * height) as usize];

    for aabb in aabbs {
        let x0 = aabb.x.min(width);
        let y0 = aabb.y.min(height);
        let x1 = (aabb.w + aabb.x).min(width);
        let y1 = (aabb.h + aabb.y).min(height);

        for y in y0..y1 {
            let row_start = (y * width) as usize;
            for x in x0..x1 {
                mask[row_start + x as usize] = 255;
            }
        }
    }

    mask
}
