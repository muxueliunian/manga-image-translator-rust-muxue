use std::path::PathBuf;

use interface_detector::textlines::Quadrilateral;
use interface_image::RawImage;
use opencv::{
    core::{Point, Scalar, Vector},
    imgproc::{polylines, LINE_8},
};

pub fn render_bboxes(img: &RawImage, qu: &[Quadrilateral], path: &PathBuf) {
    let mut img = img.as_opencv_mat().unwrap();
    for q in qu {
        let pts = q
            .pts()
            .iter()
            .map(|v| Point::new(v.x as i32, v.y as i32))
            .collect::<Vector<Point>>();
        polylines(
            &mut img,
            &pts,
            true,
            Scalar::new(255.0, 0.0, 0.0, 255.0),
            2,
            LINE_8,
            0,
        )
        .unwrap();
    }
    RawImage::try_from(img)
        .unwrap()
        .to_image()
        .unwrap()
        .save(path.join("1_bboxes_unfiltered.png"))
        .unwrap()
}
