use std::path::PathBuf;

use interface_image::RawImage;
use opencv::{
    core::{Point, Rect, Scalar, Vector},
    imgproc::{polylines, put_text, rectangle, FONT_HERSHEY_SIMPLEX, LINE_8, LINE_AA},
};
use textline_merge::TextBlock;

pub fn render_textblocks(img: &RawImage, blk: &[TextBlock], path: &PathBuf) -> anyhow::Result<()> {
    let lw = (img.as_ndarray().unwrap().shape().iter().sum::<usize>() as f64 / 2.0 * 0.003).max(2.0)
        as u32;
    let mut canvas = img.as_opencv_mat()?.clone_pointee();
    let color5 = Scalar::new(255.0, 127.0, 127.0, 0.0);
    let color4 = Scalar::new(255.0, 127.0, 0.0, 0.0);
    let color1 = Scalar::new(127.0, 255.0, 127.0, 0.0);
    let color3 = Scalar::new(127.0, 127.0, 0.0, 0.0);
    let color2 = Scalar::new(0.0, 127.0, 255.0, 0.0);
    for (i, blk) in blk.iter().enumerate() {
        let (bx1, by1, bx2, by2) = blk.xyxy();
        let rec = Rect::new(
            bx1 as i32,
            by1 as i32,
            bx2 as i32 - bx1 as i32,
            by2 as i32 - by1 as i32,
        );

        rectangle(&mut canvas, rec, color1, lw as i32, LINE_8, 0)?;
        for (j, line) in blk.lines.iter().enumerate() {
            let pts = line
                .iter()
                .map(|v| Point::new(v.x as i32, v.y as i32))
                .collect::<Vector<Point>>();
            put_text(
                &mut canvas,
                &j.to_string(),
                pts.get(0)?,
                FONT_HERSHEY_SIMPLEX,
                0.7,
                color4,
                1,
                LINE_8,
                false,
            )?;

            polylines(&mut canvas, &pts, true, color2, 2, LINE_8, 0)?;
        }
        let min_rect = blk.min_rect()?;
        let pts = min_rect
            .iter()
            .map(|v| Point::new(v.0 as i32, v.1 as i32))
            .collect::<Vector<Point>>();
        put_text(
            &mut canvas,
            &i.to_string(),
            Point::new(bx1 as i32, by1 as i32 + lw as i32),
            FONT_HERSHEY_SIMPLEX,
            lw as f64 / 3.0,
            color5,
            1.max(lw as i32 - 1),
            LINE_AA,
            false,
        )?;

        polylines(&mut canvas, &pts, true, color3, 2, LINE_8, 0)?;
    }
    RawImage::try_from(canvas)?
        .to_image()?
        .save(path.join("3_bboxes.png"))?;
    Ok(())
}
