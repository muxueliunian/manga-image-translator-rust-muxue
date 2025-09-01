use image::{DynamicImage, GenericImageView as _, RgbImage};
use interface_detector::textlines::{MyPoint, Quadrilateral};
use interface_image::RawImage;
use opencv::{
    calib3d::{find_homography, RANSAC},
    core::{
        no_array, rotate, Mat, Point2f, Scalar, Size, Vector, BORDER_CONSTANT,
        ROTATE_90_COUNTERCLOCKWISE,
    },
    imgproc::INTER_LINEAR,
};

pub fn get_transformed_region(q: &Quadrilateral, img: &RgbImage, text_height: u32) -> Mat {
    let [l1a, l1b, l2a, l2b] = <[MyPoint<f64>; 4]>::try_from(
        q.structure()
            .into_iter()
            .map(|v| v.to_f64())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let im_w = img.width() as i64;
    let im_h = img.height() as i64;
    let v_vec = l1b - l1a;
    let h_vec = l2b - l2a;
    let aabb = q.xyxy();
    let x1 = aabb.0.clamp(0, im_w);
    let y1 = aabb.1.clamp(0, im_h);
    let x2 = aabb.2.clamp(0, im_w);
    let y2 = aabb.3.clamp(0, im_h);
    let ratio = v_vec.norm() / h_vec.norm();

    // cv2.warpPerspective could overflow if image size is too large, better crop it here
    let img_croped = RawImage::from(DynamicImage::from(
        img.view(
            x1 as u32,
            y1 as u32,
            x2 as u32 - x1 as u32,
            y2 as u32 - y1 as u32,
        )
        .to_image(),
    ));

    let img_croped = img_croped.as_opencv_mat().unwrap();
    let src_points = q
        .pts()
        .iter()
        .map(|v| Point2f::new((v.x - x1) as f32, (v.y - y1) as f32))
        .collect::<Vector<Point2f>>();

    let w = text_height.max(2);
    let h = ((text_height as f64 * ratio).round() as u32).max(2);
    let dst_points = [
        Point2f::new(0.0, 0.0),
        Point2f::new(w as f32 - 1.0, 0.0),
        Point2f::new(w as f32 - 1.0, h as f32 - 1.0),
        Point2f::new(0.0, h as f32 - 1.0),
    ]
    .into_iter()
    .collect::<Vector<Point2f>>();
    let m = find_homography(&src_points, &dst_points, &mut no_array(), RANSAC, 5.0).unwrap();
    let mut region = Mat::default();
    opencv::imgproc::warp_perspective(
        &img_croped,
        &mut region,
        &m,
        Size::new(w as i32, h as i32),
        INTER_LINEAR,
        BORDER_CONSTANT,
        Scalar::default(),
    )
    .unwrap();
    if q.vertical() {
        let mut reg = Mat::default();
        rotate(&region, &mut reg, ROTATE_90_COUNTERCLOCKWISE).unwrap();
        region = reg;
    }

    region
}
