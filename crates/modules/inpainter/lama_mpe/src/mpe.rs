use std::sync::Arc;

use fast_image_resize::{images::Image, ResizeAlg, ResizeOptions, SrcCropping};
use interface_image::{ImageOp, Mask, MaskView};
use ndarray::{stack, Array2, Array3, Axis, Zip};
use util::{
    nd::to_raw,
    opencv::{filter_2d, Input},
};

pub fn load_masked_position_encoding(
    mask: MaskView,
    op: &Arc<dyn ImageOp + Send + Sync>,
) -> anyhow::Result<(Array2<i64>, Array3<i64>)> {
    let (ori_h, ori_w) = (mask.height as usize, mask.width as usize);

    let d_filter1 = Input::from_slice_2d(&[[1.0f32, 1., 0.], [1., 1., 0.], [0., 0., 0.]]).unwrap();
    let d2_filter = Input::from_slice_2d(&[[0.0f32, 0., 0.], [1., 1., 0.], [1., 1., 0.]]).unwrap();
    let d3_filter = Input::from_slice_2d(&[[0.0f32, 1., 1.], [0., 1., 1.], [0., 0., 0.]]).unwrap();
    let d4_filter = Input::from_slice_2d(&[[0.0f32, 0., 0.], [0., 1., 1.], [0., 1., 1.]]).unwrap();

    let ones_filter = Input::from_slice_2d(&vec![vec![1.0f32; 3]; 3]).unwrap();
    let str_size = 256;
    let pos_num = 128;

    let ori_mask = mask
        .as_nd()?
        .mapv(|v| if v > 127 { 1.0f32 } else { 0.0f32 });
    let mask = op.resize_mask(
        mask,
        str_size,
        str_size,
        interface_image::Interpolation::Box,
    )?;
    let mask = mask.as_nd()?;
    let (h, w) = (str_size, str_size);
    let mut mask3 = mask.view().mapv(|v| ((v == 0) as u8) as f32);
    let mut i = 0;
    let mut pos: Array2<i32> = Array2::zeros((h, w));
    let mut direct: Vec<Array2<u8>> = vec![Array2::zeros((h, w)); 4];
    if mask3.iter().any(|&v| v > 0.0) {
        while mask3.mapv(|v| 1.0 - v).sum() > 0.0 {
            let mask3_cv = Input::from(&mask3);
            i += 1;
            let mut mask3_ = filter_2d(&mask3_cv, -1, &ones_filter, Default::default()).unwrap();
            let mask3_c = mask3_.mapv(|v| if v > 0.0 { 1.0f32 } else { 0.0 });

            Zip::from(&mut mask3_).and(&mask3).for_each(|a, &b| {
                if *a > 0.0 {
                    *a = 1.0 - b;
                } else {
                    *a -= b;
                }
            });
            Zip::from(&mut pos).and(&mask3_).for_each(|a, &b| {
                if b == 1.0 {
                    *a = i;
                }
            });
            let mut m = filter_2d(&mask3_cv, -1, &d_filter1, Default::default()).unwrap();
            Zip::from(&mut m).and(&mask3).for_each(|a, &b| {
                if *a > 0.0 {
                    *a = 1.0 - b;
                } else {
                    *a -= b;
                }
            });
            Zip::from(&mut direct[0]).and(&m).for_each(|a, &b| {
                if b == 1.0 {
                    *a = 1;
                }
            });

            let mut m = filter_2d(&mask3_cv, -1, &d2_filter, Default::default()).unwrap();
            Zip::from(&mut m).and(&mask3).for_each(|a, &b| {
                if *a > 0.0 {
                    *a = 1.0 - b;
                } else {
                    *a -= b;
                }
            });
            Zip::from(&mut direct[1]).and(&m).for_each(|a, &b| {
                if b == 1.0 {
                    *a = 1;
                }
            });

            let mut m = filter_2d(&mask3_cv, -1, &d3_filter, Default::default()).unwrap();
            Zip::from(&mut m).and(&mask3).for_each(|a, &b| {
                if *a > 0.0 {
                    *a = 1.0 - b;
                } else {
                    *a -= b;
                }
            });
            Zip::from(&mut direct[2]).and(&m).for_each(|a, &b| {
                if b == 1.0 {
                    *a = 1;
                }
            });

            let mut m = filter_2d(&mask3_cv, -1, &d4_filter, Default::default()).unwrap();
            Zip::from(&mut m).and(&mask3).for_each(|a, &b| {
                if *a > 0.0 {
                    *a = 1.0 - b;
                } else {
                    *a -= b;
                }
            });
            Zip::from(&mut direct[3]).and(&m).for_each(|a, &b| {
                if b == 1.0 {
                    *a = 1;
                }
            });
            mask3 = mask3_c;
        }
    }
    let mut rel_pos = pos.mapv(|v| {
        (((v as f32 / (str_size as f32 / 2.0) * pos_num as f32) as i32).clamp(0, pos_num - 1)) as u8
    });
    let direct = if ori_w != w || ori_h != h {
        let mut r = fast_image_resize::Resizer::new();
        let src = Image::from_vec_u8(
            rel_pos.dim().1 as u32,
            rel_pos.dim().0 as u32,
            to_raw(&rel_pos, |v| v.to_vec()),
            fast_image_resize::PixelType::U8,
        )
        .unwrap();
        let mut dst = Image::new(ori_w as u32, ori_h as u32, fast_image_resize::PixelType::U8);
        r.resize(
            &src,
            &mut dst,
            &ResizeOptions {
                algorithm: ResizeAlg::Nearest,
                cropping: SrcCropping::default(),
                mul_div_alpha: false,
            },
        )
        .unwrap();
        let mut rel_pos_ = Array2::from_shape_vec((ori_h, ori_w), dst.into_vec()).unwrap();
        Zip::from(&mut rel_pos_).and(&ori_mask).for_each(|a, &b| {
            if b == 0.0 {
                *a = 0
            }
        });
        rel_pos = rel_pos_;
        let mut directs = vec![];
        for direct in direct {
            let src = Image::from_vec_u8(
                direct.dim().1 as u32,
                direct.dim().0 as u32,
                to_raw(&direct, |v| v.to_vec()),
                fast_image_resize::PixelType::U8,
            )
            .unwrap();
            let mut dst = Image::new(ori_w as u32, ori_h as u32, fast_image_resize::PixelType::U8);
            r.resize(
                &src,
                &mut dst,
                &ResizeOptions {
                    algorithm: ResizeAlg::Nearest,
                    cropping: SrcCropping::default(),
                    mul_div_alpha: false,
                },
            )
            .unwrap();
            let mut direct_ = Array2::from_shape_vec((ori_h, ori_w), dst.into_vec()).unwrap();
            Zip::from(&mut direct_).and(&ori_mask).for_each(|a, &b| {
                if b == 0.0 {
                    *a = 0
                }
            });
            directs.push(direct_.mapv(|v| v as i64));
        }
        stack(
            Axis(2),
            &directs.iter().map(|v| v.view()).collect::<Vec<_>>(),
        )
        .unwrap()
    } else {
        let convert = direct
            .iter()
            .map(|v| v.mapv(|v| v as i64))
            .collect::<Vec<_>>();
        stack(
            Axis(2),
            &convert.iter().map(|v| v.view()).collect::<Vec<_>>(),
        )
        .unwrap()
    };
    Ok((rel_pos.mapv(|v| v as i64), direct))
}
