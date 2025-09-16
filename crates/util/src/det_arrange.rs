use std::sync::Arc;

use anyhow::bail;
use interface_image::{DimType, ImageOp, Interpolation, RawImage, RawImageCow, RawImageView};
use log::info;
use ndarray::{s, stack, Array, Array3, Array4, ArrayView3, ArrayView4, Axis, ShapeError, Zip};
use rayon::prelude::*;

fn square_pad_resize<'a>(
    img: ArrayView3<'a, u8>,
    tgt_size: usize,
    processor: &Arc<dyn ImageOp + Send + Sync>,
) -> anyhow::Result<(RawImageCow<'a>, f64, isize, isize)> {
    let shape = img.shape();
    let (mut h, w) = (shape[0] as isize, shape[1] as isize);
    let mut pad_h: isize = 0;
    let mut pad_w: isize = 0;
    if w < h {
        pad_w = h - w;
    } else if h < w {
        pad_h = w - h;
        h += pad_h;
    }
    let pad_size = tgt_size as isize - h as isize;
    if pad_size > 0 {
        pad_h += pad_size;
        pad_w += pad_size;
    }
    let mut img = RawImageCow::from(img);

    if pad_h > 0 || pad_w > 0 {
        if let RawImageCow::Owned(v) =
            processor.add_border_wh(img.view(), pad_w as DimType, pad_h as DimType)
        {
            img = RawImageCow::Owned(v);
        }
    }

    let down_scale_ratio = tgt_size as f64 / shape[0] as f64;
    assert!(down_scale_ratio <= 1.0);
    if down_scale_ratio < 1.0 {
        img = RawImageCow::Owned(processor.resize(
            img.view(),
            tgt_size as u16,
            tgt_size as u16,
            Interpolation::Bilinear,
        )?);
    }

    Ok((img, down_scale_ratio, pad_h, pad_w))
}

fn stack_vec_to_array4(vec: Vec<ArrayView3<u8>>) -> anyhow::Result<Array4<u8>> {
    Ok(stack(
        Axis(0),
        &vec.iter().map(|a| a.view()).collect::<Vec<_>>(),
    )?)
}

pub fn rearrange_patches(input: Array4<u8>, p_num: usize, transpose: bool) -> Array4<u8> {
    let (total_patches, ph, pw, c) = input.dim();
    assert_eq!(
        total_patches % p_num,
        0,
        "Total patches must be divisible by p_num"
    );
    let pw_num = total_patches / p_num;

    if transpose {
        let mut output = Array::zeros((p_num, pw_num * pw, ph, c));

        for p in 0..p_num {
            for w in 0..pw_num {
                let patch = input.slice(s![p * pw_num + w, .., .., ..]);
                let mut out_slice = output.slice_mut(s![p, w * pw..(w + 1) * pw, .., ..]);

                for i in 0..ph {
                    for j in 0..pw {
                        for k in 0..c {
                            out_slice[[j, i, k]] = patch[[i, j, k]];
                        }
                    }
                }
            }
        }

        output
    } else {
        let mut output = Array::zeros((p_num, ph, pw_num * pw, c));

        for p in 0..p_num {
            for w in 0..pw_num {
                let src = input.slice(s![p * pw_num + w, .., .., ..]);
                let mut dst = output.slice_mut(s![p, .., w * pw..(w + 1) * pw, ..]);
                dst.assign(&src);
            }
        }
        output
    }
}

fn patch2batches(
    patch_lst: Vec<Arc<RawImage>>,
    p_num: usize,
    transpose: bool,
    max_batch_size: usize,
    tgt_size: u32,
    processor: &Arc<dyn ImageOp + Send + Sync>,
) -> anyhow::Result<(Vec<Vec<RawImage>>, Option<f64>, Option<isize>)> {
    let path_lst = patch_lst
        .iter()
        .map(|v| v.as_ndarray())
        .collect::<Result<Vec<_>, _>>()?;
    let path_lst = stack_vec_to_array4(path_lst)?;
    let patch_lst = rearrange_patches(path_lst, p_num, transpose);

    let mut batches: Vec<Vec<_>> = vec![vec![]];
    let mut down_scale_ratio_ = None;
    let mut pad_size_ = None;

    for (_, patch) in patch_lst.outer_iter().enumerate() {
        if batches.last().map(|v| v.len()).unwrap_or_default() >= max_batch_size {
            batches.push(vec![]);
        }
        let (p, down_scale_ratio, pad_h, pad_w) =
            square_pad_resize(patch, tgt_size as usize, processor)?;
        assert_eq!(pad_h, pad_w);
        batches.last_mut().expect("set manually").push(p.to_owned());
        down_scale_ratio_ = Some(down_scale_ratio);
        pad_size_ = Some(pad_h);
        //TODO:
        // if verbose:
        //     cv2.imwrite(f'result/rearrange_{ii}.png', p[..., ::-1])
    }
    Ok((batches, down_scale_ratio_, pad_size_))
}

fn process_arrays(
    db: &ArrayView4<f32>,
    mask: &Array4<f32>,
    tgt_size: usize,
    pad_size: isize,
) -> (Vec<Array3<f32>>, Vec<Array3<f32>>) {
    let batch_size = db.shape()[0];
    let mut db_lst = Vec::with_capacity(batch_size);
    let mut mask_lst = Vec::with_capacity(batch_size);

    if pad_size > 0 {
        let paddb = (db.shape()[3] as f32 / tgt_size as f32 * pad_size as f32).round() as usize;
        let padmsk = (mask.shape()[3] as f32 / tgt_size as f32 * pad_size as f32).round() as usize;

        let results: Vec<_> = db
            .outer_iter()
            .zip(mask.outer_iter())
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(d, m)| {
                let h = d.shape()[1] - paddb;
                let w = d.shape()[2] - paddb;
                let h_m = m.shape()[1] - padmsk;
                let w_m = m.shape()[2] - padmsk;

                (
                    d.slice(ndarray::s![.., ..h, ..w]).to_owned(),
                    m.slice(ndarray::s![.., ..h_m, ..w_m]).to_owned(),
                )
            })
            .collect();

        for (d, m) in results {
            db_lst.push(d);
            mask_lst.push(m);
        }
    } else {
        let results: Vec<_> = db
            .outer_iter()
            .zip(mask.outer_iter())
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(d, m)| (d.to_owned(), m.to_owned()))
            .collect();

        for (d, m) in results {
            db_lst.push(d);
            mask_lst.push(m);
        }
    }

    (db_lst, mask_lst)
}

pub fn extract_patch(img: RawImageView, t: usize, b: usize) -> RawImage {
    let t = t.min(img.height as usize);
    let b = b.min(img.height as usize);

    let rows = b - t;
    let row_size = img.width as usize * 3;
    let start = t * row_size;
    let end = b * row_size;

    let mut data = Vec::with_capacity(end - start);
    data.extend_from_slice(&img.data[start..end]);

    RawImage {
        data,
        width: img.width,
        height: rows as DimType,
        channels: 3,
    }
}

pub fn shoud_rearrange(img: RawImageView, tgt_size: u32) -> bool {
    let (w, h) = (img.width, img.height);

    let (_, w, h) = if h < w { (true, h, w) } else { (false, w, h) };
    let asp_ratio = h as f64 / w as f64;
    let down_scale_ratio = h as f64 / tgt_size as f64;

    down_scale_ratio > 2.5 && asp_ratio > 3.0
}

pub fn det_rearrange_forward<F>(
    img: RawImageView<'_>,
    tgt_size: u32,
    max_batch_size: u8,
    mut dbnet_batch_forward: F,
    processor: &Arc<dyn ImageOp + Send + Sync>,
) -> anyhow::Result<(Array4<f32>, Array4<f32>)>
where
    F: for<'a> FnMut(ArrayView4<'a, u8>) -> anyhow::Result<(Array4<f32>, Array4<f32>)>,
{
    let (w, h) = (img.width, img.height);
    let mut img = RawImageCow::Borrowed(img);

    let (transpose, w, h) = if h < w { (true, h, w) } else { (false, w, h) };

    info!(
        "Input image will be rearranged to square batches before fed into network. Rearranged batches will be saved to result/rearrange_%d.png"
    );

    if transpose {
        img = RawImageCow::Owned(processor.transpose(img.view()));
    }

    let pw_num = (f64::floor(2.0 * tgt_size as f64 / w as f64) as u32).max(2);
    let ph = pw_num * w as u32;
    let patch_size = ph;
    let ph_num = f64::ceil(h as f64 / ph as f64) as u32;
    let ph_step = if ph_num > 1 {
        ((h as u32 - ph) as f64 / (ph_num - 1) as f64) as u32
    } else {
        0
    };

    let p_num = f64::ceil(ph_num as f64 / pw_num as f64) as usize;
    let pad_num = p_num * pw_num as usize - ph_num as usize;
    let total_patches = ph_num as usize + pad_num;

    let mut rel_step_list = Vec::with_capacity(total_patches);
    let mut patch_list = Vec::with_capacity(total_patches);

    let patches_and_steps: Vec<_> = (0..ph_num)
        .into_par_iter()
        .map(|ii| {
            let t = ii * ph_step;
            let b = t + ph;
            let patch = extract_patch(img.view(), t as usize, b as usize);
            let rel_step = t as f64 / h as f64;
            (rel_step, Arc::new(patch))
        })
        .collect();

    for (rel_step, patch) in patches_and_steps {
        rel_step_list.push(rel_step);
        patch_list.push(patch);
    }

    if pad_num > 0 {
        let template = Arc::new(RawImage {
            data: vec![0; patch_list[0].data.len()],
            width: patch_list[0].width,
            height: patch_list[0].height,
            channels: patch_list[0].channels,
        });

        for ii in ph_num..(ph_num + pad_num as u32) {
            let t = ii * ph_step;
            rel_step_list.push(t as f64 / h as f64);
            // allow:clone[arc]
            patch_list.push(template.clone());
        }
    }

    let (batches, _, pad_size) = patch2batches(
        patch_list,
        p_num,
        transpose,
        max_batch_size as usize,
        tgt_size,
        processor,
    )?;

    let batch_results: anyhow::Result<Vec<_>> = batches
        .into_par_iter()
        .map(|batch| {
            let batch_arrays: Result<Vec<_>, ShapeError> =
                batch.iter().map(|v| v.as_ndarray()).collect();
            let batch_array4 = vec_array3_to_array4(batch_arrays?);
            batch_array4
        })
        .collect();

    let pad_size = match pad_size {
        Some(v) => v,
        None => bail!("no pad_size"),
    };
    let batch_results = batch_results?;
    let mut temp = Vec::with_capacity(batch_results.len());
    for v in batch_results {
        temp.push(dbnet_batch_forward(v.view())?);
    }
    let (db_lst, mask_lst): (Vec<_>, Vec<_>) = temp
        .into_par_iter()
        .flat_map(|(db, mask)| process_arrays(&db.view(), &mask, tgt_size as usize, pad_size))
        .unzip();
    let db = unrearrange(
        db_lst,
        transpose,
        2,
        pad_num,
        w as u32,
        h as u32,
        pw_num as usize,
        ph_step as usize,
        patch_size as usize,
        &rel_step_list,
    )?;

    let mask = unrearrange(
        mask_lst,
        transpose,
        1,
        pad_num,
        w as u32,
        h as u32,
        pw_num as usize,
        ph_step as usize,
        patch_size as usize,
        &rel_step_list,
    )?;

    Ok((db, mask))
}

fn vec_array3_to_array4<T: Clone>(arrays: Vec<ArrayView3<T>>) -> anyhow::Result<Array4<T>> {
    if arrays.is_empty() {
        bail!("input is empty");
    }

    Ok(stack(Axis(0), &arrays)?)
}

fn unrearrange(
    patch_lst: Vec<Array3<f32>>,
    transpose: bool,
    channel: usize,
    pad_num: usize,
    widht: u32,
    height: u32,
    pw_num: usize,
    ph_step: usize,
    patch_size: usize,
    rel_step_list: &Vec<f64>,
) -> anyhow::Result<Array4<f32>> {
    let h = *patch_lst[0]
        .shape()
        .last()
        .ok_or(anyhow::anyhow!("patch_lst has no shape"))?;
    let psize = h;
    let step = (ph_step as f64 * psize as f64 / patch_size as f64) as usize;
    let pw = (psize as f64 / pw_num as f64) as usize;
    let h = (pw as f64 / widht as f64 * height as f64) as usize;
    let mut tgtmap: Array3<f32> = Array3::zeros((channel, h, pw));
    let num_patches = patch_lst.len() * pw_num - pad_num;
    for (ii, p) in patch_lst.into_iter().enumerate() {
        let p = if transpose {
            p.permuted_axes([0, 2, 1])
        } else {
            p
        };
        for jj in 0..pw_num {
            let pidx = ii * pw_num + jj;
            let rel_t = rel_step_list[pidx];

            let t = f64::round(rel_t * h as f64) as usize;
            let b = h.min(t + psize);
            let l = jj * pw;
            let r = l + pw;
            let height = b - t;
            let p_slice = p.slice(s![.., 0..height, l..r]);
            let mut tgt_slice = tgtmap.slice_mut(s![.., t..b, ..]);
            Zip::from(&mut tgt_slice)
                .and(&p_slice)
                .for_each(|a, &b| *a += b);

            if pidx > 0 {
                let interleave = psize - step;
                let end = t + interleave;
                tgtmap
                    .slice_mut(s![.., t..end, ..])
                    .mapv_inplace(|x| x / 2.0);
            }
            if pidx >= num_patches - 1 {
                break;
            }
        }
    }
    let tgtmap = if transpose {
        tgtmap.permuted_axes([0, 2, 1])
    } else {
        tgtmap
    };
    Ok(tgtmap.insert_axis(Axis(0)))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use interface_image::{CpuImageProcessor, ImageOp, RawImage};
    use ndarray::{Array4, ArrayView4};

    use crate::det_arrange::det_rearrange_forward;

    #[test]
    fn find_vertical() {
        let img = RawImage::new("./imgs/01_1-optimized.png").expect("couldnt load npy file");
        let cpu = Arc::new(CpuImageProcessor::default()) as Arc<dyn ImageOp + Send + Sync>;
        let (db, mask) = det_rearrange_forward(img.view(), 2048, 4, mocking, &cpu).expect("failed");
        let ex_db: Array4<f32> =
            ndarray_npy::read_npy("npys/db2_v.npy").expect("couldnt load npy file");
        let ex_mask: Array4<f32> =
            ndarray_npy::read_npy("npys/mask2_v.npy").expect("couldnt load npy file");
        assert_eq!(db, ex_db);
        assert_eq!(mask, ex_mask);
    }

    #[test]
    fn find_horizontal() {
        let img = RawImage::new("./imgs/01_1-optimized.png").expect("couldnt load npy file");
        let cpu = Arc::new(CpuImageProcessor::default()) as Arc<dyn ImageOp + Send + Sync>;
        let img = cpu.rotate_right(img.view());
        let (db, mask) =
            det_rearrange_forward(img.view(), 2048, 4, mocking2, &cpu).expect("failed");
        let ex_db: Array4<f32> =
            ndarray_npy::read_npy("npys/db2_h.npy").expect("couldnt load npy file");
        let ex_mask: Array4<f32> =
            ndarray_npy::read_npy("npys/mask2_h.npy").expect("couldnt load npy file");
        assert_eq!(db.shape(), ex_db.shape());

        assert_eq!(db, ex_db);
        assert_eq!(mask.shape(), ex_mask.shape());

        assert_eq!(mask, ex_mask);
    }

    fn mocking(input: ArrayView4<u8>) -> anyhow::Result<(Array4<f32>, Array4<f32>)> {
        let input_ex: Array4<u8> =
            ndarray_npy::read_npy("npys/input.npy").expect("couldnt load npy file");
        assert_eq!(input.shape(), input_ex.shape());

        assert_eq!(input, input_ex);
        let db: Array4<f32> =
            ndarray_npy::read_npy("npys/db_v.npy").expect("couldnt load npy file");
        let mask: Array4<f32> =
            ndarray_npy::read_npy("npys/mask_v.npy").expect("couldnt load npy file");
        Ok((db, mask))
    }
    fn mocking2(input: ArrayView4<u8>) -> anyhow::Result<(Array4<f32>, Array4<f32>)> {
        let input_ex: Array4<u8> =
            ndarray_npy::read_npy("npys/input_h.npy").expect("couldnt load npy file");
        assert_eq!(input.shape(), input_ex.shape());

        assert_eq!(input, input_ex);
        let db: Array4<f32> =
            ndarray_npy::read_npy("npys/db_h.npy").expect("couldnt load npy file");
        let mask: Array4<f32> =
            ndarray_npy::read_npy("npys/mask_h.npy").expect("couldnt load npy file");
        Ok((db, mask))
    }
}
