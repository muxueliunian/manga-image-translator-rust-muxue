use base_util::error::PreProcessingError;
use interface_image::RawImage;
use ndarray::{Array2, Array3, Axis};
use opencv::core::{DataType, Mat, MatTraitConst, Point, ToInputArray, BORDER_DEFAULT};

use crate::nd::to_raw;

//TODO: refactor + test + bench
pub fn bilateral_filter(
    src: &impl ToInputArray,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
    border_type: i32,
) -> Result<RawImage, PreProcessingError> {
    let mut filtered = Mat::default();
    opencv::imgproc::bilateral_filter(
        src,
        &mut filtered,
        d,
        sigma_color,
        sigma_space,
        border_type,
    )?;
    Ok(filtered.try_into()?)
}

pub struct Input(Mat);

impl Input {
    pub fn from_slice_2d<T: DataType>(s: &[impl AsRef<[T]>]) -> Result<Self, PreProcessingError> {
        Ok(Input(Mat::from_slice_2d(s)?))
    }
}

impl From<Mat> for Input {
    fn from(mat: Mat) -> Self {
        Input(mat)
    }
}

impl From<&Array2<f32>> for Input {
    fn from(value: &Array2<f32>) -> Self {
        let v = to_raw(value, |s| {
            Mat::from_slice(s)
                .unwrap()
                .reshape(1, value.shape()[0] as i32)
                .unwrap()
                .clone_pointee()
        });
        Input(v)
    }
}

pub fn filter_2d(
    src: &Input,
    depth: i32,
    kernel: &Input,
    options: Filter2DOptions,
) -> Result<Array2<f32>, PreProcessingError> {
    let mut m = Mat::default();
    opencv::imgproc::filter_2d(
        &src.0,
        &mut m,
        depth,
        &kernel.0,
        options.anchor,
        options.delta,
        options.border_type,
    )?;
    Ok(convert_to_nd(m)?.remove_axis(Axis(2)))
}

pub fn convert<T, DT>(
    m: Mat,
    convert: fn(usize, usize, usize, &[DT]) -> Result<T, PreProcessingError>,
) -> Result<T, PreProcessingError> {
    let m = if m.is_continuous() { m } else { m.clone() };

    let size = m.size()?;
    let rows = size.height as usize;
    let cols = size.width as usize;
    let channels = m.channels() as usize;

    let total_len = rows * cols * channels;
    let data: &[DT] = unsafe { std::slice::from_raw_parts(m.data() as *const DT, total_len) };
    convert(rows, cols, channels, data)
}

pub fn convert_to_nd<T: Clone>(m: Mat) -> Result<Array3<T>, PreProcessingError> {
    fn nd<T: Clone>(
        rows: usize,
        cols: usize,
        channels: usize,
        data: &[T],
    ) -> Result<Array3<T>, PreProcessingError> {
        let rows = rows;
        let cols = cols;
        let channels = channels;
        let data = data;
        Ok(Array3::from_shape_vec(
            (rows, cols, channels),
            data.to_vec(),
        )?)
    }

    convert(m, nd::<T>)
}

pub struct Filter2DOptions {
    pub anchor: Point,
    pub delta: f64,
    pub border_type: i32,
}

impl Default for Filter2DOptions {
    fn default() -> Self {
        Self {
            anchor: Point::new(-1, -1),
            delta: 0.0,
            border_type: BORDER_DEFAULT,
        }
    }
}
