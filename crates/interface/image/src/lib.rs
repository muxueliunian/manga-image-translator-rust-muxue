use std::{
    fmt::{self},
    path::Path,
};

mod cpu;
pub mod dummy;
mod froms;
#[cfg(feature = "gpu")]
mod gpu;
mod rayon;

pub use cpu::CpuImageProcessor;
#[cfg(feature = "gpu")]
pub use gpu::GpuImageProcessor;
use image::{DynamicImage, RgbaImage};
use ndarray::{Array, Array2, Dim};
use opencv::core::{Mat, MatTraitConst as _};
pub use rayon::RayonImageProcessor;

#[cfg(feature = "debug")]
use crate::detectors::textlines::Quadrilateral;
#[cfg(feature = "u16-dims")]
pub type DimType = u16;
#[cfg(not(feature = "u16-dims"))]
pub type DimType = u32;

#[derive(PartialEq, Eq, Clone)]
/// A rgb image
pub struct RawImage {
    pub data: Vec<u8>,
    pub width: DimType,
    pub height: DimType,
    /// Always 3
    pub channels: u8,
}

fn blend_pixel3(s_rgb: [u8; 3], o_rgba: [u8; 4]) -> [u8; 3] {
    let alpha = o_rgba[3] as f32 / 255.0;
    [
        ((o_rgba[0] as f32 * alpha) + (s_rgb[0] as f32 * (1.0 - alpha))).round() as u8,
        ((o_rgba[1] as f32 * alpha) + (s_rgb[1] as f32 * (1.0 - alpha))).round() as u8,
        ((o_rgba[2] as f32 * alpha) + (s_rgb[2] as f32 * (1.0 - alpha))).round() as u8,
    ]
}

fn blend_pixel4(s_rgba: [u8; 4], o_rgba: [u8; 4]) -> [u8; 4] {
    if o_rgba[3] == 255 {
        return o_rgba;
    }

    let sa = s_rgba[3] as f32 / 255.0;
    let oa = o_rgba[3] as f32 / 255.0;

    let out_a = oa + sa * (1.0 - oa);

    let out_r =
        ((o_rgba[0] as f32 * oa + s_rgba[0] as f32 * sa * (1.0 - oa)) / out_a).round() as u8;
    let out_g =
        ((o_rgba[1] as f32 * oa + s_rgba[1] as f32 * sa * (1.0 - oa)) / out_a).round() as u8;
    let out_b =
        ((o_rgba[2] as f32 * oa + s_rgba[2] as f32 * sa * (1.0 - oa)) / out_a).round() as u8;
    let out_a = (out_a * 255.0).round() as u8;

    [out_r, out_g, out_b, out_a]
}

impl RawImage {
    pub fn add_a(mut self, a: Vec<u8>) -> Self {
        let mut result = Vec::with_capacity(self.data.len() + a.len());
        let chunks = self.data.chunks(self.channels as usize).collect::<Vec<_>>();
        assert_eq!(chunks.len(), a.len());
        for (chunk, &val) in chunks.iter().zip(a.iter()) {
            result.extend_from_slice(chunk);
            result.push(val);
        }
        self.data = result;
        self
    }

    pub unsafe fn set_rgba_pixel(&mut self, x: u16, y: u16, rgba: [u8; 4]) {
        let idx = (y as usize * self.width as usize + x as usize) * self.channels as usize;
        let ptr = self.data.as_mut_ptr().add(idx);

        *ptr.add(0) = rgba[0];
        *ptr.add(1) = rgba[1];
        *ptr.add(2) = rgba[2];
        *ptr.add(3) = rgba[3];
    }

    pub unsafe fn set_rgb_pixel(&mut self, x: u16, y: u16, rgb: [u8; 3]) {
        let idx = (y as usize * self.width as usize + x as usize) * self.channels as usize;
        let ptr = self.data.as_mut_ptr().add(idx);

        *ptr.add(0) = rgb[0];
        *ptr.add(1) = rgb[1];
        *ptr.add(2) = rgb[2];
    }

    pub fn apply_patch(
        &mut self,
        patch: &Self, // `other` is the patch
        x: u16,
        y: u16,
    ) {
        assert!(x + patch.width <= self.width);
        assert!(y + patch.height <= self.height);
        assert!(self.channels >= 3);

        let use_rgba = self.channels % 2 == 0;

        for j in 0..patch.height {
            for i in 0..patch.width {
                if use_rgba {
                    let s_rgba = self.rgba_pixel(x + i, y + j);
                    let p_rgba = patch.rgba_pixel(i, j);
                    let blended = if p_rgba[3] != 255 {
                        blend_pixel4(s_rgba, p_rgba)
                    } else {
                        p_rgba
                    };
                    unsafe {
                        self.set_rgba_pixel(x + i, y + j, blended);
                    }
                } else {
                    let p_rgb = patch.rgb_pixel(i, j);
                    unsafe {
                        self.set_rgb_pixel(x + i, y + j, p_rgb);
                    }
                }
            }
        }
    }

    pub fn apply(self, other: Self) -> Self {
        assert_eq!(self.height, other.height);
        assert_eq!(self.width, other.width);
        let a = (self.channels == 4 || self.channels == 2)
            && (other.channels == 4 || other.channels == 2);
        assert!(a);
        let mut out = Vec::with_capacity(self.width as usize * self.height as usize);
        for h in 0..self.height {
            for w in 0..self.width {
                let s_rgba = self.rgba_pixel(w, h);
                let o_rgba = other.rgba_pixel(w, h);
                let p = blend_pixel4(s_rgba, o_rgba);
                out.push(p);
            }
        }
        let len = out.len();
        let capacity = out.capacity();

        let ptr = out.as_ptr() as *mut u8;
        std::mem::forget(out);

        Self {
            data: unsafe { Vec::from_raw_parts(ptr, len * 4, capacity * 4) },
            width: self.width,
            height: self.height,
            channels: 4,
        }
    }

    pub fn rgb_pixel(&self, x: DimType, y: DimType) -> [u8; 3] {
        if self.channels == 1 {
            let b = self.width as usize * y as usize + x as usize;
            let b = self.data[b];
            return [b, b, b];
        } else if self.channels == 3 {
            let b = (self.width as usize * y as usize + x as usize) * 3;
            return [self.data[b], self.data[b + 1], self.data[b + 2]];
        } else {
            unimplemented!("not valid shape")
        }
    }

    pub fn rgba_pixel(&self, x: DimType, y: DimType) -> [u8; 4] {
        if self.channels == 1 {
            let b = self.width as usize * y as usize + x as usize;
            let b = self.data[b];
            return [b, b, b, 255];
        } else if self.channels == 2 {
            let b = (self.width as usize * y as usize + x as usize) * 2;
            let c = self.data[b];
            let a = self.data[b + 1];
            return [c, c, c, a];
        } else if self.channels == 3 {
            let b = (self.width as usize * y as usize + x as usize) * 3;
            return [self.data[b], self.data[b + 1], self.data[b + 2], 255];
        } else if self.channels == 4 {
            let b = (self.width as usize * y as usize + x as usize) * 4;
            return [
                self.data[b],
                self.data[b + 1],
                self.data[b + 2],
                self.data[b + 3],
            ];
        } else {
            unimplemented!("not valid shape")
        }
    }

    pub fn _rgba(img: RgbaImage) -> (Self, Vec<u8>) {
        let v: (Vec<_>, Vec<_>) = img.pixels().map(|v| (&v.0[..3], v.0[3])).unzip();
        let data = v.0.concat();
        let alpha = v.1;
        (
            RawImage {
                data,
                width: img.width() as DimType,
                height: img.height() as DimType,
                channels: 3,
            },
            alpha,
        )
    }
    pub fn rgba(img: DynamicImage) -> (Self, Option<Vec<u8>>) {
        match img {
            DynamicImage::ImageRgba8(img) => {
                let (img, alpha) = Self::_rgba(img);
                (img, Some(alpha))
            }
            DynamicImage::ImageRgba16(img) => {
                let (img, alpha) = Self::_rgba(DynamicImage::from(img).to_rgba8());
                (img, Some(alpha))
            }
            DynamicImage::ImageRgba32F(img) => {
                let (img, alpha) = Self::_rgba(DynamicImage::from(img).to_rgba8());
                (img, Some(alpha))
            }
            img => (RawImage::from(img), None),
        }
    }
    pub fn url(url: &str) -> anyhow::Result<Self> {
        let mut img = ureq::get(url).call()?;
        let body = img.body_mut();
        let img = body.read_to_vec()?;
        let img = image::load_from_memory(&img)?.to_rgb8();
        Ok(RawImage {
            width: img.width() as DimType,
            height: img.height() as DimType,
            channels: 3,
            data: img.into_raw(),
        })
    }
    pub fn channels(&self) -> Vec<Vec<u8>> {
        let count = self.width as usize * self.height as usize;
        let mut r = Vec::with_capacity(count);
        let mut g = Vec::with_capacity(count);
        let mut b = Vec::with_capacity(count);

        for chunk in self.data.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        vec![r, g, b]
    }
}

#[derive(Clone)]
pub struct Mask {
    pub width: DimType,
    pub height: DimType,
    pub data: Vec<u8>,
}

impl Mask {
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[x + y * self.width as usize]
    }
    pub fn as_opencv_mat<'a>(&'a self) -> Result<Mat, opencv::Error> {
        let mat = Mat::from_slice(&self.data)?;
        let mat = mat.reshape(1, self.height as i32)?.clone_pointee();
        Ok(mat)
    }

    pub fn as_nd(&self) -> Array2<u8> {
        Array2::from_shape_vec(
            (self.height as usize, self.width as usize),
            self.data.clone(),
        )
        .unwrap()
    }
}

#[cfg(feature = "debug")]
impl RawImage {
    pub fn draw_bbox(&mut self, textlines: &[Quadrilateral]) -> Result<(), &'static str> {
        use tiny_skia::{Color, Paint, Pixmap, Stroke};

        let rgb_img = self
            .clone()
            .to_image()
            .ok_or("Failed to convert to image")?;
        let mut pixmap =
            Pixmap::new(self.width as u32, self.height as u32).ok_or("Failed to create Pixmap")?;

        for (x, y, pixel) in rgb_img.enumerate_pixels() {
            let i = (y * self.width as u32 + x) as usize;
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];
            pixmap.pixels_mut()[i] = tiny_skia::PremultipliedColorU8::from_rgba(r, g, b, 255)
                .expect("Alpha needs to be >= rgb");
        }

        let mut paint = Paint::default();
        paint.set_color(
            Color::from_rgba(1.0, 0.0, 0.0, 1.0)
                .expect("rbga values need to be in range from 0 to 1"),
        );
        let stroke = Stroke {
            width: 2.0,
            ..Default::default()
        };

        for txt in textlines {
            use tiny_skia::PathBuilder;

            let mut pb = PathBuilder::new();
            if let Some(&(x0, y0)) = txt.pts().first() {
                pb.move_to(x0 as f32, y0 as f32);
                for &(x, y) in &txt.pts()[1..] {
                    pb.line_to(x as f32, y as f32);
                }
                pb.close();
                let path = pb.finish().ok_or("invalid path")?;
                pixmap.stroke_path(
                    &path,
                    &paint,
                    &stroke,
                    tiny_skia::Transform::identity(),
                    None,
                );
            }
        }
        self.data = pixmap
            .data()
            .chunks(4)
            .flat_map(|v| &v[..3])
            .cloned()
            .collect();
        Ok(())
    }

    pub fn display(&self) -> anyhow::Result<()> {
        use show_image::{create_window, ImageView};
        let window = create_window("Image", Default::default())?;

        let image = ImageView::new(
            show_image::ImageInfo::rgb8(self.width as u32, self.height as u32),
            &self.data,
        );

        window.set_image("frame-0", image)?;

        println!("Press Enter to close the window...");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        Ok(())
    }
}

impl fmt::Debug for RawImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawImage")
            .field("data_len", &self.data.len())
            .field("width", &self.width)
            .field("height", &self.height)
            .field("channels", &self.channels)
            .finish()
    }
}

impl Mask {
    pub fn to_image(self) -> Option<image::GrayImage> {
        #[cfg(feature = "u16-dims")]
        return image::GrayImage::from_raw(self.width as u32, self.height as u32, self.data);
        #[cfg(not(feature = "u16-dims"))]
        image::GrayImage::from_raw(self.width, self.height, self.data)
    }
}

impl RawImage {
    pub fn to_ndarray(self) -> Result<Array<u8, Dim<[usize; 3]>>, ndarray::ShapeError> {
        Array::from_shape_vec(
            Dim([
                self.height as usize,
                self.width as usize,
                self.channels as usize,
            ]),
            self.data.clone(),
        )
    }

    pub fn as_opencv_mat<'a>(&'a self) -> Result<Mat, opencv::Error> {
        let mat = Mat::from_slice(&self.data)?;
        let mat = mat
            .reshape(self.channels as i32, self.height as i32)?
            .clone_pointee();
        Ok(mat)
    }

    pub fn to_image(self) -> Option<DynamicImage> {
        match self.channels == 4 {
            true => {
                let rgba =
                    image::RgbaImage::from_raw(self.width as u32, self.height as u32, self.data)
                        .unwrap();
                Some(DynamicImage::ImageRgba8(rgba))
            }
            false => {
                let rgb =
                    image::RgbImage::from_raw(self.width as u32, self.height as u32, self.data)
                        .unwrap();
                Some(DynamicImage::ImageRgb8(rgb))
            }
        }
    }

    pub fn new<P: AsRef<Path>>(path: P) -> anyhow::Result<RawImage> {
        let v = path.as_ref();
        let path = if v.is_relative() {
            base_util::project::root_path().join(v)
        } else {
            v.to_path_buf()
        };

        let img = image::open(path)?;

        let rgb_img = img.to_rgb8();

        let (width, height) = rgb_img.dimensions();
        let data = rgb_img.into_raw();
        #[cfg(feature = "u16-dims")]
        let (width, height) = (width as u16, height as u16);
        Ok(RawImage {
            data,
            width,
            height,
            channels: 3,
        })
    }
}

pub trait ImageOp {
    fn invert(&self, image: RawImage) -> RawImage;
    fn add_border(&self, image: RawImage, target_side_length: DimType) -> RawImage {
        self.add_border_wh(image, target_side_length, target_side_length)
    }
    fn add_border_wh(&self, image: RawImage, width: DimType, height: DimType) -> RawImage;
    fn add_border_center(&self, image: RawImage, target_side_length: DimType) -> RawImage;
    fn add_border_center_wh(&self, image: RawImage, twidth: DimType, height: DimType) -> RawImage;
    fn remove_border(&self, image: RawImage, width: DimType, height: DimType) -> RawImage;
    fn remove_border_center(&self, image: RawImage, width: DimType, height: DimType) -> RawImage;
    fn rotate_right(&self, image: RawImage) -> RawImage;
    fn rotate_left(&self, image: RawImage) -> RawImage;
    fn rotate_left_mask(&self, mask: Mask) -> Mask;
    fn gamma_correction(&self, image: RawImage) -> RawImage;
    fn histogram_equalization(&self, image: RawImage) -> RawImage;
    fn transpose(&self, image: RawImage) -> RawImage;
    fn resize(
        &self,
        image: RawImage,
        width: DimType,
        height: DimType,
        interpolation: Interpolation,
    ) -> RawImage;
    fn resize_mask(
        &self,
        image: Mask,
        width: usize,
        height: usize,
        interpolation: Interpolation,
    ) -> Mask;

    fn remove_border_mask(&self, mask: Mask, width: DimType, height: DimType) -> Mask;
    fn bgr_to_rgb(&self, img: RawImage) -> RawImage;
}
pub enum Interpolation {
    Nearest,
    Box,
    Bilinear,
    BilinearExact,
    Bicubic,
    Lanczos3,
}

pub fn generate_patches_m(img: RawImage, patch_size: usize, margin: usize) -> Vec<RawImage> {
    let p = margin;
    let total_size = patch_size + 2 * p;
    let n_x = (img.width as usize + patch_size - 1) / patch_size;
    let n_y = (img.height as usize + patch_size - 1) / patch_size;
    let mut patches = Vec::with_capacity(n_x * n_y);

    for i in 0..n_y {
        let y0 = i * patch_size;
        for j in 0..n_x {
            let x0 = j * patch_size;
            let mut patch_data = vec![0; total_size * total_size * 3];

            for local_y in 0..total_size {
                let global_y = y0 as i32 - p as i32 + local_y as i32;
                for local_x in 0..total_size {
                    let global_x = x0 as i32 - p as i32 + local_x as i32;

                    if global_x >= 0
                        && global_x < img.width as i32
                        && global_y >= 0
                        && global_y < img.height as i32
                    {
                        let src_idx =
                            (global_y as usize * img.width as usize + global_x as usize) * 3;
                        let dst_idx = (local_y * total_size + local_x) * 3;

                        patch_data[dst_idx] = img.data[src_idx];
                        patch_data[dst_idx + 1] = img.data[src_idx + 1];
                        patch_data[dst_idx + 2] = img.data[src_idx + 2];
                    }
                }
            }

            patches.push(RawImage {
                channels: 3,
                width: total_size as DimType,
                height: total_size as DimType,
                data: patch_data,
            });
        }
    }

    patches
}

pub fn generate_patches(img: RawImage, patch_size: usize, padding: usize) -> Vec<RawImage> {
    assert!(patch_size > padding * 2);
    let patch_size = patch_size - padding * 2;
    generate_patches_m(img, patch_size, padding)
}
pub fn combine_patches(
    patches: Vec<RawImage>,
    width: DimType,
    height: DimType,
    patch_size: usize,
    padding: usize,
) -> RawImage {
    assert!(patch_size > padding * 2);
    let patch_size = patch_size - padding * 2;
    combine_patches_m(patches, width, height, patch_size, padding)
}

pub fn combine_patches_m(
    patches: Vec<RawImage>,
    width: DimType,
    height: DimType,
    patch_size: usize,
    margin: usize,
) -> RawImage {
    let p = margin;
    let total_size = patch_size + 2 * p;
    let width_usize = width as usize;
    let height_usize = height as usize;
    let n_x = (width_usize + patch_size - 1) / patch_size;
    let mut output_data = vec![0; width_usize * height_usize * 3];

    for (idx, patch) in patches.iter().enumerate() {
        let i = idx / n_x;
        let j = idx % n_x;
        let y0 = i * patch_size;
        let x0 = j * patch_size;

        let h = std::cmp::min(patch_size, height_usize - y0);
        let w = std::cmp::min(patch_size, width_usize - x0);

        for y in 0..h {
            for x in 0..w {
                let src_idx = ((y + p) * total_size + (x + p)) * 3;
                let dst_idx = ((y0 + y) * width_usize + (x0 + x)) * 3;

                output_data[dst_idx] = patch.data[src_idx];
                output_data[dst_idx + 1] = patch.data[src_idx + 1];
                output_data[dst_idx + 2] = patch.data[src_idx + 2];
            }
        }
    }

    RawImage {
        channels: 3,
        width,
        height,
        data: output_data,
    }
}
