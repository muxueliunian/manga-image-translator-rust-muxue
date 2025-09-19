use std::{io::Cursor, ops::Deref};

use image::{
    guess_format, write_buffer_with_format, DynamicImage, EncodableLayout, GenericImageView,
    ImageBuffer, ImageFormat, Pixel, PixelWithColorType,
};
use interface_image::RawImage;
use textline_merge::TextBlock;
pub struct Export {
    img: Image,
    pub patches: Vec<Patch>,
}

impl Image {
    pub fn export(self) -> Vec<u8> {
        let mut bytes = vec![];
        bytes.extend(self.width.to_le_bytes());
        bytes.extend(self.height.to_le_bytes());
        bytes.push(if self.raw { 1 } else { 0 });
        bytes.extend((self.data.len() as u64).to_le_bytes());
        bytes.extend(self.data);
        bytes
    }
}

impl Patch {
    pub fn export(self) -> Vec<u8> {
        let mut buffer = vec![];
        buffer.extend((self.pos.0 as u64).to_le_bytes());
        buffer.extend((self.pos.1 as u64).to_le_bytes());
        buffer.extend(self.bg.export());
        buffer.extend(self.info.export());
        buffer
    }
}

impl Export {
    pub fn export(self) -> Vec<u8> {
        let mut buffer = b"mit-rust:".to_vec();
        buffer.extend(1_u32.to_le_bytes());
        buffer.extend(self.img.export());
        buffer.extend((self.patches.len() as u64).to_le_bytes());
        for patch in self.patches {
            buffer.extend(patch.export())
        }
        buffer
    }
}

fn convert<P: Pixel + PixelWithColorType, Container>(
    img: &ImageBuffer<P, Container>,
    format: ImageFormat,
) -> Vec<u8>
where
    [P::Subpixel]: EncodableLayout,
    Container: Deref<Target = [P::Subpixel]>,
{
    let mut cursor: Cursor<Vec<u8>> = Cursor::new(vec![]);
    let buf = &img.as_raw()[..(img.width() * img.height()) as usize];
    write_buffer_with_format(
        &mut cursor,
        buf.as_bytes(),
        img.width(),
        img.height(),
        <P as PixelWithColorType>::COLOR_TYPE,
        format,
    )
    .unwrap();

    cursor.into_inner()
}

impl Export {
    pub fn new(
        raw: DynamicImage,
        inpainted: DynamicImage,
        blocks: Vec<TextBlock>,
        format: Option<ImageFormat>,
    ) -> Self {
        let mut patches = Vec::new();
        for block in blocks {
            let xyxy = block.xyxy();
            let patch = inpainted.view(
                xyxy.0 as u32,
                xyxy.1 as u32,
                xyxy.2 as u32 - xyxy.0 as u32,
                xyxy.3 as u32 - xyxy.1 as u32,
            );
            let data = match format {
                Some(format) => convert(&patch.to_image(), format),
                None => patch.to_image().as_bytes().to_vec(),
            };

            patches.push(Patch {
                info: block,
                pos: (xyxy.0 as usize, xyxy.1 as usize),
                bg: Image {
                    width: patch.width() as u16,
                    height: patch.height() as u16,
                    data,
                    raw: format.is_none(),
                },
            });
        }

        let raw_data = match format {
            Some(format) => match &raw {
                DynamicImage::ImageLuma8(img) => convert(img, format),
                DynamicImage::ImageLumaA8(img) => convert(img, format),
                DynamicImage::ImageRgb8(img) => convert(img, format),
                DynamicImage::ImageRgba8(img) => convert(img, format),
                DynamicImage::ImageLuma16(img) => convert(img, format),
                DynamicImage::ImageLumaA16(img) => convert(img, format),
                DynamicImage::ImageRgb16(img) => convert(img, format),
                DynamicImage::ImageRgba16(img) => convert(img, format),
                DynamicImage::ImageRgb32F(img) => convert(img, format),
                DynamicImage::ImageRgba32F(img) => convert(img, format),
                _ => unimplemented!("not implemented yet"),
            },
            None => raw.as_bytes().to_vec(),
        };
        Self {
            img: Image {
                width: raw.width() as u16,
                height: raw.height() as u16,
                data: raw_data,
                raw: format.is_none(),
            },
            patches,
        }
    }

    pub fn get_image(&self) -> RawImage {
        (&self.img).into()
    }
}

pub struct Image {
    width: u16,
    height: u16,
    data: Vec<u8>,
    raw: bool,
}

impl From<&Image> for RawImage {
    fn from(value: &Image) -> Self {
        if value.raw {
            RawImage {
                data: value.data.clone(),
                width: value.width,
                height: value.height,
                channels: (value.data.len() / (value.width as usize * value.height as usize)) as u8,
            }
        } else {
            let img =
                image::load(Cursor::new(&value.data), guess_format(&value.data).unwrap()).unwrap();
            let w = img.width() as u16;
            let h = img.height() as u16;
            let (data, channels) = match img {
                DynamicImage::ImageLuma8(image_buffer) => (image_buffer.into_raw(), 1),
                DynamicImage::ImageLumaA8(image_buffer) => (image_buffer.into_raw(), 2),
                DynamicImage::ImageRgb8(image_buffer) => (image_buffer.into_raw(), 3),
                DynamicImage::ImageRgba8(image_buffer) => (image_buffer.into_raw(), 4),
                DynamicImage::ImageLuma16(image_buffer) => {
                    let img = DynamicImage::from(image_buffer).to_luma8();
                    (img.into_raw(), 1)
                }
                DynamicImage::ImageLumaA16(image_buffer) => {
                    let img = DynamicImage::from(image_buffer).to_luma_alpha8();
                    (img.into_raw(), 2)
                }
                DynamicImage::ImageRgb16(image_buffer) => {
                    let img = DynamicImage::from(image_buffer).to_rgb8();
                    (img.into_raw(), 3)
                }
                DynamicImage::ImageRgba16(image_buffer) => {
                    let img = DynamicImage::from(image_buffer).to_rgba8();
                    (img.into_raw(), 4)
                }
                DynamicImage::ImageRgb32F(image_buffer) => {
                    let img = DynamicImage::from(image_buffer).to_rgb8();
                    (img.into_raw(), 3)
                }
                DynamicImage::ImageRgba32F(image_buffer) => {
                    let img = DynamicImage::from(image_buffer).to_rgba8();
                    (img.into_raw(), 4)
                }
                _ => unreachable!(),
            };
            RawImage {
                data,
                width: w,
                height: h,
                channels,
            }
        }
    }
}

impl Image {
    pub fn new(width: u16, height: u16, data: Vec<u8>, raw: bool) -> Self {
        Image {
            width,
            height,
            data,
            raw,
        }
    }
}
pub struct Obb {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    rotation: u16,
}
pub struct Point {
    x: usize,
    y: usize,
}

pub struct Patch {
    pub info: TextBlock,
    pub pos: (usize, usize),
    bg: Image,
}

impl Patch {
    pub fn get_image(&self) -> RawImage {
        (&self.bg).into()
    }
}
