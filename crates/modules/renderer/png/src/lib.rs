use std::collections::HashMap;

use cosmic_text::{
    Align, Attrs, Buffer, Color, FontSystem, LayoutRun, Metrics, Shaping, Stretch, Style,
    SwashCache, Weight,
};

use export::Export;
use interface_image::{DimType, Mask, RawImage};
use opencv::{
    core::{Mat, MatTraitConst, Point, Size, BORDER_CONSTANT, BORDER_DEFAULT},
    imgproc::{self, dilate, gaussian_blur, morphology_default_border_value},
};
use ordered_float::OrderedFloat;

pub struct PngRenderer {
    font_system: FontSystem,
    cache: SwashCache,
}

pub struct PngRenderConfig {
    pub min_fontsize: f32,
    pub max_fontsize: f32,
    pub detect_offset: f32,
    pub fg_color: Option<(u8, u8, u8)>,
    pub bg_color: Option<(u8, u8, u8)>,
    pub align: MyAlign,
    pub letter_spacing: Option<f32>,
    pub font_size: f32,
    pub line_height: f32,
    pub family: Option<String>,
}
pub enum MyAlign {
    Left,
    Center,
    Right,
}

impl PngRenderer {
    pub fn render(&mut self, exp: Export, config: PngRenderConfig) -> RawImage {
        let mut img = exp.get_image();
        for patch in exp.patches {
            let patch_img = patch.get_image();
            let (x, y) = patch.pos;
            img.apply_patch(&patch_img, x as u16, y as u16);
            let text = patch.info;
            let obb = text.obb().unwrap();
            let mut render_block = RenderTextBlock {
                align: match config.align {
                    MyAlign::Left => Align::Left,
                    MyAlign::Center => Align::Center,
                    MyAlign::Right => Align::Right,
                },
                default_font_size: config.font_size,
                default_line_height: config.line_height,
                vertical: false,
                size: (obb.w as usize, obb.h as usize),
                texts: vec![Text {
                    text: text.text,
                    letter_spacing: config.letter_spacing,
                    color: config.fg_color.or(text.fg_color),
                    bg_color: config.bg_color.or(text.bg_color),
                    font_size: config.font_size,
                    line_height: config.line_height,
                    family: config.family.clone(),
                    weight: None,
                    style: Style::Normal,
                    stretch: None,
                }],
            };

            let font_size = self
                .max_fontsize((obb.w as usize, obb.h as usize), render_block.clone(), 1.0)
                .clamp(
                    text.font_size as f32 - config.detect_offset,
                    text.font_size as f32 + config.detect_offset,
                )
                .clamp(config.min_fontsize, config.max_fontsize)
                .round() as u32;
            render_block.set_font_size(font_size as f32);
            let img = self.render_block(render_block);
        }
        img
    }
}

impl Default for PngRenderer {
    fn default() -> Self {
        Self {
            font_system: FontSystem::new(),
            cache: SwashCache::new(),
        }
    }
}

fn to_metrics(input: &RenderTextBlock) -> Metrics {
    Metrics::new(
        input.default_font_size,
        input.default_font_size * input.default_line_height,
    )
}

#[derive(Default)]
pub struct ColorMap {
    index: usize,
    map: HashMap<(u8, u8, u8), usize>,
    map2: HashMap<usize, (u8, u8, u8)>,
}

impl ColorMap {
    pub fn get_id(&mut self, color: (u8, u8, u8)) -> usize {
        if let Some(i) = self.map.get(&color) {
            return *i;
        }
        self.index += 1;
        if self.index >= 255 {
            panic!("To many colors in text block")
        }
        self.map.insert(color, self.index);
        self.map2.insert(self.index, color);

        self.index
    }

    pub fn to_image(&self, input: Mask) -> RawImage {
        let w = input.width;
        let h = input.height;
        let mut data = Vec::with_capacity(input.data.len());
        for id in input.data {
            let get = self.map2.get(&(id as usize));
            data.push(match get {
                Some(s) => [s.0, s.1, s.2, 255],
                None => [0, 0, 0, 0],
            });
        }
        let len = data.len() * 4;
        let cap = data.capacity() * 4;
        let ptr = data.as_ptr() as *mut u8;

        std::mem::forget(data);

        let flat: Vec<u8> = unsafe { Vec::from_raw_parts(ptr, len, cap) };
        RawImage {
            data: flat,
            width: w,
            height: h,
            channels: 4,
        }
    }
}

fn backdrop_kernel(font_size: i32) -> opencv::Result<opencv::core::Mat> {
    let k = (font_size as f32 / 12.0).ceil() as i32;
    let size = 2 * k + 1;

    imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(size, size),
        Point::new(-1, -1),
    )
}

fn wh(layouts: &Vec<LayoutRun<'_>>) -> (usize, usize) {
    let (h, w): (Vec<_>, Vec<_>) = layouts
        .iter()
        .map(|v| (v.line_top + v.line_height, v.line_w))
        .unzip();
    let h = h
        .iter()
        .map(|v| OrderedFloat(*v))
        .max()
        .unwrap_or_default()
        .ceil() as usize;
    let w = w
        .iter()
        .map(|v| OrderedFloat(*v))
        .max()
        .unwrap_or_default()
        .ceil() as usize;
    (w, h)
}
impl PngRenderer {
    fn create_buffer(&mut self, text: &RenderTextBlock, color_map: &mut ColorMap) -> Buffer {
        let metrics = to_metrics(&text);
        let mut buffer_ = Buffer::new(&mut self.font_system, metrics);
        let mut buffer = buffer_.borrow_with(&mut self.font_system);
        if text.vertical {
            buffer.set_size(Some(text.size.0 as f32), None);
        } else {
            buffer.set_size(None, Some(text.size.1 as f32))
        }
        let attrs = Attrs::new();
        let spans = text
            .texts
            .iter()
            .map(|v| (v.text.as_str(), v.to_attr(color_map)))
            .collect::<Vec<_>>();
        buffer.set_rich_text(
            spans.iter().map(|(text, attrs)| (*text, attrs.clone())),
            &attrs,
            Shaping::Advanced,
            Some(text.align),
        );
        buffer.shape_until_scroll(true);
        buffer_
    }

    pub fn render_block(&mut self, text: RenderTextBlock) -> RawImage {
        let font_size =
            text.texts.iter().map(|v| v.font_size).sum::<f32>() / text.texts.len() as f32;
        let mut color_map = ColorMap::default();
        let buffer = self.create_buffer(&text, &mut color_map);
        let layouts = buffer.layout_runs().collect::<Vec<_>>();
        let (w, h) = wh(&layouts);

        let mut rgb = vec![[0_u8; 4]; h as usize * w as usize];
        let mut bg = vec![0_u8; h as usize * w as usize];
        for run in layouts {
            for glyph in run.glyphs.iter() {
                let physical_glyph = glyph.physical((0., 0.), 1.0);
                let glyph_color = glyph.color_opt.unwrap_or(Color::rgb(0, 0, 0));
                self.cache.with_pixels(
                    &mut self.font_system,
                    physical_glyph.cache_key,
                    glyph_color,
                    |x, y, color| {
                        let x = physical_glyph.x + x;
                        let y = run.line_y as i32 + physical_glyph.y + y;
                        let a = color.a();
                        if a == 0 || x < 0 || y < 0 {
                            return;
                        }
                        let x = x as usize;
                        let y = y as usize;
                        rgb[y * w + x] = [color.r(), color.g(), color.b(), a];
                        if a >= 127 {
                            bg[y * w + x] = glyph.metadata as u8;
                        }
                    },
                );
            }
        }

        let src = Mat::from_slice(&bg).unwrap();
        let src = src.reshape(1, h as i32).unwrap();
        let mut dst = Mat::default();
        dilate(
            &src,
            &mut dst,
            &backdrop_kernel(font_size as i32).unwrap(),
            Point::new(-1, -1),
            1,
            BORDER_CONSTANT,
            morphology_default_border_value().unwrap(),
        )
        .unwrap();
        let bg = color_map.to_image(Mask::from(dst));
        let len = rgb.len() * 4;
        let cap = rgb.capacity() * 4;
        let ptr = rgb.as_ptr() as *mut u8;

        std::mem::forget(rgb);

        let flat: Vec<u8> = unsafe { Vec::from_raw_parts(ptr, len, cap) };
        let src = Mat::from_slice(&bg.data).unwrap();
        let src = src.reshape(4, h as i32).unwrap();
        let mut dst = Mat::default();
        let k = (font_size as f32 / 12.0).ceil() as i32;
        gaussian_blur(
            &src,
            &mut dst,
            Size::new(2 * k + 1, 2 * k + 1),
            0.0,
            0.0,
            BORDER_DEFAULT,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap();

        let text = RawImage {
            width: w as DimType,
            height: h as DimType,
            data: flat,
            channels: 4,
        };
        RawImage::try_from(dst).unwrap().apply(text)
    }

    pub fn max_fontsize(
        &mut self,
        target_size: (usize, usize),
        mut text: RenderTextBlock,
        eps: f32,
    ) -> f32 {
        let mut measure = |size: f32| {
            let mut color_map = ColorMap::default();
            text.set_font_size(size);
            let buffer = self.create_buffer(&text, &mut color_map);
            let layouts = buffer.layout_runs().collect::<Vec<_>>();
            wh(&layouts)
        };
        let mut low = 0.0;
        let mut high = 1.0;
        while {
            let (w, h) = measure(high);
            w <= target_size.0 && h <= target_size.1
        } {
            high *= 2.0;
        }

        while high - low > eps {
            let mid = (low + high) / 2.0;
            let (w, h) = measure(mid);
            if w <= target_size.0 && h <= target_size.1 {
                low = mid;
            } else {
                high = mid;
            }
        }

        low
    }
}

#[derive(Clone)]
pub struct RenderTextBlock {
    align: Align,
    default_font_size: f32,
    default_line_height: f32,
    vertical: bool,
    size: (usize, usize),
    texts: Vec<Text>,
}

impl RenderTextBlock {
    fn set_font_size(&mut self, font_size: f32) {
        self.default_font_size = font_size;
        self.texts.iter_mut().for_each(|v| v.font_size = font_size);
    }
}

#[derive(Clone)]
pub struct Text {
    text: String,
    letter_spacing: Option<f32>,
    color: Option<(u8, u8, u8)>,
    bg_color: Option<(u8, u8, u8)>,
    stretch: Option<Stretch>,
    style: Style,
    weight: Option<Weight>,
    family: Option<String>,
    font_size: f32,
    line_height: f32,
}

impl Text {
    pub fn to_attr<'a>(&'a self, color_map: &mut ColorMap) -> Attrs<'a> {
        let mut attrs = Attrs::new();
        let color = self.color.unwrap_or_default();
        attrs = attrs
            .color(Color::rgb(color.0, color.1, color.2))
            .style(self.style)
            .metrics(Metrics::new(
                self.font_size,
                self.font_size * self.line_height,
            ))
            .metadata(color_map.get_id(self.bg_color.unwrap_or((255, 255, 255))));
        if let Some(letter_spacing) = self.letter_spacing {
            attrs = attrs.letter_spacing(letter_spacing)
        }
        if let Some(stretch) = self.stretch {
            attrs = attrs.stretch(stretch);
        }
        if let Some(weight) = self.weight {
            attrs = attrs.weight(weight);
        }
        if let Some(family) = &self.family {
            attrs = attrs.family(cosmic_text::Family::Name(family));
        }

        attrs
    }
}

#[cfg(test)]
mod tests {
    use cosmic_text::Style;
    use env_logger::Env;

    use crate::{PngRenderer, RenderTextBlock, Text};

    #[test]
    fn render_test() {
        env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();
        let mut renderer = PngRenderer::default();
        let block = RenderTextBlock {
            align: cosmic_text::Align::Center,
            default_font_size: 1.0,
            default_line_height: 1.2,
            vertical: false,
            size: (1000, 2000),
            texts: vec![Text {
                text: "Hello world, this is a test".to_owned(),
                letter_spacing: None,
                color: Some((255, 0, 0)),
                bg_color: None,
                stretch: None,
                style: Style::Normal,
                weight: None,
                family: Some("Arial".to_owned()),
                font_size: 24.0,
                line_height: 1.2,
            }],
        };
        let img = renderer.render_block(block);
        img.to_image().unwrap().save("text.png").unwrap();
    }
}
