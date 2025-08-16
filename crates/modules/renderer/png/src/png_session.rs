use std::collections::HashMap;

use cosmic_text::{
    Align, Attrs, Buffer, Color, FontSystem, Metrics, Shaping, Stretch, Style, SwashCache, Weight,
};

use interface_image::{DimType, Mask, RawImage};
use opencv::{
    core::{Mat, MatTraitConst, Point, Size, BORDER_CONSTANT},
    imgproc::{self, dilate, morphology_default_border_value},
};
use ordered_float::OrderedFloat;

pub struct PngRenderer {
    font_system: FontSystem,
    cache: SwashCache,
}

impl Default for PngRenderer {
    fn default() -> Self {
        Self {
            font_system: FontSystem::new(),
            cache: SwashCache::new(),
        }
    }
}

fn to_metrics(input: &TextBlock) -> Metrics {
    Metrics::new(
        input.default_font_size,
        input.default_font_size * input.default_line_height,
    )
}

#[derive(Default)]
pub struct ColorMap {
    index: usize,
    map: HashMap<[u8; 3], usize>,
    map2: HashMap<usize, [u8; 3]>,
}

impl ColorMap {
    pub fn get_id(&mut self, color: [u8; 3]) -> usize {
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
                Some(s) => [s[0], s[1], s[2], 255],
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

impl PngRenderer {
    pub fn render(&mut self, text: TextBlock) {
        let metrics = to_metrics(&text);
        let mut buffer_ = Buffer::new(&mut self.font_system, metrics);
        let mut buffer = buffer_.borrow_with(&mut self.font_system);
        if text.vertical {
            buffer.set_size(Some(text.size.0 as f32), None);
        } else {
            buffer.set_size(None, Some(text.size.1 as f32))
        }
        let attrs = Attrs::new();
        let mut color_map = ColorMap::default();

        let spans = text
            .texts
            .iter()
            .map(|v| (v.text.as_str(), v.to_attr(&mut color_map)))
            .collect::<Vec<_>>();
        buffer.set_rich_text(
            spans.iter().map(|(text, attrs)| (*text, attrs.clone())),
            &attrs,
            Shaping::Advanced,
            Some(text.align),
        );
        buffer.shape_until_scroll(true);
        let buffer = buffer_;
        let layouts = buffer.layout_runs().collect::<Vec<_>>();
        let (h, w): (Vec<_>, Vec<_>) = layouts
            .iter()
            .map(|v| (v.line_top + v.line_height, v.line_w))
            .unzip();
        let h = h
            .iter()
            .map(|v| OrderedFloat(*v))
            .max()
            .unwrap_or_default()
            .ceil() as usize
            + 200;
        let w = w
            .iter()
            .map(|v| OrderedFloat(*v))
            .max()
            .unwrap_or_default()
            .ceil() as usize
            + 200;
        println!("Width: {}, Height: {}", w, h);
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
                        let x = (physical_glyph.x + x) as usize;
                        let y = (run.line_y as i32 + physical_glyph.y + y) as usize;
                        rgb[x * w + y] = [color.r(), color.g(), color.b(), 255];
                        bg[x * w + y] = glyph.metadata as u8;
                    },
                );
            }
        }

        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_ELLIPSE, // Circular shape
            Size::new(3, 3),
            Point::new(-1, -1),
        )
        .unwrap();
        let src = Mat::from_slice(&bg).unwrap();
        let src = src.reshape(1, h as i32).unwrap();
        let mut dst = Mat::default();
        dilate(
            &src,
            &mut dst,
            &kernel,
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
        RawImage {
            width: w as DimType,
            height: h as DimType,
            data: flat,
            channels: 4,
        }
        .to_image()
        .unwrap()
        .save("text.png")
        .unwrap();
        bg.to_image().unwrap().save("backdrop.png").unwrap();
    }
}

pub struct TextBlock {
    align: Align,
    default_font_size: f32,
    default_line_height: f32,
    vertical: bool,
    size: (usize, usize),
    texts: Vec<Text>,
}

pub struct Text {
    text: String,
    letter_spacing: Option<f32>,
    color: Option<(u8, u8, u8)>,
    bg_color: Option<[u8; 3]>,
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
            .metadata(color_map.get_id(self.bg_color.unwrap_or([255; 3])));
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

    use crate::png_session::{PngRenderer, Text, TextBlock};

    #[test]
    fn render_test() {
        let mut renderer = PngRenderer::default();
        let block = TextBlock {
            align: cosmic_text::Align::Center,
            default_font_size: 12.0,
            default_line_height: 1.2,
            vertical: false,
            size: (1000, 2000),
            texts: vec![Text {
                text: "Hello world, this is a test".to_owned(),
                letter_spacing: None,
                color: None,
                bg_color: None,
                stretch: None,
                style: Style::Normal,
                weight: None,
                family: None,
                font_size: 12.0,
                line_height: 1.2,
            }],
        };
        renderer.render(block);
    }
}
