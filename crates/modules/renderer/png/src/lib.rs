use std::collections::HashMap;

use anyhow::bail;
use cosmic_text::{
    Align, Attrs, Buffer, Color, FontSystem, LayoutRun, Metrics, Shaping, Stretch, Style,
    SwashCache, Weight,
};

use export::Export;
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
    pub text_direction: TextDirectionMode,
}
impl Default for PngRenderConfig {
    fn default() -> Self {
        Self {
            min_fontsize: 8.0,
            max_fontsize: 96.0,
            detect_offset: 8.0,
            fg_color: None,
            bg_color: None,
            align: MyAlign::Center,
            letter_spacing: None,
            font_size: 24.0,
            line_height: 1.2,
            family: None,
            text_direction: TextDirectionMode::Auto,
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum TextDirectionMode {
    #[default]
    Auto,
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy)]
pub enum MyAlign {
    Left,
    Center,
    Right,
}

impl PngRenderer {
    pub fn render(&mut self, exp: Export, config: PngRenderConfig) -> RawImage {
        let mut img = exp.get_image();
        normalize_color_image(&mut img);
        let overlay = exp.get_overlay();
        apply_inpaint_overlay(&mut img, &overlay);
        let mut placed_boxes = Vec::new();

        for block in exp.blocks {
            let Some(obb) = block.obb() else {
                continue;
            };
            let text = block
                .translations
                .get("last_trans")
                .and_then(|key| block.translations.get(key))
                .or_else(|| block.translations.values().next())
                .cloned()
                .unwrap_or_else(|| block.text.clone());
            if text.trim().is_empty() {
                continue;
            }

            let source_text = block.text.clone();
            let detected_vertical =
                auto_detect_vertical(&source_text, block.lines.len(), obb.w, obb.h);
            let vertical = match config.text_direction {
                TextDirectionMode::Auto => detected_vertical,
                TextDirectionMode::Horizontal => false,
                TextDirectionMode::Vertical => true,
            };
            let size = render_size_for_direction(
                (
                    obb.w.ceil().max(1.0) as usize,
                    obb.h.ceil().max(1.0) as usize,
                ),
                vertical,
            );
            let mut render_block = RenderTextBlock {
                align: match config.align {
                    MyAlign::Left => Align::Left,
                    MyAlign::Center => Align::Center,
                    MyAlign::Right => Align::Right,
                },
                default_font_size: config.font_size,
                default_line_height: config.line_height,
                vertical,
                size,
                texts: vec![Text {
                    text,
                    letter_spacing: config.letter_spacing,
                    color: config.fg_color.or(block.fg_color).or(Some((0, 0, 0))),
                    bg_color: config.bg_color.or(block.bg_color).or(Some((255, 255, 255))),
                    font_size: config.font_size,
                    line_height: config.line_height,
                    family: config.family.clone(),
                    weight: None,
                    style: Style::Normal,
                    stretch: None,
                }],
            };

            let mut font_size =
                self.max_fontsize(size, render_block.clone(), config.max_fontsize, 1.0);
            if block.font_size > 0 {
                let detected = block.font_size as f32;
                font_size = font_size.clamp(
                    (detected - config.detect_offset).max(1.0),
                    detected + config.detect_offset,
                );
            }
            font_size = font_size
                .clamp(config.min_fontsize, config.max_fontsize)
                .round()
                .max(1.0);
            render_block.set_font_size(font_size);

            let text_img = self.render_block(render_block);
            let theta = if vertical { obb.theta } else { 0.0 };
            let (x, y, placed) = find_non_overlapping_position(
                &placed_boxes,
                img.width as f64,
                img.height as f64,
                &text_img,
                obb.x,
                obb.y,
                theta,
            );
            alpha_composite_rotated(&mut img, &text_img, x, y, theta);
            placed_boxes.push(placed);
        }
        img
    }
}

fn auto_detect_vertical(source_text: &str, line_count: usize, w: f64, h: f64) -> bool {
    if looks_like_horizontal_sentence(source_text) {
        return false;
    }
    if line_count > 1 && source_text.chars().count() >= line_count.saturating_mul(3) {
        return false;
    }
    h > w * 1.35
}

fn looks_like_horizontal_sentence(text: &str) -> bool {
    let chars = text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<Vec<_>>();
    if chars.len() >= 8 {
        return true;
    }
    let horizontal_marks = [
        '。', '、', '，', '？', '！', ',', '.', '?', '!', '…', 'ー', '〜', '~',
    ];
    let mark_count = chars
        .iter()
        .filter(|ch| horizontal_marks.contains(ch))
        .count();
    mark_count > 0 && chars.len() >= 4
}

#[derive(Clone, Copy, Debug)]
struct Aabb {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

impl Aabb {
    fn intersects(self, other: Self) -> bool {
        self.x1 < other.x2 && self.x2 > other.x1 && self.y1 < other.y2 && self.y2 > other.y1
    }

    fn in_bounds(self, width: f64, height: f64) -> bool {
        self.x1 >= 0.0 && self.y1 >= 0.0 && self.x2 <= width && self.y2 <= height
    }

    fn area(self) -> f64 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }
}

fn find_non_overlapping_position(
    placed_boxes: &[Aabb],
    img_width: f64,
    img_height: f64,
    overlay: &RawImage,
    center_x: f64,
    center_y: f64,
    theta: f64,
) -> (f64, f64, Aabb) {
    let base = rotated_aabb(overlay, center_x, center_y, theta);
    if !collides(base, placed_boxes) && base.in_bounds(img_width, img_height) {
        return (center_x, center_y, base);
    }

    let step = overlay.width.max(overlay.height) as f64 * 0.35 + 10.0;
    let directions = [
        (0.0, -1.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (1.0, -1.0),
        (-1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    ];
    let mut best = (
        center_x,
        center_y,
        base,
        collision_score(base, placed_boxes),
    );

    for ring in 1..=6 {
        for (dx, dy) in directions {
            let x = center_x + dx * step * ring as f64;
            let y = center_y + dy * step * ring as f64;
            let candidate = rotated_aabb(overlay, x, y, theta);
            if !candidate.in_bounds(img_width, img_height) {
                continue;
            }
            let score = collision_score(candidate, placed_boxes);
            if score <= f64::EPSILON {
                return (x, y, candidate);
            }
            if score < best.3 {
                best = (x, y, candidate, score);
            }
        }
    }

    (best.0, best.1, best.2)
}

fn collides(candidate: Aabb, placed_boxes: &[Aabb]) -> bool {
    placed_boxes
        .iter()
        .any(|placed| candidate.intersects(*placed))
}

fn collision_score(candidate: Aabb, placed_boxes: &[Aabb]) -> f64 {
    placed_boxes
        .iter()
        .map(|placed| intersection(candidate, *placed).area())
        .sum()
}

fn intersection(a: Aabb, b: Aabb) -> Aabb {
    Aabb {
        x1: a.x1.max(b.x1),
        y1: a.y1.max(b.y1),
        x2: a.x2.min(b.x2),
        y2: a.y2.min(b.y2),
    }
}

fn rotated_aabb(overlay: &RawImage, center_x: f64, center_y: f64, theta: f64) -> Aabb {
    let half_w = overlay.width as f64 / 2.0;
    let half_h = overlay.height as f64 / 2.0;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ];
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (dx, dy) in corners {
        let x = center_x + dx * cos_t - dy * sin_t;
        let y = center_y + dx * sin_t + dy * cos_t;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    Aabb {
        x1: min_x,
        y1: min_y,
        x2: max_x,
        y2: max_y,
    }
}

fn render_size_for_direction(size: (usize, usize), vertical: bool) -> (usize, usize) {
    if vertical {
        size
    } else {
        let long = size.0.max(size.1);
        let short = size.0.min(size.1);
        (long.max(1), short.max(1))
    }
}

fn normalize_color_image(img: &mut RawImage) {
    match img.channels {
        1 => {
            let mut data = Vec::with_capacity(img.data.len() * 3);
            for gray in &img.data {
                data.extend([*gray, *gray, *gray]);
            }
            img.data = data;
            img.channels = 3;
        }
        2 => {
            let mut data = Vec::with_capacity(img.width as usize * img.height as usize * 4);
            for px in img.data.chunks_exact(2) {
                data.extend([px[0], px[0], px[0], px[1]]);
            }
            img.data = data;
            img.channels = 4;
        }
        _ => {}
    }
}

fn apply_inpaint_overlay(base: &mut RawImage, overlay: &RawImage) {
    if base.width != overlay.width || base.height != overlay.height || base.channels < 3 {
        return;
    }
    base.data
        .chunks_mut(base.channels as usize)
        .zip(overlay.data.chunks(overlay.channels as usize))
        .for_each(|(base_px, overlay_px)| {
            if overlay.channels >= 4 {
                let alpha = overlay_px[3] as f32 / 255.0;
                if alpha <= 0.0 {
                    return;
                }
                for channel in 0..3 {
                    base_px[channel] = ((overlay_px[channel] as f32 * alpha)
                        + (base_px[channel] as f32 * (1.0 - alpha)))
                        .round() as u8;
                }
            } else if overlay.channels >= 3 {
                base_px[..3].copy_from_slice(&overlay_px[..3]);
            }
        });
}

fn alpha_composite_rotated(
    base: &mut RawImage,
    overlay: &RawImage,
    center_x: f64,
    center_y: f64,
    theta: f64,
) {
    if overlay.width == 0 || overlay.height == 0 || overlay.channels < 4 || base.channels < 3 {
        return;
    }

    let half_w = overlay.width as f64 / 2.0;
    let half_h = overlay.height as f64 / 2.0;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ];
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (dx, dy) in corners {
        let x = center_x + dx * cos_t - dy * sin_t;
        let y = center_y + dx * sin_t + dy * cos_t;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    let start_x = min_x.floor().max(0.0) as i32;
    let start_y = min_y.floor().max(0.0) as i32;
    let end_x = max_x.ceil().min(base.width as f64 - 1.0) as i32;
    let end_y = max_y.ceil().min(base.height as f64 - 1.0) as i32;
    if start_x > end_x || start_y > end_y {
        return;
    }

    for y in start_y..=end_y {
        for x in start_x..=end_x {
            let dx = x as f64 + 0.5 - center_x;
            let dy = y as f64 + 0.5 - center_y;
            let src_x = dx * cos_t + dy * sin_t + half_w;
            let src_y = -dx * sin_t + dy * cos_t + half_h;
            if src_x < 0.0
                || src_y < 0.0
                || src_x >= overlay.width as f64
                || src_y >= overlay.height as f64
            {
                continue;
            }
            alpha_composite_pixel(
                base,
                x as usize,
                y as usize,
                overlay.rgba_pixel(src_x as DimType, src_y as DimType),
            );
        }
    }
}

fn alpha_composite_pixel(base: &mut RawImage, x: usize, y: usize, rgba: [u8; 4]) {
    if rgba[3] == 0 {
        return;
    }
    let idx = (y * base.width as usize + x) * base.channels as usize;
    let alpha = rgba[3] as f32 / 255.0;
    for channel in 0..3 {
        base.data[idx + channel] = ((rgba[channel] as f32 * alpha)
            + (base.data[idx + channel] as f32 * (1.0 - alpha)))
            .round() as u8;
    }
    if base.channels >= 4 {
        base.data[idx + 3] = 255;
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
    pub fn get_id(&mut self, color: (u8, u8, u8)) -> anyhow::Result<usize> {
        if let Some(i) = self.map.get(&color) {
            return Ok(*i);
        }
        self.index += 1;
        if self.index >= 255 {
            bail!("To many colors in text block")
        }
        self.map.insert(color, self.index);
        self.map2.insert(self.index, color);

        Ok(self.index)
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
        if text.vertical {
            return self.create_vertical_buffer(text, color_map);
        }

        let metrics = to_metrics(&text);
        let mut buffer_ = Buffer::new(&mut self.font_system, metrics);
        let mut buffer = buffer_.borrow_with(&mut self.font_system);
        buffer.set_size(Some(text.size.0 as f32), Some(text.size.1 as f32));
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

    fn create_vertical_buffer(
        &mut self,
        text: &RenderTextBlock,
        color_map: &mut ColorMap,
    ) -> Buffer {
        let mut vertical_text = text.clone();
        vertical_text.texts.iter_mut().for_each(|span| {
            span.text = span
                .text
                .chars()
                .filter(|ch| *ch != '\r')
                .flat_map(|ch| if ch == '\n' { vec![ch] } else { vec![ch, '\n'] })
                .collect::<String>()
                .trim_end()
                .to_owned();
        });

        let metrics = to_metrics(&vertical_text);
        let mut buffer_ = Buffer::new(&mut self.font_system, metrics);
        let mut buffer = buffer_.borrow_with(&mut self.font_system);
        buffer.set_size(
            Some(vertical_text.size.0 as f32),
            Some(vertical_text.size.1 as f32),
        );
        let attrs = Attrs::new();
        let spans = vertical_text
            .texts
            .iter()
            .map(|v| (v.text.as_str(), v.to_attr(color_map)))
            .collect::<Vec<_>>();
        buffer.set_rich_text(
            spans.iter().map(|(text, attrs)| (*text, attrs.clone())),
            &attrs,
            Shaping::Advanced,
            Some(vertical_text.align),
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
        let w = text.size.0.max(1);
        let h = text.size.1.max(1);

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
                        if x >= w || y >= h {
                            return;
                        }
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
        let text = RawImage {
            width: w as DimType,
            height: h as DimType,
            data: flat,
            channels: 4,
        };
        bg.apply(text)
    }

    pub fn max_fontsize(
        &mut self,
        target_size: (usize, usize),
        mut text: RenderTextBlock,
        max_size: f32,
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
        let mut high = 1.0_f32.min(max_size.max(1.0));
        while {
            let (w, h) = measure(high);
            w <= target_size.0 && h <= target_size.1 && high < max_size
        } {
            high = (high * 2.0).min(max_size);
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
            .metadata(
                color_map
                    .get_id(self.bg_color.unwrap_or((255, 255, 255)))
                    .unwrap(),
            );
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

    use crate::{apply_inpaint_overlay, PngRenderer, RenderTextBlock, Text};

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

        assert_eq!(img.width, 1000);
        assert_eq!(img.height, 2000);
        assert_eq!(img.channels, 4);
        assert!(
            img.data.chunks(4).any(|pixel| pixel[3] > 0),
            "text rendering should produce visible pixels"
        );
    }

    #[test]
    fn max_fontsize_respects_cap_for_unmeasured_text() {
        let mut renderer = PngRenderer::default();
        let block = RenderTextBlock {
            align: cosmic_text::Align::Center,
            default_font_size: 1.0,
            default_line_height: 1.2,
            vertical: false,
            size: (100, 100),
            texts: vec![Text {
                text: String::new(),
                letter_spacing: None,
                color: Some((0, 0, 0)),
                bg_color: None,
                stretch: None,
                style: Style::Normal,
                weight: None,
                family: Some("Arial".to_owned()),
                font_size: 1.0,
                line_height: 1.2,
            }],
        };

        let font_size = renderer.max_fontsize((100, 100), block, 8.0, 0.25);

        assert!(font_size.is_finite());
        assert!(font_size <= 8.0);
    }

    #[test]
    fn apply_inpaint_overlay_blends_rgba_overlay() {
        let mut base = interface_image::RawImage {
            data: vec![255, 255, 255, 255, 255, 255],
            width: 2,
            height: 1,
            channels: 3,
        };
        let overlay = interface_image::RawImage {
            data: vec![10, 20, 30, 255, 200, 0, 0, 128],
            width: 2,
            height: 1,
            channels: 4,
        };

        apply_inpaint_overlay(&mut base, &overlay);

        assert_eq!(&base.data[..3], &[10, 20, 30]);
        assert_eq!(&base.data[3..], &[227, 127, 127]);
    }
}
