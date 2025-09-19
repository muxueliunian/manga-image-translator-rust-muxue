use std::fs::File;
use std::io::Write;
use std::path::Path;

use base64::engine::general_purpose;
use base64::Engine as _;
use export::Export;
use image::ExtendedColorType;
use image::ImageEncoder;
use interface_image::RawImage;
use serde::{Deserialize, Serialize};
use v_htmlescape::escape;

pub struct HtmlRenderer;

pub fn copy_files(path: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    if !path.join("lazyInit.js").exists() {
        File::create(path.join("lazyInit.js"))?
            .write_all(include_bytes!("../static/lazyInit.js"))?;
    }
    if !path.join("style.css").exists() {
        File::create(path.join("style.css"))?.write_all(include_bytes!("../static/style.css"))?;
    }
    if !path.join("script.js").exists() {
        File::create(path.join("script.js"))?.write_all(include_bytes!("../static/script.js"))?;
    }
    Ok(())
}

impl HtmlRenderer {
    pub fn render(data: Vec<Export>, font: Option<String>, archive: bool) -> (Vec<u8>, bool) {
        let mut html = vec![r#"<meta charset="UTF-8" />"#.to_owned()];
        let mut files = vec![];
        for v in data {
            let mut jsons = vec![];

            macro_rules! insert {
                ($img:expr) => {{
                    if archive {
                        files.push($img);
                        (files.len() - 1, files.last().unwrap())
                    } else {
                        (0, &$img)
                    }
                }};
            }
            let img = v.get_image();

            for patch in v.patches {
                let last = patch.info.translations.get("last_trans").unwrap();
                let pimg = patch.get_image();

                let (i, p_img) = insert!(pimg);
                let bg = match archive {
                    true => format!("./{}.png", i),
                    false => to_base64_img(p_img),
                };
                jsons.push(JsonData {
                    x: patch.pos.0 as u32,
                    y: patch.pos.1 as u32,
                    width: p_img.width as u32,
                    height: p_img.height as u32,
                    rotation: patch.info.angle as u32,
                    color: patch.info.fg_color.unwrap_or((0, 0, 0)),
                    shadow: patch
                        .info
                        .bg_color
                        .map(|v| (v.0, v.1, v.2, 1.0))
                        .unwrap_or((255, 255, 255, 1.0)),
                    text: patch.info.translations.get(last).unwrap().to_owned(),
                    background: bg,
                });
            }
            let (index, img) = insert!(img);

            html.push(generate(font.clone(), jsons, index, img, archive));
        }
        html.push("<!--<script>var maxWidth = 300;</script> -->".to_owned());
        html.push(r#"<script src="/lazyInit.js" defer></script>"#.to_owned());
        (html.join("\n").as_bytes().to_vec(), archive)
    }
}

#[derive(Deserialize, Serialize)]
pub struct JsonData {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    rotation: u32,
    color: (u8, u8, u8),
    shadow: (u8, u8, u8, f32),
    text: String,
    background: String,
}

fn to_base64_img(img: &RawImage) -> String {
    let mut data = vec![];
    let encoder = image::codecs::png::PngEncoder::new(&mut data);
    let ch = img.channels;
    encoder
        .write_image(
            &img.data,
            img.width as u32,
            img.height as u32,
            if ch == 4 {
                ExtendedColorType::Rgba8
            } else {
                ExtendedColorType::Rgb8
            },
        )
        .unwrap();
    let base64_str = general_purpose::STANDARD.encode(&data);
    format!(r#"data:image/png;base64,{}"#, base64_str)
}

fn generate(
    font: Option<String>,
    data: Vec<JsonData>,
    index: usize,
    img: &RawImage,
    archive: bool,
) -> String {
    let data = serde_json::to_string(&data).unwrap();
    let data_str = escape(&data);
    let font = font.unwrap_or_else(|| "arial".to_owned());
    let font_escaped = escape(&font);
    let path = match archive {
        true => format!("./{index}.png"),
        false => to_base64_img(img),
    };
    format!(
        r###"
        <div
            class="canvas-wrapper"
            style="
                --ui-font-family: {};
            "
            data-overlays='{}'
        >
            <img class="base-image" src="{}" alt="Image" />
        </div>
 "###,
        font_escaped, data_str, path
    )
}
