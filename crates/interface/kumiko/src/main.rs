mod page;
mod panel;
mod segment;
fn main() {
    let img = image::open("/Users/frederik/code/rust/detector/crates/kumiko/xkcd.png")
        .unwrap()
        .to_rgb8();
    let w = img.width();
    let h = img.height();
    let panels = detect_panels(img.clone().into_raw(), w, h, true, None, true);
    let img = DynamicImage::from(img);
    println!("{}", panels.len());
    for (i, panel) in panels.into_iter().enumerate() {
        let img = img.view(
            panel.x as u32,
            panel.y as u32,
            panel.w() as u32,
            panel.h() as u32,
        );
        img.to_image().save(format!("{i}.png")).unwrap();
    }
}

use image::{DynamicImage, GenericImageView};

use crate::page::detect_panels;
