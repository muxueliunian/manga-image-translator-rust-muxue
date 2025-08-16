use std::collections::HashMap;

pub struct Export {
    img: Image,
    patches: Vec<Patch>,
}

pub struct Image {
    width: u16,
    height: u16,
    data: Vec<u8>,
    raw: bool,
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
    obb: Obb,
    textlines: Vec<[Point; 4]>,
    bg: Image,
    original: String,
    translations: HashMap<String, String>,
}
