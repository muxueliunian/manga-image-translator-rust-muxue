use std::{fs::File, io::Write as _, path::Path};

use interface_image::{Mask, RawImage};
use serde::Serialize;

pub mod bbox;
pub mod textblocks;

pub fn save_mask(mask: &Mask, path: &Path) -> anyhow::Result<()> {
    mask.clone().to_image()?.save(path)?;
    Ok(())
}

pub fn save_img(img: &RawImage, path: &Path) -> anyhow::Result<()> {
    img.clone().to_image()?.save(path)?;
    Ok(())
}

pub fn save_json<T>(config: &T, path: &Path) -> anyhow::Result<()>
where
    T: ?Sized + Serialize,
{
    File::create(path)?.write_all(serde_json::to_string(config)?.as_bytes())?;
    Ok(())
}
