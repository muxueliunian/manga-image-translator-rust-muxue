use interface_image::{Mask, RawImage};
use interface_model::Model;

pub trait Inpainter: Model {
    type Options;

    fn inpaint(
        &self,
        image: RawImage,
        mask: &Mask,
        options: Self::Options,
    ) -> anyhow::Result<RawImage>;
}
