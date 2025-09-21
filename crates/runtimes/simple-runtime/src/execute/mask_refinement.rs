use interface_image::{Mask, RawImage};
use log::info;
use textline_merge::TextBlock;

use crate::{execute::ImageProcessor, settings::MaskRefinementSettings, setup::Models};

impl Models {
    pub fn run_mask_refinement(
        img: &RawImage,
        mask: &Mask,
        textblocks: &Vec<TextBlock>,
        config: &MaskRefinementSettings,
        img_processor: &ImageProcessor,
    ) -> anyhow::Result<Mask> {
        assert!(!textblocks.is_empty());
        info!("Run Mask Refinement: {:?}", config.method);
        mask_refinement::dispatch(
            &textblocks,
            &img,
            &mask,
            config.method,
            config.ignore_bubble,
            config.dilation_offset,
            config.kernel_size as i32,
            config.furigana,
            &img_processor,
        )
    }
}
