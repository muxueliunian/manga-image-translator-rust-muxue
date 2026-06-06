use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Default, JsonSchema)]
#[serde(default)]
pub struct RenderSettings {
    pub renderer: Renderer,
}

#[derive(Serialize, Deserialize, Default, JsonSchema, PartialEq, Eq)]
pub enum Renderer {
    #[default]
    Png,
    Raw,
    Html,
}

impl Renderer {
    pub fn extension(&self) -> &str {
        match self {
            Renderer::Png => "png",
            Renderer::Raw => "mit.bin",
            Renderer::Html => "html",
        }
    }
}
