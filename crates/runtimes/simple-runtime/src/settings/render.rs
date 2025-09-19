use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Default, JsonSchema)]
#[serde(default)]
pub struct RenderSettings {
    pub renderer: Renderer,
}

#[derive(Serialize, Deserialize, Default, JsonSchema, PartialEq, Eq)]
pub enum Renderer {
    Raw,
    #[default]
    Html,
}
