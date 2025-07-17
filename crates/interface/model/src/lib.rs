use std::{collections::HashMap, sync::Arc};

pub use base_util::error::ModelLoadError;
#[cfg(feature = "onnx")]
use base_util::onnx::{all_providers, Providers};

use crate::db::ModelDb;

pub mod db;

pub trait Model {
    fn name(&self) -> &'static str;
    fn kind(&self) -> &'static str;
    fn models(&self) -> HashMap<&'static str, ModelSource>;
    fn loaded(&self) -> bool;
    fn unload(&mut self);
    fn load(&mut self) -> Result<(), ModelLoadError>;
}

#[derive(Clone)]
pub struct CreateData {
    pub mode_db: Arc<ModelDb>,
    #[cfg(feature = "onnx")]
    pub providers: Vec<Providers>,
}

impl CreateData {
    pub fn all() -> Self {
        Self {
            mode_db: Arc::new(ModelDb {}),
            #[cfg(feature = "onnx")]
            providers: all_providers(),
        }
    }

    #[cfg(feature = "onnx")]
    pub fn new(providers: Vec<Providers>) -> Self {
        Self {
            mode_db: Arc::new(ModelDb {}),
            providers,
        }
    }
    #[cfg(not(feature = "onnx"))]
    pub fn new() -> Self {
        Self {
            mode_db: Arc::new(ModelDb {}),
        }
    }
}

pub struct ModelSource {
    pub url: &'static str,
    pub hash: &'static str,
}
