use std::{collections::HashMap, path::PathBuf, sync::Arc};

use crate::db::ModelDb;
pub use base_util::error::ModelLoadError;
#[cfg(feature = "onnx")]
use base_util::onnx::{all_providers, Providers};

pub mod db;

pub trait ModelLoad {
    type T;
    fn loaded(&self) -> bool;
    fn get_model(&mut self) -> Option<&mut Self::T>;
    fn load(&mut self) -> Result<&mut Self::T, ModelLoadError> {
        if self.loaded() {
            return Ok(self.get_model().unwrap());
        }
        self.reload()
    }
    fn reload(&mut self) -> Result<&mut Self::T, ModelLoadError>;
}

pub trait Model {
    fn name(&self) -> &'static str;
    fn kind(&self) -> &'static str;
    fn models(&self) -> HashMap<&'static str, ModelSource>;
    fn unload(&mut self);
    fn download_model(&self, key: &str, file: &str) -> Result<PathBuf, ModelLoadError> {
        let models = self.models();
        let model = models.get(key).ok_or(ModelLoadError::ModelNotRegistered)?;
        ModelDb {}.get(self.kind(), self.name(), file, &model.url, &model.hash)
    }
    fn loaded_(&self) -> bool;
    fn reload_(&mut self) -> Result<(), ModelLoadError>;
}

#[macro_export]
macro_rules! impl_model_load_helpers {
    ($name:literal, $kind:literal) => {
        fn reload_(&mut self) -> Result<(), base_util::error::ModelLoadError> {
            self.reload()?;
            Ok(())
        }

        fn loaded_(&self) -> bool {
            self.loaded()
        }

        fn name(&self) -> &'static str {
            $name
        }

        fn kind(&self) -> &'static str {
            $kind
        }
    };
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
