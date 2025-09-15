use std::{collections::HashMap, path::PathBuf, sync::Arc};

use crate::db::ModelDb;
use anyhow::anyhow;
#[cfg(feature = "onnx")]
use base_util::onnx::{all_providers, Providers};

pub mod db;

pub trait ModelLoad {
    type T;
    fn loaded(&self) -> bool;
    fn get_model(&mut self) -> Option<&mut Self::T>;
    fn load(&mut self) -> anyhow::Result<&mut Self::T> {
        if self.loaded() {
            return Ok(self.get_model().expect("Checked before"));
        }
        self.reload()
    }
    fn reload(&mut self) -> anyhow::Result<&mut Self::T>;
}

pub trait Model {
    fn name(&self) -> &'static str;
    fn kind(&self) -> &'static str;
    fn models(&self) -> HashMap<&'static str, ModelSource>;
    fn unload(&mut self);
    fn download_model(&self, key: &str, file: &str) -> anyhow::Result<PathBuf> {
        let models = self.models();
        let model = models.get(key).ok_or(anyhow!("Model not found"))?;
        ModelDb {}.get(self.kind(), self.name(), file, &model.url, &model.hash)
    }
    fn loaded_(&self) -> bool;
    fn reload_(&mut self) -> anyhow::Result<()>;
}

#[macro_export]
macro_rules! impl_model_load_helpers {
    ($kind:literal, $name:literal) => {
        fn reload_(&mut self) -> anyhow::Result<()> {
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
