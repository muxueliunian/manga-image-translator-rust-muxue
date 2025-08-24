use std::{fs::create_dir_all, path::PathBuf};

use clap::Parser as _;
use config::Config;
use log::{error, warn};
use walkdir::WalkDir;

use crate::{settings::Settings, setup::Models};

pub mod cli;
mod debug;
mod dict;
mod execute;
pub mod settings;
pub mod setup;
mod ui;

#[tokio::main]
async fn main() {
    let ui = std::env::args().count() == 1;
    if ui {
        env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

        let native_options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([400.0, 300.0])
                .with_min_inner_size([300.0, 220.0]),
            // .with_icon(
            //     // NOTE: Adding an icon is optional
            //     eframe::icon_data::from_png_bytes(
            //         &include_bytes!("../assets/icon-256.png")[..],
            //     )
            //     .expect("Failed to load icon"),
            // ),
            ..Default::default()
        };
        eframe::run_native(
            "Manga Image Translator",
            native_options,
            Box::new(|cc| Ok(Box::new(ui::MitApp::new(cc)))),
        )
        .expect("Failed to run egui");
        return;
    }
    let cli = cli::Cli::parse();
    let mut input = WalkDir::new(&cli.input)
        .into_iter()
        .filter_map(|v| v.ok())
        .map(|v| v.path().to_path_buf())
        .filter(|v| v.is_file())
        .map(|v| {
            v.strip_prefix(&cli.input)
                .map(|v| v.to_path_buf())
                .unwrap_or(v)
        })
        .collect::<Vec<_>>();
    let mut settings = Config::builder();
    if let Some(config) = cli.config {
        if !config.exists() {
            panic!("Config file does not exist")
        }
        settings = settings.add_source(config::File::from(config));
    }
    let settings = settings.build().expect("Failed to build settings");
    let settings = settings.try_deserialize::<Settings>().unwrap_or_default();
    if !cli.overwrite {
        //TODO: add other extensions
        input = input
            .into_iter()
            .filter(|v| !cli.output.join(v).exists())
            .filter(|v| {
                ["png", "jpg", "jpeg", "webp"].contains(
                    &v.extension()
                        .map(|v| v.to_string_lossy())
                        .unwrap_or_default()
                        .to_lowercase()
                        .as_str(),
                )
            })
            .collect::<Vec<_>>();
    }
    let mut models = Models::new(2, true, false).await;
    for path in input {
        let path = cli.input.join(path);
        if !path.exists() || !path.is_file() {
            warn!("File {} cant be found", path.display());
            continue;
        }
        let img = match image::open(&path) {
            Ok(img) => img,
            Err(err) => {
                error!("Failed to open image {}: {}", path.display(), err);
                continue;
            }
        };
        let debug_path = if cli.verbose > 2 {
            let id = uuid::Uuid::new_v4();
            let p = PathBuf::from(format!("debug/{}", id.to_string()));
            create_dir_all(&p).expect("Failed to create debug directory");
            Some(p)
        } else {
            None
        };
        models.execute(img, &settings, debug_path).await;
    }
}
