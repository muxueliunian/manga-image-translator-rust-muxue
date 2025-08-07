mod components;
mod main;
use std::path::PathBuf;

use crate::ui::components::file_upload_button;
use eframe::App;

#[derive(serde::Deserialize, serde::Serialize)]
pub struct MitApp {
    files: Vec<PathBuf>,
    file_sidebar: bool,
    settings: Settings,
}

impl Default for MitApp {
    fn default() -> Self {
        Self {
            files: Default::default(),
            file_sidebar: true,
            settings: Default::default(),
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Default)]
pub struct Settings {
    panel: SettingsPanel,
    display_settings: DisplaySettings,
}
// tools
// unfocused: mask brush, mark text areas, upscale,
// focused: fontsize, fontweight, fontstyle, edit text, translation, make draggable/resizable
#[derive(serde::Deserialize, serde::Serialize)]
struct DisplaySettings {
    display_image: bool,
    display_overlay: bool,
    display_text: bool,
    display_translation: bool,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            display_image: true,
            display_overlay: true,
            display_text: true,
            display_translation: true,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Default)]
enum SettingsPanel {
    Display,
    Models, //load, unload
    #[default]
    Tools,
}

impl MitApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        if let Some(storage) = cc.storage {
            let mut s: MitApp = eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
            s.files = s.files.into_iter().filter(|v| v.exists()).collect();
            s
        } else {
            Default::default()
        }
    }
}

impl App for MitApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.files.is_empty() {
                let available_size = ui.available_size();
                let button_size = egui::vec2(100.0, 40.0); // approximate size

                let offset = (available_size - button_size) / 2.0;

                ui.allocate_ui_at_rect(
                    egui::Rect::from_min_size(ui.min_rect().min + offset, button_size),
                    |ui| {
                        self.files.extend(file_upload_button(
                            ui,
                            "Select Files",
                            &["png", "jpg", "jpeg", "qoi", "webp"],
                        ));
                    },
                );
            } else {
                self.main_app(ui);
            }
        });
    }
}
