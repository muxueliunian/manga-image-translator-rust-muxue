mod components;
use std::path::PathBuf;

use eframe::App;
use egui::Layout;

use crate::ui::components::file_upload_button;

#[derive(serde::Deserialize, serde::Serialize, Default)]
pub struct MitApp {
    files: Vec<PathBuf>,
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
                ui.label("Hello World");
            }
        });
    }
}
