use egui::Ui;

use crate::ui::{components::file_path_list, MitApp};

impl MitApp {
    pub fn main_app(&mut self, ui: &mut Ui) {
        if self.file_sidebar {
            egui::SidePanel::left("left_panel")
                .resizable(true)
                .default_width(150.0)
                .width_range(80.0..=200.0)
                .show_inside(ui, |ui| {
                    ui.separator();
                    file_path_list(ui, &mut self.files);
                });
        }

        egui::SidePanel::right("right_panel")
            .resizable(true)
            .default_width(150.0)
            .width_range(80.0..=200.0)
            .show_inside(ui, |ui| {
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| ui.label("text3"));
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label("text2");
            });
        });
    }
}
