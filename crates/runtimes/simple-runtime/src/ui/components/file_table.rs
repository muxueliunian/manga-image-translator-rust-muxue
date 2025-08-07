use std::path::PathBuf;

use egui::{Button, Ui};

pub fn file_path_list(ui: &mut Ui, files: &mut Vec<PathBuf>) {
    let mut to_remove = vec![];

    egui::ScrollArea::both().show(ui, |ui| {
        for (i, file) in files.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(file.display().to_string());

                if ui.add(Button::new("❌").small()).clicked() {
                    to_remove.push(i);
                }
            });
        }
    });

    for i in to_remove.into_iter().rev() {
        files.remove(i);
    }
}
