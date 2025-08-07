use std::path::PathBuf;

use egui::{Button, RichText, Ui};
use rfd::FileDialog;

pub fn file_upload_button(ui: &mut Ui, label: &str, exts: &[&str]) -> Vec<PathBuf> {
    if ui
        .add(
            Button::new(RichText::new(label).size(16.0).strong())
                .wrap_mode(egui::TextWrapMode::Extend),
        )
        .clicked()
    {
        if let Some(paths) = FileDialog::new()
            .add_filter("extension filter", exts)
            .pick_files()
        {
            return paths
                .into_iter()
                .filter(|v| {
                    exts.contains(
                        &v.extension()
                            .map(|v| v.to_string_lossy())
                            .unwrap_or_default()
                            .to_lowercase()
                            .as_str(),
                    )
                })
                .collect();
        }
    }

    vec![]
}
