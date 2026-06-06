use egui::{Color32, CornerRadius, Frame, Margin, RichText, Stroke, TextEdit, Ui};
use rfd::FileDialog;

use crate::settings::ProviderPreset;
use crate::ui::{
    components::file_path_list, MitApp, ResultStatus, TargetLanguage, UiRenderer, IMAGE_EXTS,
};

impl MitApp {
    pub fn main_app(&mut self, ui: &mut Ui) {
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            ui.add_space(18.0);
            ui.vertical(|ui| {
                ui.heading(RichText::new("漫画图片翻译器").size(28.0));
                ui.label(
                    RichText::new("选择图片，配置兼容 OpenAI 的模型，然后批量输出翻译结果。")
                        .color(Color32::from_rgb(105, 95, 86)),
                );
            });
        });
        ui.add_space(12.0);

        ui.columns(2, |columns| {
            columns[0].set_width(380.0);
            self.files_panel(&mut columns[0]);
            self.config_panel(&mut columns[1]);
        });
    }

    fn files_panel(&mut self, ui: &mut Ui) {
        section_frame().show(ui, |ui| {
            ui.heading("输入图片");
            ui.label(RichText::new("支持多选图片，也可以直接加入整个文件夹。").color(mut_text()));
            ui.add_space(8.0);

            ui.horizontal_wrapped(|ui| {
                if ui.button("添加图片").clicked() {
                    if let Some(paths) = FileDialog::new()
                        .add_filter("图片", IMAGE_EXTS)
                        .pick_files()
                    {
                        self.add_files(paths);
                    }
                }
                if ui.button("添加文件夹").clicked() {
                    if let Some(path) = FileDialog::new().pick_folder() {
                        self.add_folder(path);
                    }
                }
                if ui.button("清空").clicked() && !self.is_processing {
                    self.files.clear();
                    self.results.clear();
                    self.log("已清空选择列表。");
                }
            });

            ui.add_space(8.0);
            ui.label(format!("已选择 {} 个文件", self.files.len()));
            inner_frame().show(ui, |ui| {
                ui.set_min_height(240.0);
                file_path_list(ui, &mut self.files, !self.is_processing);
            });

            ui.add_space(16.0);
            ui.heading("输出目录");
            ui.horizontal(|ui| {
                let mut output = self.output_dir.display().to_string();
                ui.add_enabled(
                    !self.is_processing,
                    TextEdit::singleline(&mut output).desired_width(f32::INFINITY),
                );
                if !self.is_processing {
                    self.output_dir = output.into();
                }
                if ui.button("浏览").clicked() && !self.is_processing {
                    if let Some(path) = FileDialog::new().pick_folder() {
                        self.output_dir = path;
                    }
                }
            });

            ui.add_space(16.0);
            ui.horizontal_wrapped(|ui| {
                let can_start = !self.is_processing && !self.files.is_empty();
                if ui
                    .add_enabled(can_start, egui::Button::new("开始翻译"))
                    .clicked()
                {
                    self.start_processing();
                }
                if self.is_processing {
                    ui.label(RichText::new("当前版本会完成队列中的任务。").color(mut_text()));
                }
            });
        });
    }

    fn config_panel(&mut self, ui: &mut Ui) {
        section_frame().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("翻译设置");
                ui.horizontal(|ui| {
                    ui.label("目标语言");
                    egui::ComboBox::from_id_salt("target_language")
                        .selected_text(self.config.target_language.label())
                        .show_ui(ui, |ui| {
                            for language in TargetLanguage::ALL {
                                ui.selectable_value(
                                    &mut self.config.target_language,
                                    language,
                                    language.label(),
                                );
                            }
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("输出格式");
                    egui::ComboBox::from_id_salt("renderer")
                        .selected_text(self.config.renderer.label())
                        .show_ui(ui, |ui| {
                            for renderer in UiRenderer::ALL {
                                ui.selectable_value(
                                    &mut self.config.renderer,
                                    renderer,
                                    renderer.label(),
                                );
                            }
                        });
                });

                ui.add_space(16.0);
                ui.heading("第三方模型");
                ui.label(
                    RichText::new("使用兼容 OpenAI Chat Completions 的接口。").color(mut_text()),
                );
                egui::Grid::new("openai_form")
                    .num_columns(2)
                    .spacing([16.0, 10.0])
                    .show(ui, |ui| {
                        ui.label("厂商预设");
                        let old_preset = self.config.openai.preset;
                        egui::ComboBox::from_id_salt("openai_provider_preset")
                            .selected_text(self.config.openai.preset.label())
                            .show_ui(ui, |ui| {
                                for preset in [
                                    ProviderPreset::DeepSeek,
                                    ProviderPreset::OpenAI,
                                    ProviderPreset::OpenRouter,
                                    ProviderPreset::SiliconFlow,
                                    ProviderPreset::DashScope,
                                    ProviderPreset::Moonshot,
                                    ProviderPreset::Zhipu,
                                    ProviderPreset::Custom,
                                ] {
                                    ui.selectable_value(
                                        &mut self.config.openai.preset,
                                        preset,
                                        preset.label(),
                                    );
                                }
                            });
                        if self.config.openai.preset != old_preset
                            && self.config.openai.preset != ProviderPreset::Custom
                        {
                            self.config.openai.apply_preset_base_url();
                        }
                        ui.end_row();

                        ui.label("Base URL");
                        ui.text_edit_singleline(&mut self.config.openai.base_url);
                        ui.end_row();

                        ui.label("API Key");
                        ui.add(
                            TextEdit::singleline(&mut self.config.openai.api_key).password(true),
                        );
                        ui.end_row();

                        ui.label("模型");
                        ui.text_edit_singleline(&mut self.config.openai.model);
                        ui.end_row();

                        ui.label("Temperature");
                        ui.add(
                            egui::DragValue::new(&mut self.config.openai.temperature)
                                .range(0.0..=2.0)
                                .speed(0.05),
                        );
                        ui.end_row();

                        ui.label("Top P");
                        ui.add(
                            egui::DragValue::new(&mut self.config.openai.top_p)
                                .range(0.0..=1.0)
                                .speed(0.05),
                        );
                        ui.end_row();

                        ui.label("超时秒数");
                        ui.add(
                            egui::DragValue::new(&mut self.config.openai.timeout_seconds)
                                .range(1..=3600)
                                .speed(1.0),
                        );
                        ui.end_row();
                    });

                ui.add_space(8.0);
                ui.label("System Prompt");
                ui.add(
                    TextEdit::multiline(&mut self.config.openai.system_prompt)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY),
                );
                ui.label("User Prompt");
                ui.add(
                    TextEdit::multiline(&mut self.config.openai.user_prompt)
                        .desired_rows(5)
                        .desired_width(f32::INFINITY),
                );

                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    if ui.button("保存配置").clicked() {
                        self.save_config_to_disk();
                    }
                    if ui.button("加载配置").clicked() {
                        self.load_config_from_disk(true);
                    }
                    if ui.button("重置").clicked() && !self.is_processing {
                        self.config = Default::default();
                        self.log("已重置配置表单。");
                    }
                });

                ui.add_space(12.0);
                self.results_panel(ui);
                ui.add_space(12.0);
                self.logs_panel(ui);
            });
        });
    }

    fn results_panel(&mut self, ui: &mut Ui) {
        ui.heading("结果");
        inner_frame().show(ui, |ui| {
            ui.set_min_height(120.0);
            egui::ScrollArea::vertical()
                .max_height(180.0)
                .show(ui, |ui| {
                    if self.results.is_empty() {
                        ui.label("暂无结果。");
                        return;
                    }
                    egui::Grid::new("results_grid")
                        .striped(true)
                        .num_columns(4)
                        .spacing([12.0, 6.0])
                        .show(ui, |ui| {
                            ui.strong("状态");
                            ui.strong("输入");
                            ui.strong("输出");
                            ui.strong("信息");
                            ui.end_row();
                            for item in &self.results {
                                let status = match &item.status {
                                    ResultStatus::Done => egui::RichText::new(item.status.label())
                                        .color(egui::Color32::from_rgb(20, 130, 70)),
                                    ResultStatus::Failed => {
                                        egui::RichText::new(item.status.label())
                                            .color(egui::Color32::from_rgb(180, 40, 40))
                                    }
                                    _ => egui::RichText::new(item.status.label()),
                                };
                                ui.label(status);
                                ui.label(item.input.display().to_string());
                                ui.label(
                                    item.output
                                        .as_ref()
                                        .map(|v| v.display().to_string())
                                        .unwrap_or_else(|| "-".to_owned()),
                                );
                                ui.label(&item.message);
                                ui.end_row();
                            }
                        });
                });
        });
    }

    fn logs_panel(&mut self, ui: &mut Ui) {
        ui.heading("日志");
        inner_frame().show(ui, |ui| {
            ui.set_min_height(100.0);
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .max_height(160.0)
                .show(ui, |ui| {
                    for line in &self.logs {
                        ui.label(line);
                    }
                });
        });
    }
}

fn section_frame() -> Frame {
    Frame::new()
        .fill(Color32::from_rgb(255, 252, 247))
        .stroke(Stroke::new(1.0, Color32::from_rgb(228, 219, 207)))
        .corner_radius(CornerRadius::same(14))
        .inner_margin(Margin::same(18))
        .outer_margin(Margin::same(10))
}

fn inner_frame() -> Frame {
    Frame::new()
        .fill(Color32::from_rgb(250, 246, 240))
        .stroke(Stroke::new(1.0, Color32::from_rgb(230, 221, 210)))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::same(10))
}

fn mut_text() -> Color32 {
    Color32::from_rgb(105, 95, 86)
}
