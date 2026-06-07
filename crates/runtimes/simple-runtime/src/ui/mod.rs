mod components;
mod main;

use std::{
    fs::{create_dir_all, File},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc,
    },
};

use eframe::App;
use egui::{
    epaint::Shadow, Color32, FontData, FontDefinitions, FontFamily, RichText, Stroke, Style, Vec2,
    Visuals,
};
use serde::{Deserialize, Serialize};
use tokio::{runtime::Handle, sync::Mutex};

use crate::{
    prepare_renderer_assets, render_export_bytes_with_settings,
    settings::{OpenAICompatibleSettings, ProviderPreset, Renderer, Settings as RuntimeSettings},
    setup::Models,
};

pub(super) const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "qoi", "webp"];
const CONFIG_PATH: &str = "config/app.json";

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct MitApp {
    files: Vec<PathBuf>,
    output_dir: PathBuf,
    config: AppConfig,
    logs: Vec<String>,
    results: Vec<ResultItem>,
    is_processing: bool,
    status: String,
    #[serde(skip)]
    models: Option<Arc<Mutex<Models>>>,
    #[serde(skip)]
    runtime: Option<Handle>,
    #[serde(skip)]
    worker_rx: Option<Receiver<WorkerMessage>>,
    #[serde(skip)]
    style_ready: bool,
}

impl Default for MitApp {
    fn default() -> Self {
        Self {
            files: Vec::new(),
            output_dir: PathBuf::from("results"),
            config: AppConfig::default(),
            logs: vec!["UI ready.".to_owned()],
            results: Vec::new(),
            is_processing: false,
            status: "Idle".to_owned(),
            models: None,
            runtime: None,
            worker_rx: None,
            style_ready: false,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub target_language: TargetLanguage,
    pub renderer: UiRenderer,
    pub openai: OpenAiCompatibleConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            target_language: TargetLanguage::ChineseSimplified,
            renderer: UiRenderer::Png,
            openai: OpenAiCompatibleConfig::default(),
        }
    }
}

impl AppConfig {
    fn runtime_settings(&self) -> RuntimeSettings {
        let mut settings = RuntimeSettings::default();
        settings.translator.target = serde_json::from_value(
            serde_json::json!({ "translator": "OpenAICompatible", "target": self.target_language.code() }),
        )
        .unwrap_or_default();
        settings.translator.openai_compatible = self.openai.to_runtime_settings();
        settings.render.renderer = self.renderer.into();
        settings
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetLanguage {
    ChineseSimplified,
    ChineseTraditional,
    English,
    Japanese,
    Korean,
    French,
    German,
    Spanish,
}

impl TargetLanguage {
    pub(super) const ALL: [TargetLanguage; 8] = [
        TargetLanguage::ChineseSimplified,
        TargetLanguage::ChineseTraditional,
        TargetLanguage::English,
        TargetLanguage::Japanese,
        TargetLanguage::Korean,
        TargetLanguage::French,
        TargetLanguage::German,
        TargetLanguage::Spanish,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            TargetLanguage::ChineseSimplified => "简体中文",
            TargetLanguage::ChineseTraditional => "繁体中文",
            TargetLanguage::English => "英语",
            TargetLanguage::Japanese => "日语",
            TargetLanguage::Korean => "韩语",
            TargetLanguage::French => "法语",
            TargetLanguage::German => "德语",
            TargetLanguage::Spanish => "西班牙语",
        }
    }

    fn code(self) -> &'static str {
        match self {
            TargetLanguage::ChineseSimplified => "chs",
            TargetLanguage::ChineseTraditional => "cht",
            TargetLanguage::English => "en",
            TargetLanguage::Japanese => "ja",
            TargetLanguage::Korean => "ko",
            TargetLanguage::French => "fr",
            TargetLanguage::German => "de",
            TargetLanguage::Spanish => "es",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UiRenderer {
    Png,
    Html,
    Raw,
}

impl UiRenderer {
    pub(super) const ALL: [UiRenderer; 3] = [UiRenderer::Png, UiRenderer::Html, UiRenderer::Raw];

    pub(super) fn label(self) -> &'static str {
        match self {
            UiRenderer::Png => "PNG 图片",
            UiRenderer::Html => "HTML",
            UiRenderer::Raw => "MIT 二进制",
        }
    }
}

impl From<UiRenderer> for Renderer {
    fn from(value: UiRenderer) -> Self {
        match value {
            UiRenderer::Png => Renderer::Png,
            UiRenderer::Html => Renderer::Html,
            UiRenderer::Raw => Renderer::Raw,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenAiCompatibleConfig {
    pub preset: ProviderPreset,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub system_prompt: String,
    pub user_prompt: String,
    pub temperature: f32,
    pub top_p: f32,
    pub timeout_seconds: u64,
}

impl Default for OpenAiCompatibleConfig {
    fn default() -> Self {
        Self {
            preset: ProviderPreset::DeepSeek,
            base_url: ProviderPreset::DeepSeek
                .base_url()
                .unwrap_or_default()
                .to_owned(),
            api_key: String::new(),
            model: "deepseek-chat".to_owned(),
            system_prompt: OpenAICompatibleSettings::default().system_prompt,
            user_prompt: OpenAICompatibleSettings::default().user_prompt_template,
            temperature: 0.2,
            top_p: 1.0,
            timeout_seconds: 120,
        }
    }
}

impl OpenAiCompatibleConfig {
    fn to_runtime_settings(&self) -> OpenAICompatibleSettings {
        OpenAICompatibleSettings {
            provider_preset: self.preset,
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            model: self.model.clone(),
            system_prompt: self.system_prompt.clone(),
            user_prompt_template: self.user_prompt.clone(),
            temperature: Some(self.temperature),
            top_p: Some(self.top_p),
            timeout_secs: self.timeout_seconds,
        }
    }

    pub fn apply_preset_base_url(&mut self) {
        if let Some(base_url) = self.preset.base_url() {
            self.base_url = base_url.to_owned();
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ResultItem {
    input: PathBuf,
    output: Option<PathBuf>,
    status: ResultStatus,
    message: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ResultStatus {
    Pending,
    Running,
    Done,
    Skipped,
    Failed,
}

impl ResultStatus {
    pub(super) fn label(&self) -> &'static str {
        match self {
            ResultStatus::Pending => "等待中",
            ResultStatus::Running => "处理中",
            ResultStatus::Done => "已完成",
            ResultStatus::Skipped => "已跳过",
            ResultStatus::Failed => "失败",
        }
    }
}

enum WorkerMessage {
    Log(String),
    Status(String),
    Result(ResultItem),
    Finished,
}

impl MitApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        models: Arc<Mutex<Models>>,
        runtime: Handle,
    ) -> Self {
        let mut app = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Self::default()
        };
        app.models = Some(models);
        app.runtime = Some(runtime);
        app.style_ready = false;
        app.files.retain(|v| v.exists());
        app.load_config_from_disk(false);
        app
    }

    fn config_path() -> PathBuf {
        PathBuf::from(CONFIG_PATH)
    }

    pub(super) fn add_files(&mut self, paths: Vec<PathBuf>) {
        for path in paths {
            if !self.files.iter().any(|v| v == &path) {
                self.files.push(path);
            }
        }
        self.log(format!("已选择 {} 个文件。", self.files.len()));
    }

    pub(super) fn add_folder(&mut self, dir: PathBuf) {
        let mut found = Vec::new();
        for entry in walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_map(|v| v.ok())
        {
            let path = entry.path();
            if path.is_file() && is_supported_image(path) {
                found.push(path.to_path_buf());
            }
        }
        let count = found.len();
        self.add_files(found);
        self.log(format!("已从 {} 添加 {count} 张图片。", dir.display()));
    }

    pub(super) fn save_config_to_disk(&mut self) {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            if let Err(err) = create_dir_all(parent) {
                self.log(format!("Failed to create config directory: {err}"));
                return;
            }
        }
        let save_result = serde_json::to_string_pretty(&self.config)
            .map_err(anyhow::Error::from)
            .and_then(|json| {
                File::create(&path)?.write_all(json.as_bytes())?;
                Ok(())
            });
        match save_result {
            Ok(()) => self.log(format!("配置已保存到 {}。", path.display())),
            Err(err) => self.log(format!("配置保存失败：{err}")),
        }
    }

    pub(super) fn load_config_from_disk(&mut self, report_missing: bool) {
        let path = Self::config_path();
        match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<AppConfig>(&text) {
                Ok(config) => {
                    self.config = config;
                    self.log(format!("已从 {} 加载配置。", path.display()));
                }
                Err(err) => self.log(format!("配置解析失败：{err}")),
            },
            Err(err) if report_missing => {
                self.log(format!("无法加载 {}：{err}", path.display()));
            }
            Err(_) => {}
        }
    }

    pub(super) fn start_processing(&mut self) {
        if self.is_processing {
            self.log("已有翻译任务正在运行。");
            return;
        }
        if self.files.is_empty() {
            self.log("请先选择至少一张图片。");
            return;
        }
        let Some(models) = self.models.clone() else {
            self.log("模型尚不可用。");
            return;
        };
        let Some(runtime) = self.runtime.clone() else {
            self.log("Tokio 运行时不可用。");
            return;
        };

        let files = self.files.clone();
        let output_dir = self.output_dir.clone();
        let settings = self.config.runtime_settings();
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.is_processing = true;
        self.status = format!("处理中 0/{}", files.len());
        self.results.clear();
        self.results
            .extend(files.iter().cloned().map(|input| ResultItem {
                input,
                output: None,
                status: ResultStatus::Pending,
                message: String::new(),
            }));
        self.log(format!("开始翻译 {} 个文件。", files.len()));

        std::thread::spawn(move || {
            runtime.block_on(async move {
                run_translation_job(files, output_dir, settings, models, tx).await;
            });
        });
    }

    fn poll_worker(&mut self) {
        let mut finished = false;
        if let Some(rx) = self.worker_rx.take() {
            while let Ok(message) = rx.try_recv() {
                match message {
                    WorkerMessage::Log(line) => self.log(line),
                    WorkerMessage::Status(status) => self.status = status,
                    WorkerMessage::Result(item) => self.upsert_result(item),
                    WorkerMessage::Finished => {
                        finished = true;
                        self.is_processing = false;
                        self.status = "已完成".to_owned();
                        self.log("翻译任务已完成。");
                    }
                }
            }
            if !finished {
                self.worker_rx = Some(rx);
            }
        }
    }

    fn upsert_result(&mut self, item: ResultItem) {
        if let Some(existing) = self.results.iter_mut().find(|v| v.input == item.input) {
            *existing = item;
        } else {
            self.results.push(item);
        }
    }

    pub(super) fn log(&mut self, line: impl Into<String>) {
        self.logs.push(line.into());
        if self.logs.len() > 500 {
            let drain = self.logs.len() - 500;
            self.logs.drain(0..drain);
        }
    }
}

impl App for MitApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if !self.style_ready {
            install_chinese_fonts(ctx);
            install_warm_light_theme(ctx);
            self.style_ready = true;
        }
        self.poll_worker();
        if self.is_processing {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }

        egui::TopBottomPanel::top("app_top_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("漫画图片翻译器");
                ui.separator();
                ui.label(RichText::new(&self.status).strong());
                if self.is_processing {
                    ui.spinner();
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.main_app(ui);
        });
    }
}

async fn run_translation_job(
    files: Vec<PathBuf>,
    output_dir: PathBuf,
    settings: RuntimeSettings,
    models: Arc<Mutex<Models>>,
    tx: Sender<WorkerMessage>,
) {
    let total = files.len();
    let out_ext = settings.render.renderer.extension().to_owned();
    for (index, input) in files.into_iter().enumerate() {
        let _ = tx.send(WorkerMessage::Status(format!(
            "处理中 {}/{}",
            index + 1,
            total
        )));
        let _ = tx.send(WorkerMessage::Result(ResultItem {
            input: input.clone(),
            output: None,
            status: ResultStatus::Running,
            message: "正在处理".to_owned(),
        }));

        let result = process_one(&input, &output_dir, &settings, &out_ext, &models).await;
        match result {
            Ok(Some(output)) => {
                let _ = tx.send(WorkerMessage::Log(format!(
                    "完成：{} -> {}",
                    input.display(),
                    output.display()
                )));
                let _ = tx.send(WorkerMessage::Result(ResultItem {
                    input,
                    output: Some(output),
                    status: ResultStatus::Done,
                    message: "已完成".to_owned(),
                }));
            }
            Ok(None) => {
                let _ = tx.send(WorkerMessage::Log(format!(
                    "跳过：{} 未检测到可翻译内容",
                    input.display()
                )));
                let _ = tx.send(WorkerMessage::Result(ResultItem {
                    input,
                    output: None,
                    status: ResultStatus::Skipped,
                    message: "未检测到可翻译内容".to_owned(),
                }));
            }
            Err(err) => {
                let _ = tx.send(WorkerMessage::Log(format!(
                    "失败：{}：{err}",
                    input.display()
                )));
                let _ = tx.send(WorkerMessage::Result(ResultItem {
                    input,
                    output: None,
                    status: ResultStatus::Failed,
                    message: err.to_string(),
                }));
            }
        }
    }
    let _ = tx.send(WorkerMessage::Finished);
}

async fn process_one(
    input: &Path,
    output_dir: &Path,
    settings: &RuntimeSettings,
    out_ext: &str,
    models: &Arc<Mutex<Models>>,
) -> anyhow::Result<Option<PathBuf>> {
    let img = image::open(input)?;
    let exp = {
        let mut models = models.lock().await;
        models.execute(img, settings, None).await?
    };
    let Some(exp) = exp else {
        return Ok(None);
    };

    let mut output = output_dir.join(
        input
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("output")),
    );
    output.set_extension(out_ext);
    if let Some(parent) = output.parent() {
        create_dir_all(parent)?;
    }

    prepare_renderer_assets(&output, &settings.render.renderer)?;
    let data = render_export_bytes_with_settings(exp, settings)?;
    File::create(&output)?.write_all(&data)?;
    Ok(Some(output))
}

fn install_chinese_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "cjk".to_owned(),
        Arc::new(FontData::from_static(include_bytes!(
            "../../../../../python/source/fonts/Arial-Unicode-Regular.ttf"
        ))),
    );

    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, "cjk".to_owned());
    }

    ctx.set_fonts(fonts);
}

fn install_warm_light_theme(ctx: &egui::Context) {
    let mut style: Style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(10.0, 8.0);
    style.spacing.button_padding = Vec2::new(14.0, 7.0);
    style.spacing.interact_size = Vec2::new(44.0, 34.0);
    style.visuals = Visuals::light();
    style.visuals.window_fill = Color32::from_rgb(248, 245, 240);
    style.visuals.panel_fill = Color32::from_rgb(248, 245, 240);
    style.visuals.faint_bg_color = Color32::from_rgb(244, 238, 230);
    style.visuals.extreme_bg_color = Color32::from_rgb(255, 252, 247);
    style.visuals.code_bg_color = Color32::from_rgb(246, 239, 230);
    style.visuals.warn_fg_color = Color32::from_rgb(151, 85, 25);
    style.visuals.error_fg_color = Color32::from_rgb(180, 56, 45);
    style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(255, 252, 247);
    style.visuals.widgets.noninteractive.bg_stroke =
        Stroke::new(1.0, Color32::from_rgb(226, 216, 204));
    style.visuals.widgets.noninteractive.fg_stroke =
        Stroke::new(1.0, Color32::from_rgb(54, 47, 43));
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(239, 232, 222);
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(218, 207, 195));
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(68, 59, 53));
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(229, 216, 201);
    style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(187, 150, 120));
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(205, 111, 71);
    style.visuals.widgets.active.bg_stroke = Stroke::new(1.0, Color32::from_rgb(161, 79, 47));
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    style.visuals.widgets.open.bg_fill = Color32::from_rgb(239, 232, 222);
    style.visuals.selection.bg_fill = Color32::from_rgb(205, 111, 71);
    style.visuals.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    style.visuals.hyperlink_color = Color32::from_rgb(173, 83, 45);
    style.visuals.window_shadow = Shadow {
        offset: [0, 10],
        blur: 24,
        spread: 0,
        color: Color32::from_black_alpha(18),
    };
    ctx.set_style(style);
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .map(|v| v.to_string_lossy().to_lowercase())
        .map(|ext| IMAGE_EXTS.contains(&ext.as_str()))
        .unwrap_or(false)
}
