use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use actix_files::NamedFile;
use actix_multipart::form::{tempfile::TempFile, text::Text, MultipartForm};
use actix_web::{
    get, post,
    web::{self, Data},
    App, HttpRequest, HttpResponse, HttpServer, Responder,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{prepare_renderer_assets, render_export_bytes_with_settings, settings, setup::Models};

const UPLOAD_DIR: &str = "./uploads";
const RESULTS_DIR: &str = "./results";
const WEB_INDEX: &str = include_str!("../web/index.html");

struct ApiState {
    models: Arc<Mutex<Models>>,
}

#[get("/")]
async fn index() -> impl Responder {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(WEB_INDEX)
}

#[get("/defaults/settings")]
async fn defaults_settings() -> impl Responder {
    HttpResponse::Ok().json(settings::Settings::default())
}

#[get("/defaults/detector")]
async fn defaults_detector() -> impl Responder {
    HttpResponse::Ok().json(settings::DetectorSettings::default())
}

#[get("/defaults/ocr")]
async fn defaults_ocr() -> impl Responder {
    HttpResponse::Ok().json(settings::OCRSettings::default())
}

#[get("/defaults/inpainter")]
async fn defaults_inpainter() -> impl Responder {
    HttpResponse::Ok().json(settings::InpainterSettings::default())
}

#[get("/defaults/mask_refinement")]
async fn defaults_mask_refinement() -> impl Responder {
    HttpResponse::Ok().json(settings::MaskRefinementSettings::default())
}

#[get("/defaults/translator")]
async fn defaults_translator() -> impl Responder {
    HttpResponse::Ok().json(settings::TranslatorSettings::default())
}

#[get("/image/{uuid}")]
async fn get_image(uuid: web::Path<String>, req: HttpRequest) -> impl Responder {
    let filename = uuid.into_inner();
    if Uuid::parse_str(&filename).is_err() {
        return HttpResponse::BadRequest().body("Invalid UUID");
    }

    let path = PathBuf::from(UPLOAD_DIR).join(&filename);
    named_file(path, &req)
}

#[get("/results/file/{name}")]
async fn get_result(name: web::Path<String>, req: HttpRequest) -> impl Responder {
    let filename = name.into_inner();
    if !is_safe_file_name(&filename) {
        return HttpResponse::BadRequest().body("Invalid result name");
    }

    let path = PathBuf::from(RESULTS_DIR).join(filename);
    named_file(path, &req)
}

fn named_file(path: PathBuf, req: &HttpRequest) -> HttpResponse {
    if !path.exists() {
        return HttpResponse::NotFound().body("File not found");
    }

    match NamedFile::open(path) {
        Ok(file) => file.use_last_modified(true).into_response(req),
        Err(err) => HttpResponse::InternalServerError().body(format!("Failed to read file: {err}")),
    }
}

#[derive(Debug, MultipartForm)]
struct UploadForm {
    file: TempFile,
}

#[post("/image/upload")]
async fn upload_image(MultipartForm(form): MultipartForm<UploadForm>) -> impl Responder {
    if let Err(err) = fs::create_dir_all(UPLOAD_DIR) {
        return HttpResponse::InternalServerError()
            .body(format!("Failed to create upload dir: {err}"));
    }

    let p = form.file.file.path();
    let uuid = Uuid::new_v4().to_string();
    let to = PathBuf::from(UPLOAD_DIR).join(&uuid);
    if let Err(err) = fs::rename(p, to) {
        return HttpResponse::InternalServerError().body(format!("Failed to move file: {err}"));
    }

    HttpResponse::Ok().body(uuid)
}

#[derive(Debug, MultipartForm)]
struct TranslateForm {
    file: TempFile,
    settings: Text<String>,
}

#[derive(Serialize)]
struct TranslateResponse {
    id: String,
    file_name: String,
    result_url: String,
}

#[post("/translate")]
async fn translate(
    state: Data<ApiState>,
    MultipartForm(form): MultipartForm<TranslateForm>,
) -> impl Responder {
    let settings = match serde_json::from_str::<settings::Settings>(&form.settings) {
        Ok(settings) => settings,
        Err(err) => {
            return HttpResponse::BadRequest().body(format!("Invalid settings JSON: {err}"));
        }
    };

    let img = match image::open(form.file.file.path()) {
        Ok(img) => img,
        Err(err) => {
            return HttpResponse::BadRequest().body(format!("Failed to open image: {err}"));
        }
    };

    let id = Uuid::new_v4().to_string();
    let file_name = format!("{id}.{}", settings.render.renderer.extension());
    let result_path = PathBuf::from(RESULTS_DIR).join(&file_name);

    if let Err(err) = fs::create_dir_all(RESULTS_DIR) {
        return HttpResponse::InternalServerError()
            .body(format!("Failed to create results dir: {err}"));
    }

    let export = {
        let mut models = state.models.lock().await;
        match models.execute(img, &settings, None).await {
            Ok(Some(export)) => export,
            Ok(None) => {
                return HttpResponse::UnprocessableEntity()
                    .body("No translatable text was detected in this image");
            }
            Err(err) => {
                return HttpResponse::InternalServerError()
                    .body(format!("Translation failed: {err}"));
            }
        }
    };

    if let Err(err) = prepare_renderer_assets(&result_path, &settings.render.renderer) {
        return HttpResponse::InternalServerError()
            .body(format!("Failed to prepare renderer assets: {err}"));
    }

    let data = match render_export_bytes_with_settings(export, &settings) {
        Ok(data) => data,
        Err(err) => {
            return HttpResponse::InternalServerError().body(format!("Render failed: {err}"));
        }
    };
    if let Err(err) = fs::write(&result_path, data) {
        return HttpResponse::InternalServerError().body(format!("Failed to save result: {err}"));
    }

    HttpResponse::Ok().json(TranslateResponse {
        id,
        file_name: file_name.clone(),
        result_url: format!("/results/file/{file_name}"),
    })
}

#[derive(Serialize)]
struct ResultItem {
    file_name: String,
    url: String,
    modified: u64,
    size: u64,
}

#[get("/results/list")]
async fn results_list() -> impl Responder {
    let mut items = Vec::new();
    let entries = match fs::read_dir(RESULTS_DIR) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return HttpResponse::Ok().json(items);
        }
        Err(err) => {
            return HttpResponse::InternalServerError()
                .body(format!("Failed to read results: {err}"));
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !matches!(
            path.extension().and_then(OsStr::to_str),
            Some("png" | "html" | "bin")
        ) {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(|v| v.to_str()).map(str::to_owned) else {
            continue;
        };

        let Ok(meta) = entry.metadata() else {
            continue;
        };

        let modified = meta
            .modified()
            .ok()
            .and_then(|v| v.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|v| v.as_secs())
            .unwrap_or_default();

        items.push(ResultItem {
            url: format!("/results/file/{file_name}"),
            file_name,
            modified,
            size: meta.len(),
        });
    }

    items.sort_by(|a, b| b.modified.cmp(&a.modified));
    HttpResponse::Ok().json(items)
}

#[derive(Deserialize)]
struct ExportRequest {
    files: Vec<String>,
    output_dir: String,
}

#[derive(Serialize)]
struct ExportResponse {
    copied: usize,
    output_dir: String,
}

#[post("/results/export")]
async fn export_results(req: web::Json<ExportRequest>) -> impl Responder {
    if req.files.is_empty() {
        return HttpResponse::BadRequest().body("No result files selected");
    }

    let output_dir = PathBuf::from(req.output_dir.trim());
    if output_dir.as_os_str().is_empty() {
        return HttpResponse::BadRequest().body("Output directory is empty");
    }

    if let Err(err) = fs::create_dir_all(&output_dir) {
        return HttpResponse::InternalServerError()
            .body(format!("Failed to create output dir: {err}"));
    }

    let mut copied = 0;
    for file in &req.files {
        if !is_safe_file_name(file) {
            return HttpResponse::BadRequest().body(format!("Invalid result name: {file}"));
        }

        let from = PathBuf::from(RESULTS_DIR).join(file);
        if !from.exists() {
            return HttpResponse::NotFound().body(format!("Result not found: {file}"));
        }

        let to = output_dir.join(file);
        if let Err(err) = fs::copy(&from, &to) {
            return HttpResponse::InternalServerError()
                .body(format!("Failed to copy {file}: {err}"));
        }
        copied += 1;
    }

    HttpResponse::Ok().json(ExportResponse {
        copied,
        output_dir: output_dir.display().to_string(),
    })
}

fn is_safe_file_name(name: &str) -> bool {
    let path = Path::new(name);
    !name.is_empty()
        && path.file_name().and_then(|v| v.to_str()) == Some(name)
        && !name.contains(['/', '\\'])
}

pub async fn main(host: &str, port: u16, models: Arc<Mutex<Models>>) -> std::io::Result<()> {
    fs::create_dir_all(UPLOAD_DIR)?;
    fs::create_dir_all(RESULTS_DIR)?;

    let state = Data::new(ApiState { models });
    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .service(defaults_settings)
            .service(defaults_detector)
            .service(defaults_ocr)
            .service(defaults_mask_refinement)
            .service(defaults_translator)
            .service(defaults_inpainter)
            .service(upload_image)
            .service(get_image)
            .service(translate)
            .service(results_list)
            .service(get_result)
            .service(export_results)
            .service(index)
    })
    .bind((host, port))?
    .run()
    .await
}
