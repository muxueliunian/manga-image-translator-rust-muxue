use std::sync::Arc;

use base_util::onnx::all_providers;
use criterion::{criterion_group, criterion_main, Criterion};
use dbnet::DbNetDetector;
use interface_detector::DefaultOptions;
use interface_detector::Detector;
use interface_detector::PreprocessorOptions;
use interface_image::{CpuImageProcessor, ImageOp, RawImage};
use interface_model::Model;
use interface_model::ModelLoad;

fn criterion_benchmark(c: &mut Criterion) {
    let mut data = DbNetDetector::new(all_providers(), false);
    let img = RawImage::new("./imgs/232264684-5a7bcf8e-707b-4925-86b0-4212382f1680.png")
        .expect("Failed to load image");
    let cpu_image_processor =
        Arc::new(CpuImageProcessor::default()) as Arc<dyn ImageOp + Send + Sync>;

    c.bench_function("load_unload", |b| {
        b.iter(|| {
            data.load().expect("Failed to load model");
            data.unload();
        })
    });

    c.bench_function("infer", |b| {
        data.load().expect("Failed to load model");
        b.iter(|| {
            data.infer(
                interface_image::RawImageCow::Borrowed(img.view()),
                DefaultOptions::default(),
                &cpu_image_processor,
            )
        })
    });

    c.bench_function("detection", |b| {
        data.load().expect("Failed to load model");
        b.iter(|| {
            data.detect(
                &img,
                PreprocessorOptions::default(),
                DefaultOptions::default(),
                &cpu_image_processor,
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
