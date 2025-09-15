use criterion::{criterion_group, criterion_main, Criterion};
use interface_image::{CpuImageProcessor, ImageOp as _, Interpolation, Mask, RayonImageProcessor};

fn bench_resize_cpu(
    processor: &mut CpuImageProcessor,
    image: &mut Mask,
    interpolation: Interpolation,
) {
    processor
        .resize_mask(
            image,
            image.width as usize * 2,
            image.height as usize * 2,
            interpolation,
        )
        .unwrap();
}

fn bench_resize_rayon(
    processor: &mut RayonImageProcessor,
    image: &mut Mask,
    interpolation: Interpolation,
) {
    processor
        .resize_mask(
            image,
            image.width as usize * 2,
            image.height as usize * 2,
            interpolation,
        )
        .unwrap();
}

#[cfg(feature = "gpu")]
fn bench_resize_gpu(
    processor: &mut crate::image::GpuImageProcessor,
    image: &Mask,
    interpolation: Interpolation,
) {
    processor.resize_mask(
        image.data.clone(),
        image.width as usize,
        image.height as usize,
        image.width as usize * 2,
        image.height as usize * 2,
        interpolation,
    );
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut image = Mask {
        width: 2000,
        height: 2000,
        data: vec![0; 2000 * 2000],
    };

    let mut cpu_processor = CpuImageProcessor::default();
    let mut rayon_processor = RayonImageProcessor::default();
    #[cfg(feature = "gpu")]
    let mut gpu_processor = crate::image::GpuImageProcessor::new();

    c.bench_function("resize_mask_bilinear_cpu", |b| {
        b.iter(|| bench_resize_cpu(&mut cpu_processor, &mut image, Interpolation::Bilinear))
    });
    c.bench_function("resize_mask_bilinear_rayon", |b| {
        b.iter(|| bench_resize_rayon(&mut rayon_processor, &mut image, Interpolation::Bilinear))
    });
    #[cfg(feature = "gpu")]
    c.bench_function("resize_mask_bilinear_gpu", |b| {
        b.iter(|| bench_resize_gpu(&mut gpu_processor, &image, Interpolation::Bilinear))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
