use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::Path;

use color_thief_ng::{Algorithm, ColorFormat};

fn get_image_buffer(img: image::DynamicImage) -> Vec<u8> {
    match img {
        image::DynamicImage::ImageRgb8(buffer) => buffer.to_vec(),
        _ => unreachable!(),
    }
}

fn bench_q1(c: &mut Criterion) {
    let img = image::open(Path::new("images/photo1.jpg")).unwrap();
    let pixels = get_image_buffer(img);
    c.bench_function("MMCQ/q1", |b| {
        b.iter(|| {
            color_thief_ng::get_palette(
                black_box(Algorithm::Mmcq),
                black_box(&pixels),
                black_box(ColorFormat::Rgb),
                black_box(1),
                black_box(10),
            )
        })
    });
}

fn bench_q10(c: &mut Criterion) {
    let img = image::open(Path::new("images/photo1.jpg")).unwrap();
    let pixels = get_image_buffer(img);
    c.bench_function("MMCQ/q10", |b| {
        b.iter(|| {
            color_thief_ng::get_palette(
                black_box(Algorithm::Mmcq),
                black_box(&pixels),
                black_box(ColorFormat::Rgb),
                black_box(10),
                black_box(10),
            )
        })
    });
}

fn bench_q1_kmeans(c: &mut Criterion) {
    let img = image::open(Path::new("images/photo1.jpg")).unwrap();
    let pixels = get_image_buffer(img);
    c.bench_function("KMeans/q1", |b| {
        b.iter(|| {
            color_thief_ng::get_palette(
                black_box(Algorithm::KMeans {
                    max_iterations: 100,
                    seed: Some(0),
                }),
                black_box(&pixels),
                black_box(ColorFormat::Rgb),
                black_box(1),
                black_box(10),
            )
        })
    });
}

fn bench_q10_kmeans(c: &mut Criterion) {
    let img = image::open(Path::new("images/photo1.jpg")).unwrap();
    let pixels = get_image_buffer(img);
    c.bench_function("KMeans/q10", |b| {
        b.iter(|| {
            color_thief_ng::get_palette(
                black_box(Algorithm::KMeans {
                    max_iterations: 100,
                    seed: Some(0),
                }),
                black_box(&pixels),
                black_box(ColorFormat::Rgb),
                black_box(10),
                black_box(10),
            )
        })
    });
}

fn bench_q1_octree(c: &mut Criterion) {
    let img = image::open(Path::new("images/photo1.jpg")).unwrap();
    let pixels = get_image_buffer(img);
    c.bench_function("Octree/q1", |b| {
        b.iter(|| {
            color_thief_ng::get_palette(
                black_box(Algorithm::Octree { max_depth: None }),
                black_box(&pixels),
                black_box(ColorFormat::Rgb),
                black_box(1),
                black_box(10),
            )
        })
    });
}

fn bench_q10_octree(c: &mut Criterion) {
    let img = image::open(Path::new("images/photo1.jpg")).unwrap();
    let pixels = get_image_buffer(img);
    c.bench_function("Octree/q10", |b| {
        b.iter(|| {
            color_thief_ng::get_palette(
                black_box(Algorithm::Octree { max_depth: None }),
                black_box(&pixels),
                black_box(ColorFormat::Rgb),
                black_box(10),
                black_box(10),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_q1,
    bench_q10,
    bench_q1_kmeans,
    bench_q10_kmeans,
    bench_q1_octree,
    bench_q10_octree
);
criterion_main!(benches);
