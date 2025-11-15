extern crate color_thief;
extern crate image;

use color_thief::{Algorithm, Color, ColorFormat};
use std::path;

fn get_image_buffer(img: image::DynamicImage) -> (Vec<u8>, ColorFormat) {
    match img {
        image::DynamicImage::ImageRgb8(buffer) => (buffer.to_vec(), ColorFormat::Rgb),
        image::DynamicImage::ImageRgba8(buffer) => (buffer.to_vec(), ColorFormat::Rgba),
        _ => unreachable!(),
    }
}

#[allow(dead_code)]
fn assert_color_approx(left: Color, right: Color, tolerance: u8) {
    assert!(
        left.r.abs_diff(right.r) <= tolerance
            && left.g.abs_diff(right.g) <= tolerance
            && left.b.abs_diff(right.b) <= tolerance,
        "Color mismatch: {:?} vs {:?} (tolerance {})",
        left,
        right,
        tolerance
    );
}

#[test]
fn image1_mmcq() {
    let img = image::open(&path::Path::new("images/photo1.jpg")).unwrap();
    let (buffer, color_type) = get_image_buffer(img);
    let colors = color_thief::get_palette(Algorithm::Mmcq, &buffer, color_type, 10, 10).unwrap();

    println!("{:?}", colors);

    // TODO: check with JavaScript implementation of MMCQ
    // assert!(colors[0].r < 60 && colors[0].g < 50 && colors[0].b < 40);
    // assert!(colors.iter().any(|c| c.r > 200 && c.g > 190 && c.b > 120)); // Light colors
    // assert!(colors.iter().any(|c| c.g > 190 && c.b > 200)); // Blue-green colors
    // assert!(colors.iter().any(|c| c.r > 200 && c.g < 100 && c.b < 20)); // Red accent
}

#[test]
fn image1_kmeans() {
    let img = image::open(&path::Path::new("images/photo1.jpg")).unwrap();
    let (buffer, color_type) = get_image_buffer(img);
    let colors = color_thief::get_palette(
        Algorithm::KMeans {
            max_iterations: 100,
            seed: Some(0),
        },
        &buffer,
        color_type,
        10,
        10,
    )
    .unwrap();

    // Verify color clusters exist in expected ranges
    assert!(colors.iter().any(|c| c.r < 60 && c.g < 50 && c.b < 40)); // Dark colors
    assert!(colors.iter().any(|c| c.r > 200 && c.g > 190 && c.b > 120)); // Light colors
    assert!(colors.iter().any(|c| c.g > 190 && c.b > 200)); // Blue-green colors
}

#[test]
fn image2_mmcq() {
    let img = image::open(&path::Path::new("images/iguana.png")).unwrap();
    let (buffer, color_type) = get_image_buffer(img);
    let colors = color_thief::get_palette(Algorithm::Mmcq, &buffer, color_type, 10, 10).unwrap();

    // Verify we have expected color groups
    assert!(colors.iter().any(|c| c.r > 200 && c.g > 200 && c.b > 200)); // White/gray
    assert!(colors.iter().any(|c| c.r < 80 && c.g < 70 && c.b < 60)); // Dark brown
    assert!(colors.iter().any(|c| c.r > 170 && c.g > 140 && c.b > 110)); // Light brown
}

#[test]
fn image2_kmeans() {
    let img = image::open(&path::Path::new("images/iguana.png")).unwrap();
    let (buffer, color_type) = get_image_buffer(img);
    let colors = color_thief::get_palette(
        Algorithm::KMeans {
            max_iterations: 100,
            seed: Some(0),
        },
        &buffer,
        color_type,
        10,
        10,
    )
    .unwrap();

    // Verify key color groups exist
    assert!(colors.iter().any(|c| c.r > 200 && c.g > 200)); // White/gray
    assert!(colors.iter().any(|c| c.r < 80 && c.g < 70 && c.b < 60)); // Dark brown
    assert!(colors.iter().any(|c| c.r > 170 && c.g > 140 && c.b > 110)); // Light brown
}
