extern crate color_thief_ng;

use color_thief_ng::{Algorithm, Color, ColorFormat};

#[test]
fn synthetic_image_precision() {
    // Create a simple 2x2 image with 4 distinct colors
    let pixels = vec![
        255, 0, 0, // Red
        0, 255, 0, // Green
        0, 0, 255, // Blue
        255, 255, 0, // Yellow
    ];

    let color_format = ColorFormat::Rgb;

    // Test MMCQ
    let colors_mmcq =
        color_thief_ng::get_palette(Algorithm::Mmcq, &pixels, color_format, 1, 4).unwrap();
    // MMCQ should find these colors, though they might be shifted slightly due to SIGNAL_BITS (5 bits)
    // 255 -> 11111000 in binary (if using 5 bits), which is 248.
    // The current implementation shifts by 3 bits.
    // 255 >> 3 = 31.
    // Scaled back: (31 + 0.5) * 8 = 252.
    for c in &colors_mmcq {
        assert!((c.r == 252 || c.r == 4) && (c.g == 252 || c.g == 4) && (c.b == 252 || c.b == 4));
    }
    assert_eq!(colors_mmcq.len(), 4);

    // Test KMeans
    let colors_kmeans = color_thief_ng::get_palette(
        Algorithm::KMeans {
            max_iterations: 100,
            seed: Some(0),
        },
        &pixels,
        color_format,
        1,
        4,
    )
    .unwrap();
    // KMeans should be very precise for small datasets
    assert!(colors_kmeans.contains(&Color::new(255, 0, 0)));
    assert!(colors_kmeans.contains(&Color::new(0, 255, 0)));
    assert!(colors_kmeans.contains(&Color::new(0, 0, 255)));
    assert!(colors_kmeans.contains(&Color::new(255, 255, 0)));
    assert_eq!(colors_kmeans.len(), 4);
}
