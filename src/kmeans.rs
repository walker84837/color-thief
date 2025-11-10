use super::{Color, ColorFormat, PaletteGenerator};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use thiserror::Error;

pub struct KMeans;

/// Represents an error that can occur during the K-Means algorithm.
#[derive(Debug, Error)]
#[error("k-means error")]
pub struct KMeansError;

impl PaletteGenerator for KMeans {
    type Error = KMeansError;

    fn generate_palette(
        &self,
        pixels: &[u8],
        color_format: ColorFormat,
        quality: u8,
        max_colors: u8,
    ) -> Result<Vec<Color>, Self::Error> {
        assert!(quality > 0 && quality <= 10);
        assert!(max_colors > 1);

        let colors_count = color_format.channels();
        let step = quality as usize;

        // Estimate final length of `samples` from the for loop
        let estimated_capacity = pixels.len() / (colors_count * step);
        let mut samples = Vec::with_capacity(estimated_capacity);

        for i in (0..pixels.len()).step_by(colors_count * step) {
            if i + colors_count > pixels.len() {
                break;
            }
            let (r, g, b, a) = color_parts(pixels, color_format, i);
            if a >= 125 && !(r > 250 && g > 250 && b > 250) {
                samples.push(Color::new(r, g, b));
            }
        }

        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let k = max_colors as usize;

        let max_iter = 100;
        let centroids = kmeans(&samples, k, max_iter);

        let mut seen = HashSet::with_capacity(k);
        let mut unique_centroids = Vec::with_capacity(centroids.len());
        for color in centroids {
            if seen.insert(color) {
                unique_centroids.push(color);
            }
        }

        unique_centroids.truncate(max_colors as usize);
        Ok(unique_centroids)
    }
}

fn kmeans(samples: &[Color], k: usize, max_iter: usize) -> Vec<Color> {
    let mut centroids = initialize_centroids(samples, k);
    let mut assignments = vec![0; samples.len()];
    let mut rng = rand::rng();

    for _ in 0..max_iter {
        let changed = assign_clusters(samples, &centroids, &mut assignments);
        if !changed {
            break;
        }
        centroids = update_centroids(samples, &assignments, k, &mut rng);
    }

    centroids
}

fn initialize_centroids(samples: &[Color], k: usize) -> Vec<Color> {
    let mut rng = rand::rng();
    let mut centroids = Vec::with_capacity(k);
    let mut indices: Vec<usize> = (0..samples.len()).collect();
    indices.shuffle(&mut rng);

    for &i in indices.iter().take(k) {
        centroids.push(samples[i]);
    }

    let mut unique = HashSet::with_capacity(k);
    for color in &centroids {
        unique.insert(*color);
    }
    while unique.len() < k {
        if let Some(&sample) = samples.choose(&mut rng) {
            if unique.insert(sample) {
                centroids.push(sample);
            }
        } else {
            break;
        }
    }

    centroids.truncate(k);
    centroids
}

fn assign_clusters(samples: &[Color], centroids: &[Color], assignments: &mut [usize]) -> bool {
    let new_assignments: Vec<usize> = samples
        .par_iter()
        .map(|sample| {
            let mut min_dist = f32::INFINITY;
            let mut best_cluster = 0;
            for (cluster_idx, centroid) in centroids.iter().enumerate() {
                let dist = color_distance(sample, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = cluster_idx;
                }
            }
            best_cluster
        })
        .collect();

    let mut changed = false;
    for (i, &new_assignment) in new_assignments.iter().enumerate() {
        if assignments[i] != new_assignment {
            assignments[i] = new_assignment;
            changed = true;
        }
    }
    changed
}

fn color_distance(c1: &Color, c2: &Color) -> f32 {
    let dr = c1.r as f32 - c2.r as f32;
    let dg = c1.g as f32 - c2.g as f32;
    let db = c1.b as f32 - c2.b as f32;
    dr * dr + dg * dg + db * db
}

fn update_centroids(
    samples: &[Color],
    assignments: &[usize],
    k: usize,
    rng: &mut ThreadRng,
) -> Vec<Color> {
    let mut sums = vec![(0.0, 0.0, 0.0); k];
    let mut counts = vec![0; k];

    for (sample, &cluster) in samples.iter().zip(assignments) {
        sums[cluster].0 += sample.r as f32;
        sums[cluster].1 += sample.g as f32;
        sums[cluster].2 += sample.b as f32;
        counts[cluster] += 1;
    }

    let mut new_centroids = Vec::with_capacity(k);
    for i in 0..k {
        if counts[i] == 0 {
            if let Some(&sample) = samples.choose(rng) {
                new_centroids.push(sample);
            } else {
                new_centroids.push(Color::new(0, 0, 0));
            }
        } else {
            let r = (sums[i].0 / counts[i] as f32).round() as u8;
            let g = (sums[i].1 / counts[i] as f32).round() as u8;
            let b = (sums[i].2 / counts[i] as f32).round() as u8;
            new_centroids.push(Color::new(r, g, b));
        }
    }

    new_centroids
}

fn color_parts(pixels: &[u8], color_format: ColorFormat, pos: usize) -> (u8, u8, u8, u8) {
    match color_format {
        ColorFormat::Rgb => (pixels[pos], pixels[pos + 1], pixels[pos + 2], 255),
        ColorFormat::Rgba => (
            pixels[pos],
            pixels[pos + 1],
            pixels[pos + 2],
            pixels[pos + 3],
        ),
        ColorFormat::Argb => (
            pixels[pos + 1],
            pixels[pos + 2],
            pixels[pos + 3],
            pixels[pos],
        ),
        ColorFormat::Bgr => (pixels[pos + 2], pixels[pos + 1], pixels[pos], 255),
        ColorFormat::Bgra => (
            pixels[pos + 2],
            pixels[pos + 1],
            pixels[pos],
            pixels[pos + 3],
        ),
    }
}
