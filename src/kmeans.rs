use super::{Color, ColorFormat, PaletteGenerator};
use rand::{SeedableRng, prelude::*, rngs::StdRng};
use rayon::prelude::*;
use std::collections::HashSet;
use thiserror::Error;

pub struct KMeans {
    pub max_iterations: usize,
    pub seed: Option<u64>,
}

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

        let centroids = kmeans(&samples, k, self.max_iterations, self.seed);

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

fn kmeans(samples: &[Color], k: usize, max_iter: usize, seed: Option<u64>) -> Vec<Color> {
    if samples.is_empty() {
        return Vec::new();
    }

    // clamp k to sample count
    let k = k.min(samples.len());

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::rng()),
    };

    let mut centroids = initialize_centroids(samples, k, &mut rng);
    let mut assignments = vec![usize::MAX; samples.len()];

    // preallocate buffers for update_centroids and reuse each iteration
    let mut sums_r = vec![0u64; k];
    let mut sums_g = vec![0u64; k];
    let mut sums_b = vec![0u64; k];
    let mut counts = vec![0usize; k];

    for _ in 0..max_iter {
        let changed = assign_clusters(samples, &centroids, &mut assignments);
        if !changed {
            break;
        }
        centroids = update_centroids(
            samples,
            &assignments,
            k,
            &mut rng,
            &mut sums_r,
            &mut sums_g,
            &mut sums_b,
            &mut counts,
        );
    }

    centroids
}

fn initialize_centroids(samples: &[Color], k: usize, rng: &mut dyn RngCore) -> Vec<Color> {
    let k_eff = k.min(samples.len());
    let mut centroids: Vec<Color> = samples.choose_multiple(rng, k_eff).cloned().collect();

    // If k > samples.len() (shouldn't happen often) fill up with random choices
    // but ensure uniqueness until we either reach k or run out of unique samples.
    if centroids.len() < k {
        let mut seen = HashSet::with_capacity(k);
        for &c in &centroids {
            seen.insert(c);
        }
        while centroids.len() < k {
            if let Some(&s) = samples.choose(rng) {
                if seen.insert(s) {
                    centroids.push(s);
                } else {
                    // if we can't find new unique quickly, break to avoid infinite loop
                    if seen.len() == samples.len() {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    centroids.truncate(k);
    centroids
}

fn assign_clusters(samples: &[Color], centroids: &[Color], assignments: &mut [usize]) -> bool {
    let new_assignments: Vec<usize> = samples
        .par_iter()
        .map(|sample| {
            let mut min_dist = u32::MAX;
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

#[inline]
fn color_distance(c1: &Color, c2: &Color) -> u32 {
    let dr = c1.r.abs_diff(c2.r) as u32;
    let dg = c1.g.abs_diff(c2.g) as u32;
    let db = c1.b.abs_diff(c2.b) as u32;
    dr * dr + dg * dg + db * db
}

fn update_centroids(
    samples: &[Color],
    assignments: &[usize],
    k: usize,
    rng: &mut dyn RngCore,
    sums_r: &mut [u64],
    sums_g: &mut [u64],
    sums_b: &mut [u64],
    counts: &mut [usize],
) -> Vec<Color> {
    debug_assert_eq!(sums_r.len(), k);
    debug_assert_eq!(sums_g.len(), k);
    debug_assert_eq!(sums_b.len(), k);
    debug_assert_eq!(counts.len(), k);

    // clear accumulators
    for i in 0..k {
        sums_r[i] = 0;
        sums_g[i] = 0;
        sums_b[i] = 0;
        counts[i] = 0;
    }

    for (s, &cluster) in samples.iter().zip(assignments.iter()) {
        counts[cluster] += 1;
        sums_r[cluster] += s.r as u64;
        sums_g[cluster] += s.g as u64;
        sums_b[cluster] += s.b as u64;
    }

    let mut new_centroids = Vec::with_capacity(k);
    for i in 0..k {
        if counts[i] == 0 {
            // replace empty cluster with a random sample
            if let Some(&s) = samples.choose(rng) {
                new_centroids.push(s);
            } else {
                new_centroids.push(Color::new(0, 0, 0));
            }
        } else {
            // integer average with rounding
            let r = ((sums_r[i] + (counts[i] as u64 / 2)) / counts[i] as u64) as u8;
            let g = ((sums_g[i] + (counts[i] as u64 / 2)) / counts[i] as u64) as u8;
            let b = ((sums_b[i] + (counts[i] as u64 / 2)) / counts[i] as u64) as u8;
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
