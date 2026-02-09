use super::{Color, ColorFormat, PaletteGenerator};
use rand::{prelude::*, rngs::StdRng, SeedableRng};
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

    // Dynamic batch size: use smaller batches for better convergence
    let batch_size = match samples.len() {
        n if n < 100 => n.min(10),
        n if n < 1000 => (n / 20).max(20).min(50),
        n => (n / 50).max(50).min(200),
    };
    let mut prev_centroids = Vec::with_capacity(k);

    for iter in 0..max_iter {
        // Use mini-batch for faster updates
        let changed =
            assign_clusters_mini_batch(samples, &centroids, &mut assignments, batch_size, &mut rng);
        if !changed {
            break;
        }

        prev_centroids.clone_from(&centroids);
        centroids = update_centroids_mini_batch(
            samples,
            &assignments,
            k,
            &mut rng,
            batch_size,
            &mut sums_r,
            &mut sums_g,
            &mut sums_b,
            &mut counts,
        );

        // Early convergence check: if centroids haven't moved much
        if iter > 0 && centroids_converged(&prev_centroids, &centroids, 2) {
            break;
        }
    }

    centroids
}

fn initialize_centroids(samples: &[Color], k: usize, rng: &mut dyn RngCore) -> Vec<Color> {
    let k_eff = k.min(samples.len());

    // Use k-means++ style initialization for better convergence
    let mut centroids = Vec::with_capacity(k);

    // Choose first centroid randomly
    if let Some(first) = samples.choose(rng) {
        centroids.push(*first);
    }

    // Choose remaining centroids with probability proportional to distance squared
    while centroids.len() < k_eff {
        let mut distances = Vec::with_capacity(samples.len());
        let mut total_distance = 0u64;

        // Calculate minimum distance to existing centroids - parallel for large datasets
        if samples.len() > 1000 {
            let (distances_vec, total) = samples
                .par_iter()
                .map(|sample| min_distance_to_centroids(&centroids, sample) as u64)
                // Parallel fold: accumulate distances and sum totals in each thread
                .fold(
                    || (Vec::new(), 0u64),
                    |(mut dists, total), dist| {
                        dists.push(dist);
                        (dists, total + dist)
                    },
                )
                // Parallel reduce: combine results from all threads
                .reduce(
                    || (Vec::new(), 0u64),
                    |(mut dists1, total1), (dists2, total2)| {
                        dists1.extend(dists2);
                        (dists1, total1 + total2)
                    },
                );
            distances = distances_vec;
            total_distance = total;
        } else {
            // Sequential for small datasets
            for sample in samples {
                let min_dist_sq = min_distance_to_centroids(&centroids, sample);
                distances.push(min_dist_sq as u64);
                total_distance += min_dist_sq as u64;
            }
        }

        if total_distance == 0 {
            // All points are the same, pick randomly
            if let Some(sample) = samples.choose(rng) {
                centroids.push(*sample);
            }
            break;
        } else {
            let mut choice = rng.next_u64() % total_distance;
            for (i, &dist) in distances.iter().enumerate() {
                if choice < dist {
                    centroids.push(samples[i]);
                    break;
                }
                choice -= dist;
            }
        }
    }

    // Fill remaining slots with random choices if needed
    while centroids.len() < k {
        if let Some(&s) = samples.choose(rng) {
            if !centroids.contains(&s) {
                centroids.push(s);
            } else if centroids.len() == samples.len() {
                break;
            }
        } else {
            break;
        }
    }

    centroids.truncate(k);
    centroids
}

fn assign_clusters_mini_batch(
    samples: &[Color],
    centroids: &[Color],
    assignments: &mut [usize],
    batch_size: usize,
    rng: &mut dyn RngCore,
) -> bool {
    // Randomly select a mini-batch - use pre-allocated indices to reduce allocations
    let mut batch_indices = Vec::with_capacity(batch_size);
    let sample_range = 0..samples.len();
    batch_indices.extend(sample_range.choose_multiple(rng, batch_size));

    // Use parallelization for larger batches, sequential for smaller ones
    let new_assignments: Vec<(usize, usize)> = if batch_indices.len() >= 20 {
        // Parallel version for larger batches
        batch_indices
            .par_iter()
            .map(|&idx| {
                let sample = &samples[idx];
                let (best_cluster, _min_dist) = find_closest_centroid(sample, centroids);
                (idx, best_cluster)
            })
            .collect()
    } else {
        // Sequential version for smaller batches (avoids parallel overhead)
        batch_indices
            .iter()
            .map(|&idx| {
                let sample = &samples[idx];
                let (best_cluster, _min_dist) = find_closest_centroid(sample, centroids);
                (idx, best_cluster)
            })
            .collect()
    };

    let mut changed = false;
    for (idx, best_cluster) in new_assignments {
        if assignments[idx] != best_cluster {
            assignments[idx] = best_cluster;
            changed = true;
        }
    }

    changed
}

fn find_closest_centroid(sample: &Color, centroids: &[Color]) -> (usize, u32) {
    let mut min_dist = u32::MAX;
    let mut best_cluster = 0;

    for (cluster_idx, centroid) in centroids.iter().enumerate() {
        let dist = color_distance_sq(sample, centroid);
        if dist < min_dist {
            min_dist = dist;
            best_cluster = cluster_idx;
        }
    }

    (best_cluster, min_dist)
}

fn update_centroids_mini_batch(
    samples: &[Color],
    assignments: &[usize],
    k: usize,
    rng: &mut dyn RngCore,
    _batch_size: usize,
    sums_r: &mut [u64],
    sums_g: &mut [u64],
    sums_b: &mut [u64],
    counts: &mut [usize],
) -> Vec<Color> {
    // Clear accumulators
    for i in 0..k {
        sums_r[i] = 0;
        sums_g[i] = 0;
        sums_b[i] = 0;
        counts[i] = 0;
    }

    // Parallel accumulation for assigned samples
    let assigned_samples: Vec<_> = assignments
        .iter()
        .enumerate()
        .filter_map(|(idx, &cluster)| {
            if cluster != usize::MAX && cluster < k {
                Some((idx, cluster))
            } else {
                None
            }
        })
        .collect();

    if assigned_samples.len() > 100 {
        // Parallel version for larger datasets
        let results: Vec<_> = assigned_samples
            .par_iter()
            .map(|&(idx, cluster)| {
                let sample = samples[idx];
                (cluster, sample.r as u64, sample.g as u64, sample.b as u64)
            })
            .collect();

        for (cluster, r, g, b) in results {
            counts[cluster] += 1;
            sums_r[cluster] += r;
            sums_g[cluster] += g;
            sums_b[cluster] += b;
        }
    } else {
        // Sequential version for smaller datasets
        for &(idx, cluster) in &assigned_samples {
            counts[cluster] += 1;
            sums_r[cluster] += samples[idx].r as u64;
            sums_g[cluster] += samples[idx].g as u64;
            sums_b[cluster] += samples[idx].b as u64;
        }
    }

    let mut new_centroids = Vec::with_capacity(k);
    for i in 0..k {
        if counts[i] == 0 {
            // Replace empty cluster with a random sample
            if let Some(&s) = samples.choose(rng) {
                new_centroids.push(s);
            } else {
                new_centroids.push(Color::new(0, 0, 0));
            }
        } else {
            // Integer average with rounding
            let half_count = counts[i] as u64 >> 1;
            let count = counts[i] as u64;
            let r = ((sums_r[i] + half_count) / count) as u8;
            let g = ((sums_g[i] + half_count) / count) as u8;
            let b = ((sums_b[i] + half_count) / count) as u8;
            new_centroids.push(Color::new(r, g, b));
        }
    }
    new_centroids
}

#[inline]
fn centroids_converged(prev: &[Color], current: &[Color], threshold: u8) -> bool {
    if prev.len() != current.len() {
        return false;
    }

    for (p, c) in prev.iter().zip(current.iter()) {
        let dr = p.r.abs_diff(c.r);
        let dg = p.g.abs_diff(c.g);
        let db = p.b.abs_diff(c.b);
        if dr > threshold || dg > threshold || db > threshold {
            return false;
        }
    }
    true
}

/// Gets the squared distance between two colors
const fn color_distance_sq(c1: &Color, c2: &Color) -> u32 {
    let dr = c1.r.abs_diff(c2.r) as u32;
    let dg = c1.g.abs_diff(c2.g) as u32;
    let db = c1.b.abs_diff(c2.b) as u32;
    dr * dr + dg * dg + db * db
}

/// Extracts RGBA color components from a pixel buffer based on the specified color format.
/// Returns a tuple of (red, green, blue, alpha) values.
const fn color_parts(pixels: &[u8], color_format: ColorFormat, pos: usize) -> (u8, u8, u8, u8) {
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

/// Calculates the minimum squared distance from a sample to all existing centroids.
/// Used in k-means++ initialization for probabilistic centroid selection.
const fn min_distance_to_centroids(centroids: &[Color], sample: &Color) -> u32 {
    let mut i = 0;
    let mut min_dist_sq = u32::MAX;

    while i < centroids.len() {
        let dist = color_distance_sq(sample, &centroids[i]);
        if dist < min_dist_sq {
            min_dist_sq = dist;
        }
        i += 1;
    }

    min_dist_sq
}
