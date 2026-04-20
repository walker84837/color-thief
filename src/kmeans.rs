use super::{Color, ColorFormat, PaletteGenerator, pixel};
use rand::{Rng, SeedableRng, rngs::StdRng};
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

struct KMeansBuffers {
    sums_r: Vec<u64>,
    sums_g: Vec<u64>,
    sums_b: Vec<u64>,
    counts: Vec<usize>,
}

impl KMeansBuffers {
    fn new(k: usize) -> Self {
        Self {
            sums_r: vec![0u64; k],
            sums_g: vec![0u64; k],
            sums_b: vec![0u64; k],
            counts: vec![0usize; k],
        }
    }

    fn clear(&mut self) {
        self.sums_r.fill(0);
        self.sums_g.fill(0);
        self.sums_b.fill(0);
        self.counts.fill(0);
    }
}

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

            let (r, g, b, a) = pixel::color_parts(pixels, color_format, i);
            if !pixel::should_skip_pixel(r, g, b, a) {
                samples.push(Color::new(r, g, b));
            }
        }

        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let k = max_colors as usize;
        let centroids = kmeans(&samples, k, self.max_iterations, self.seed);

        // De-duplicate and truncate
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

    // Clamp k to sample count
    let k = k.min(samples.len());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(rand::rng().next_u64()),
    };

    let mut centroids = initialize_centroids(samples, k, &mut rng);
    let mut assignments = vec![usize::MAX; samples.len()];

    // Track which samples have been assigned (avoids scanning all samples each iteration)
    let mut assigned_indices: Vec<usize> = Vec::new();

    // Preallocate buffers for update_centroids and reuse each iteration
    let mut buffers = KMeansBuffers::new(k);

    // Batch size scales with dataset: 20% for tiny, 2% for medium, 0.5% for large
    // Minimum is k (one sample per cluster) so tiny images converge correctly
    let batch_size = match samples.len() {
        n if n <= 1000 => n.div_ceil(5).max(k).min(n),
        n if n <= 100_000 => (n / 50).max(k),
        n => (n / 200).clamp(k.max(500), 5000),
    };

    let mut prev_centroids = Vec::with_capacity(k);

    for iter in 0..max_iter {
        // Use mini-batch for faster updates
        let changed = assign_clusters_mini_batch(
            samples,
            &centroids,
            &mut assignments,
            &mut assigned_indices,
            batch_size,
            &mut rng,
        );
        if !changed {
            break;
        }

        prev_centroids.clone_from(&centroids);
        centroids = update_centroids_mini_batch(
            samples,
            &assignments,
            &assigned_indices,
            k,
            &mut rng,
            &mut buffers,
        );

        // Early convergence check: if centroids haven't moved much
        if iter > 0 && centroids_converged(&prev_centroids, &centroids, 2) {
            break;
        }
    }

    // Final full assignment: every sample influences centroids by finding nearest
    assignments
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, dest)| {
            let (best, _) = find_closest_centroid(&samples[idx], &centroids);
            *dest = best;
        });

    assigned_indices.clear();
    assigned_indices.extend(0..samples.len());

    centroids = update_centroids_mini_batch(
        samples,
        &assignments,
        &assigned_indices,
        k,
        &mut rng,
        &mut buffers,
    );

    centroids
}

fn initialize_centroids(samples: &[Color], k: usize, rng: &mut StdRng) -> Vec<Color> {
    let k_eff = k.min(samples.len());

    // Use k-means++ style initialization for better convergence
    let mut centroids = Vec::with_capacity(k);

    // Choose first centroid randomly
    if !samples.is_empty() {
        let idx = rng.next_u64() as usize % samples.len();
        centroids.push(samples[idx]);
    }

    // Reusable distance buffer (avoids reallocation per iteration)
    let mut distances = Vec::with_capacity(samples.len());

    // Choose remaining centroids with probability proportional to distance squared
    while centroids.len() < k_eff {
        // Order-preserving parallel distance computation for large datasets
        let total_distance = if samples.len() > 1000 {
            // Ensure buffer has correct length - subsequent iterations reuse without resize
            if distances.len() != samples.len() {
                distances.resize(samples.len(), 0);
            }
            samples
                .par_iter()
                .map(|sample| min_distance_to_centroids(&centroids, sample) as u64)
                .collect_into_vec(&mut distances);
            distances.iter().sum()
        } else {
            distances.clear();
            let mut total = 0u64;
            for sample in samples {
                let min_dist_sq = min_distance_to_centroids(&centroids, sample);
                distances.push(min_dist_sq as u64);
                total += min_dist_sq as u64;
            }
            total
        };

        if total_distance == 0 {
            // All points are the same, pick randomly
            if !samples.is_empty() {
                let idx = rng.next_u64() as usize % samples.len();
                centroids.push(samples[idx]);
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
        if !samples.is_empty() {
            let idx = rng.next_u64() as usize % samples.len();
            let s = samples[idx];
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
    assigned_indices: &mut Vec<usize>,
    batch_size: usize,
    rng: &mut StdRng,
) -> bool {
    // Randomly select a mini-batch - use pre-allocated indices to reduce allocations
    let mut batch_indices = Vec::with_capacity(batch_size);
    let sample_len = samples.len();
    for _ in 0..batch_size {
        let idx = rng.next_u64() as usize % sample_len;
        batch_indices.push(idx);
    }

    // Find closest centroid for each sample in the batch
    let new_assignments: Vec<(usize, usize)> = batch_indices
        .iter()
        .map(|&idx| {
            let sample = &samples[idx];
            let (best_cluster, _min_dist) = find_closest_centroid(sample, centroids);
            (idx, best_cluster)
        })
        .collect();

    let mut changed = false;
    for (idx, best_cluster) in new_assignments {
        if assignments[idx] != best_cluster {
            // Track first assignment to avoid scanning all samples in the update loop
            if assignments[idx] == usize::MAX {
                assigned_indices.push(idx);
            }
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
    assigned_indices: &[usize],
    k: usize,
    rng: &mut StdRng,
    buffers: &mut KMeansBuffers,
) -> Vec<Color> {
    // Clear accumulators
    buffers.clear();

    // Accumulate assigned samples into centroid sums (skips unassigned samples)
    for &idx in assigned_indices {
        let cluster = assignments[idx];
        buffers.counts[cluster] += 1;
        buffers.sums_r[cluster] += samples[idx].r as u64;
        buffers.sums_g[cluster] += samples[idx].g as u64;
        buffers.sums_b[cluster] += samples[idx].b as u64;
    }

    let mut new_centroids = Vec::with_capacity(k);
    for i in 0..k {
        if buffers.counts[i] == 0 {
            // Replace empty cluster with a random sample
            if !samples.is_empty() {
                let idx = rng.next_u64() as usize % samples.len();
                new_centroids.push(samples[idx]);
            } else {
                new_centroids.push(Color::new(0, 0, 0));
            }
        } else {
            // Integer average with rounding
            let half_count = buffers.counts[i] as u64 >> 1;
            let count = buffers.counts[i] as u64;
            let r = ((buffers.sums_r[i] + half_count) / count) as u8;
            let g = ((buffers.sums_g[i] + half_count) / count) as u8;
            let b = ((buffers.sums_b[i] + half_count) / count) as u8;
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

const fn color_distance_sq(c1: &Color, c2: &Color) -> u32 {
    let dr = c1.r.abs_diff(c2.r) as u32;
    let dg = c1.g.abs_diff(c2.g) as u32;
    let db = c1.b.abs_diff(c2.b) as u32;

    dr * dr + dg * dg + db * db
}

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
