// Copyright 2017, Reizner Evgeniy <razrfalcon@gmail.com>.
// See the COPYRIGHT file at the top-level directory of this distribution.
// Licensed under the MIT license, see the LICENSE file or <http://opensource.org/licenses/MIT>

/*!
*color-thief-rs* is a [color-thief](https://github.com/lokesh/color-thief)
algorithm reimplementation in Rust.

The implementation itself is a heavily modified
[Swift version](https://github.com/yamoridon/ColorThiefSwift) of the same algorithm.
*/

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use rand::prelude::*;
use std::cmp;
use std::collections::HashSet;
use std::fmt::Display;

pub use rgb::RGB8 as Color;

const SIGNAL_BITS: i32 = 5;
const RIGHT_SHIFT: i32 = 8 - SIGNAL_BITS;
const MULTIPLIER: i32 = 1 << RIGHT_SHIFT;
const MULTIPLIER_64: f64 = MULTIPLIER as f64;
const HISTOGRAM_SIZE: usize = 1 << (3 * SIGNAL_BITS);
const VBOX_LENGTH: usize = 1 << SIGNAL_BITS;
const FRACTION_BY_POPULATION: f64 = 0.75;
const MAX_ITERATIONS: i32 = 1000;

/// Choice of algorithm for color palette generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Algorithm {
    /// Original MMCQ (Modified Median Cut Quantization) algorithm. This algorithm was used in the original code of `color-thief-rs`.
    ///
    /// *Summary*: Fast, distinct colors, less accurate.
    ///
    /// This algorithm works by recursively dividing the color space into smaller boxes (vboxes)
    /// until the desired number of colors is reached. It's generally faster and effective for
    /// reducing the number of colors while preserving visual quality, especially for images
    /// with distinct color regions. It might be less accurate for images with subtle color gradients.
    Mmcq,
    /// K-means clustering algorithm. This algorithm was added in the fork.
    ///
    /// *Summary*: Slower, subtle gradients, more accurate.
    ///
    /// K-means is an iterative algorithm that partitions `n` observations into `k` clusters,
    /// where each observation belongs to the cluster with the nearest mean (centroid). In the
    /// context of color quantization, it groups similar colors together to form the palette.
    /// K-means can sometimes produce more perceptually uniform palettes than MMCQ, making it
    /// more accurate for images with subtle color variations, but it can also be slower and
    /// more sensitive to the initial choice of centroids.
    KMeans,
}

/// Represent a color format of an underlying image data.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ColorFormat {
    /// Red, Green, Blue color format.
    Rgb,
    /// Red, Green, Blue, Alpha color format.
    Rgba,
    /// Alpha, Red, Green, Blue color format.
    Argb,
    /// Blue, Green, Red color format.
    Bgr,
    /// Blue, Green, Red, Alpha color format.
    Bgra,
}

impl ColorFormat {
    /// Returns the number of channels in a color format.
    fn channels(&self) -> usize {
        match self {
            ColorFormat::Rgb => 3,
            ColorFormat::Rgba => 4,
            ColorFormat::Argb => 4,
            ColorFormat::Bgr => 3,
            ColorFormat::Bgra => 4,
        }
    }
}

/// List of all errors.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Error {
    /// An invalid VBox was encountered. This can happen if:
    /// - a VBox (a representation of a color range) contains no pixels after filtering;
    /// - if the input parameters lead to an impossible VBox state, such as requesting 0 colors for the palette.
    InvalidVBox,
    /// Failed to cut a VBox. This occurs when:
    /// - the algorithm attempts to divide a VBox into two smaller VBoxes but cannot find a suitable split point;
    /// - the VBox contains only a single color, a very small range of colors;
    /// - if the pixel distribution within it prevents a meaningful division.
    VBoxCutFailed,
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let msg = match *self {
            Error::InvalidVBox => "an invalid VBox",
            Error::VBoxCutFailed => "failed to cut a VBox",
        };
        write!(f, "{}", msg)
    }
}

impl std::error::Error for Error {}

/// Returns a representative color palette of an image.
///
/// * `algorithm` - The algorithm to use for palette generation.
/// * `pixels` - A raw image data.
/// * `color_format` - Represent a color format of the image data.
/// * `quality` - Quality of the output (step size for sampling pixels).
/// * `max_colors` - Maximum number of colors in the output palette.
pub fn get_palette(
    algorithm: Algorithm,
    pixels: &[u8],
    color_format: ColorFormat,
    quality: u8,
    max_colors: u8,
) -> Result<Vec<Color>, Error> {
    assert!(quality > 0 && quality <= 10);
    assert!(max_colors > 1);

    match algorithm {
        Algorithm::Mmcq => Mmcq.generate_palette(pixels, color_format, quality, max_colors),
        Algorithm::KMeans => KMeans.generate_palette(pixels, color_format, quality, max_colors),
    }
}

trait PaletteGenerator {
    fn generate_palette(
        &self,
        pixels: &[u8],
        color_format: ColorFormat,
        quality: u8,
        max_colors: u8,
    ) -> Result<Vec<Color>, Error>;
}

struct Mmcq;

impl PaletteGenerator for Mmcq {
    fn generate_palette(
        &self,
        pixels: &[u8],
        color_format: ColorFormat,
        quality: u8,
        max_colors: u8,
    ) -> Result<Vec<Color>, Error> {
        let (vbox, histogram) = make_histogram_and_vbox(pixels, color_format, quality);
        quantize(&vbox, &histogram, max_colors)
    }
}

struct KMeans;

impl PaletteGenerator for KMeans {
    fn generate_palette(
        &self,
        pixels: &[u8],
        color_format: ColorFormat,
        quality: u8,
        max_colors: u8,
    ) -> Result<Vec<Color>, Error> {
        let colors_count = color_format.channels();
        let step = quality as usize;

        let mut samples = Vec::new();
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
        if k == 0 {
            return Err(Error::InvalidVBox);
        }

        let max_iter = 100;
        let centroids = kmeans(&samples, k, max_iter);

        let mut seen = HashSet::new();
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

    let mut unique = HashSet::new();
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
    let mut changed = false;
    for (i, sample) in samples.iter().enumerate() {
        let mut min_dist = f64::INFINITY;
        let mut best_cluster = 0;
        for (cluster_idx, centroid) in centroids.iter().enumerate() {
            let dist = color_distance(sample, centroid);
            if dist < min_dist {
                min_dist = dist;
                best_cluster = cluster_idx;
            }
        }
        if assignments[i] != best_cluster {
            assignments[i] = best_cluster;
            changed = true;
        }
    }
    changed
}

fn color_distance(c1: &Color, c2: &Color) -> f64 {
    let dr = c1.r as f64 - c2.r as f64;
    let dg = c1.g as f64 - c2.g as f64;
    let db = c1.b as f64 - c2.b as f64;
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
        sums[cluster].0 += sample.r as f64;
        sums[cluster].1 += sample.g as f64;
        sums[cluster].2 += sample.b as f64;
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
            let r = (sums[i].0 / counts[i] as f64).round() as u8;
            let g = (sums[i].1 / counts[i] as f64).round() as u8;
            let b = (sums[i].2 / counts[i] as f64).round() as u8;
            new_centroids.push(Color::new(r, g, b));
        }
    }

    new_centroids
}

#[derive(Clone)]
enum ColorChannel {
    Red,
    Green,
    Blue,
}

#[derive(Clone)]
struct VBox {
    r_min: u8,
    r_max: u8,
    g_min: u8,
    g_max: u8,
    b_min: u8,
    b_max: u8,
    average: Color,
    volume: i32,
    count: i32,
}

impl VBox {
    fn new(r_min: u8, r_max: u8, g_min: u8, g_max: u8, b_min: u8, b_max: u8) -> VBox {
        VBox {
            r_min,
            r_max,
            g_min,
            g_max,
            b_min,
            b_max,
            average: Color::new(0, 0, 0),
            volume: 0,
            count: 0,
        }
    }

    fn recalc(&mut self, histogram: &[i32]) {
        self.average = self.calc_average(histogram);
        self.count = self.calc_count(histogram);
        self.volume = self.calc_volume();
    }

    fn calc_volume(&self) -> i32 {
        (self.r_max as i32 - self.r_min as i32 + 1)
            * (self.g_max as i32 - self.g_min as i32 + 1)
            * (self.b_max as i32 - self.b_min as i32 + 1)
    }

    fn calc_count(&self, histogram: &[i32]) -> i32 {
        let mut count = 0;
        for i in self.r_min..=self.r_max {
            for j in self.g_min..=self.g_max {
                for k in self.b_min..=self.b_max {
                    let index = make_color_index_of(i, j, k);
                    count += histogram[index];
                }
            }
        }
        count
    }

    fn calc_average(&self, histogram: &[i32]) -> Color {
        let mut ntot = 0;
        let mut r_sum = 0;
        let mut g_sum = 0;
        let mut b_sum = 0;

        for i in self.r_min..=self.r_max {
            for j in self.g_min..=self.g_max {
                for k in self.b_min..=self.b_max {
                    let index = make_color_index_of(i, j, k);
                    let hval = histogram[index] as f64;
                    ntot += hval as i32;
                    r_sum += (hval * (i as f64 + 0.5) * MULTIPLIER_64) as i32;
                    g_sum += (hval * (j as f64 + 0.5) * MULTIPLIER_64) as i32;
                    b_sum += (hval * (k as f64 + 0.5) * MULTIPLIER_64) as i32;
                }
            }
        }

        if ntot > 0 {
            let r = (r_sum / ntot) as u8;
            let g = (g_sum / ntot) as u8;
            let b = (b_sum / ntot) as u8;
            Color::new(r, g, b)
        } else {
            let r = MULTIPLIER * (self.r_min as i32 + self.r_max as i32 + 1) / 2;
            let g = MULTIPLIER * (self.g_min as i32 + self.g_max as i32 + 1) / 2;
            let b = MULTIPLIER * (self.b_min as i32 + self.b_max as i32 + 1) / 2;
            Color::new(
                cmp::min(r, 255) as u8,
                cmp::min(g, 255) as u8,
                cmp::min(b, 255) as u8,
            )
        }
    }

    fn widest_color_channel(&self) -> ColorChannel {
        let r_width = self.r_max - self.r_min;
        let g_width = self.g_max - self.g_min;
        let b_width = self.b_max - self.b_min;
        let max = cmp::max(cmp::max(r_width, g_width), b_width);

        if max == r_width {
            ColorChannel::Red
        } else if max == g_width {
            ColorChannel::Green
        } else {
            ColorChannel::Blue
        }
    }
}

fn make_histogram_and_vbox(pixels: &[u8], color_format: ColorFormat, step: u8) -> (VBox, Vec<i32>) {
    let mut histogram = vec![0; HISTOGRAM_SIZE];
    let mut r_min = u8::MAX;
    let mut r_max = u8::MIN;
    let mut g_min = u8::MAX;
    let mut g_max = u8::MIN;
    let mut b_min = u8::MAX;
    let mut b_max = u8::MIN;

    let colors_count = color_format.channels();
    let step = step as usize;

    for i in (0..pixels.len()).step_by(colors_count * step) {
        if i + colors_count > pixels.len() {
            break;
        }
        let (r, g, b, a) = color_parts(pixels, color_format, i);
        if a < 125 || (r > 250 && g > 250 && b > 250) {
            continue;
        }

        let shifted_r = r >> RIGHT_SHIFT;
        let shifted_g = g >> RIGHT_SHIFT;
        let shifted_b = b >> RIGHT_SHIFT;

        r_min = cmp::min(r_min, shifted_r);
        r_max = cmp::max(r_max, shifted_r);
        g_min = cmp::min(g_min, shifted_g);
        g_max = cmp::max(g_max, shifted_g);
        b_min = cmp::min(b_min, shifted_b);
        b_max = cmp::max(b_max, shifted_b);

        let index = make_color_index_of(shifted_r, shifted_g, shifted_b);
        histogram[index] += 1;
    }

    let mut vbox = VBox::new(r_min, r_max, g_min, g_max, b_min, b_max);
    vbox.recalc(&histogram);
    (vbox, histogram)
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

fn apply_median_cut(histogram: &[i32], vbox: &mut VBox) -> Result<(VBox, Option<VBox>), Error> {
    if vbox.count == 0 {
        return Err(Error::InvalidVBox);
    }
    if vbox.count == 1 {
        return Ok((vbox.clone(), None));
    }

    let axis = vbox.widest_color_channel();
    let (total, partial_sum, look_ahead_sum) = compute_partial_sums(histogram, vbox, axis.clone());

    if total == 0 {
        return Err(Error::VBoxCutFailed);
    }

    cut(
        axis.clone(),
        vbox,
        histogram,
        &partial_sum,
        &look_ahead_sum,
        total,
    )
}

fn compute_partial_sums(
    histogram: &[i32],
    vbox: &VBox,
    axis: ColorChannel,
) -> (i32, Vec<i32>, Vec<i32>) {
    let mut total = 0;
    let mut partial_sum = vec![-1; VBOX_LENGTH];
    let mut look_ahead_sum = vec![-1; VBOX_LENGTH];

    match axis {
        ColorChannel::Red => {
            for i in vbox.r_min..=vbox.r_max {
                let mut sum = 0;
                for j in vbox.g_min..=vbox.g_max {
                    for k in vbox.b_min..=vbox.b_max {
                        let index = make_color_index_of(i, j, k);
                        sum += histogram[index];
                    }
                }
                total += sum;
                partial_sum[i as usize] = total;
            }
        }
        ColorChannel::Green => {
            for i in vbox.g_min..=vbox.g_max {
                let mut sum = 0;
                for j in vbox.r_min..=vbox.r_max {
                    for k in vbox.b_min..=vbox.b_max {
                        let index = make_color_index_of(j, i, k);
                        sum += histogram[index];
                    }
                }
                total += sum;
                partial_sum[i as usize] = total;
            }
        }
        ColorChannel::Blue => {
            for i in vbox.b_min..=vbox.b_max {
                let mut sum = 0;
                for j in vbox.r_min..=vbox.r_max {
                    for k in vbox.g_min..=vbox.g_max {
                        let index = make_color_index_of(j, k, i);
                        sum += histogram[index];
                    }
                }
                total += sum;
                partial_sum[i as usize] = total;
            }
        }
    }

    for (i, sum) in partial_sum.iter().enumerate().filter(|&(_, &s)| s != -1) {
        look_ahead_sum[i] = total - sum;
    }

    (total, partial_sum, look_ahead_sum)
}

fn cut(
    axis: ColorChannel,
    vbox: &VBox,
    histogram: &[i32],
    partial_sum: &[i32],
    look_ahead_sum: &[i32],
    total: i32,
) -> Result<(VBox, Option<VBox>), Error> {
    let (vbox_min, vbox_max) = match axis {
        ColorChannel::Red => (vbox.r_min as i32, vbox.r_max as i32),
        ColorChannel::Green => (vbox.g_min as i32, vbox.g_max as i32),
        ColorChannel::Blue => (vbox.b_min as i32, vbox.b_max as i32),
    };

    for i in vbox_min..=vbox_max {
        if partial_sum[i as usize] > total / 2 {
            let mut vbox1 = vbox.clone();
            let mut vbox2 = vbox.clone();

            let left = i - vbox_min;
            let right = vbox_max - i;

            let mut d2 = if left <= right {
                cmp::min(vbox_max - 1, i + right / 2)
            } else {
                cmp::max(vbox_min, (i as f64 - left as f64 / 2.0) as i32)
            };

            while d2 < vbox_min || (d2 < vbox_max && partial_sum[d2 as usize] <= 0) {
                d2 += 1;
            }

            let mut count2 = look_ahead_sum[d2 as usize];
            while count2 == 0 && d2 > vbox_min && partial_sum[d2 as usize - 1] > 0 {
                d2 -= 1;
                count2 = look_ahead_sum[d2 as usize];
            }

            match axis {
                ColorChannel::Red => {
                    vbox1.r_max = d2 as u8;
                    vbox2.r_min = (d2 + 1) as u8;
                }
                ColorChannel::Green => {
                    vbox1.g_max = d2 as u8;
                    vbox2.g_min = (d2 + 1) as u8;
                }
                ColorChannel::Blue => {
                    vbox1.b_max = d2 as u8;
                    vbox2.b_min = (d2 + 1) as u8;
                }
            }

            vbox1.recalc(histogram);
            vbox2.recalc(histogram);
            return Ok((vbox1, Some(vbox2)));
        }
    }

    Err(Error::VBoxCutFailed)
}

fn quantize(vbox: &VBox, histogram: &[i32], max_colors: u8) -> Result<Vec<Color>, Error> {
    let mut pq = vec![vbox.clone()];
    let target = (FRACTION_BY_POPULATION * max_colors as f64).ceil() as u8;

    iterate(&mut pq, compare_by_count, target, histogram)?;
    pq.sort_by(compare_by_product);

    let len = pq.len() as u8;
    iterate(&mut pq, compare_by_product, max_colors - len, histogram)?;

    pq.reverse();
    let mut colors: Vec<Color> = pq.into_iter().map(|v| v.average).collect();
    colors.truncate(max_colors as usize);
    Ok(colors)
}

fn iterate<P>(
    queue: &mut Vec<VBox>,
    comparator: P,
    target: u8,
    histogram: &[i32],
) -> Result<(), Error>
where
    P: FnMut(&VBox, &VBox) -> cmp::Ordering + Copy,
{
    let mut color = 1;
    for _ in 0..MAX_ITERATIONS {
        if let Some(mut vbox) = queue.pop() {
            if vbox.count == 0 {
                queue.sort_by(comparator);
                continue;
            }

            let vboxes = apply_median_cut(histogram, &mut vbox)?;
            queue.push(vboxes.0);
            if let Some(vbox2) = vboxes.1 {
                queue.push(vbox2);
                color += 1;
            }

            queue.sort_by(comparator);

            if color >= target {
                break;
            }
        }
    }
    Ok(())
}

fn compare_by_count(a: &VBox, b: &VBox) -> cmp::Ordering {
    a.count.cmp(&b.count)
}

fn compare_by_product(a: &VBox, b: &VBox) -> cmp::Ordering {
    a.count.cmp(&b.count).then_with(|| {
        let a_product = a.count as i64 * a.volume as i64;
        let b_product = b.count as i64 * b.volume as i64;
        a_product.cmp(&b_product)
    })
}

/// Get reduced-space color index for a pixel.
fn make_color_index_of(red: u8, green: u8, blue: u8) -> usize {
    (((red as i32) << (2 * SIGNAL_BITS)) + ((green as i32) << SIGNAL_BITS) + blue as i32) as usize
}
