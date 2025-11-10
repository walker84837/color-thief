// Copyright 2017, Reizner Evgeniy <razrfalcon@gmail.com>.
// See the COPYRIGHT file at the top-level directory of this distribution.
// Licensed under the MIT license, see the LICENSE file or <http://opensource.org/licenses/MIT>

//! *color-thief-rs* is a [color-thief](https://github.com/lokesh/color-thief)
//! algorithm reimplementation in Rust.
//!
//! The implementation itself is a heavily modified
//! [Swift version](https://github.com/yamoridon/ColorThiefSwift) of the same algorithm.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod kmeans;
mod mmcq;

use std::str::FromStr;
use thiserror::Error;

use kmeans::KMeansError;
use mmcq::MmcqError;

pub use rgb::RGB8 as Color;

const SIGNAL_BITS: i32 = 5;
const RIGHT_SHIFT: i32 = 8 - SIGNAL_BITS;
const MULTIPLIER: i32 = 1 << RIGHT_SHIFT;
const MULTIPLIER_64: f64 = MULTIPLIER as f64;
const HISTOGRAM_SIZE: usize = 1 << (3 * SIGNAL_BITS);
const VBOX_LENGTH: usize = 1 << SIGNAL_BITS;
const FRACTION_BY_POPULATION: f64 = 0.75;
const MMCQ_ITERATION_LIMIT: i32 = 1000;

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
    KMeans {
        /// The maximum number of iterations for the K-Means algorithm to run.
        max_iterations: usize,
    },
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

/// Represents an error that can occur during color palette generation.
#[derive(Debug, Error)]
pub enum Error {
    /// An invalid VBox was encountered. This can happen if:
    /// - a VBox (a representation of a color range) contains no pixels after filtering;
    /// - if the input parameters lead to an impossible VBox state, such as requesting 0 colors for the palette.
    #[error("invalid vbox")]
    InvalidVBox,
    /// An error occurred in the MMCQ algorithm.
    #[error(transparent)]
    Mmcq(#[from] MmcqError),
    /// An error occurred in the K-Means algorithm.
    #[error(transparent)]
    KMeans(#[from] KMeansError),
}

/// Represents an error for invalid input when parsing a color format
#[derive(Debug, Error)]
#[error("invalid color format")]
pub struct ColorParseError;

impl FromStr for ColorFormat {
    type Err = ColorParseError;

    fn from_str(s: &str) -> Result<ColorFormat, Self::Err> {
        const VARIANTS: &[(&str, ColorFormat)] = &[
            ("rgb", ColorFormat::Rgb),
            ("rgba", ColorFormat::Rgba),
            ("argb", ColorFormat::Argb),
            ("bgr", ColorFormat::Bgr),
            ("bgra", ColorFormat::Bgra),
        ];

        // Look through the list of variants and programmatically check them for equality
        VARIANTS
            .iter()
            .find(|(name, _)| s.eq_ignore_ascii_case(name))
            .map(|(_, fmt)| *fmt)
            .ok_or(ColorParseError)
    }
}

/// Returns a representative color palette of an image.
///
/// * `algorithm` - The algorithm to use for palette generation.
/// * `pixels` - A raw image data.
/// * `color_format` - Represent a color format of the image data.
/// * `quality` - Quality of the output (step size for sampling pixels). Should be between 1 and 10.
/// * `max_colors` - Maximum number of colors in the output palette. Should be greater than 1.
pub fn get_palette(
    algorithm: Algorithm,
    pixels: &[u8],
    color_format: ColorFormat,
    quality: u8,
    max_colors: u8,
) -> Result<Vec<Color>, Error> {
    assert!(
        quality > 0 && quality <= 10,
        "quality should be between 1 and 10"
    );
    assert!(max_colors > 1, "max_colors should be greater than 1");

    match algorithm {
        Algorithm::Mmcq => mmcq::Mmcq
            .generate_palette(pixels, color_format, quality, max_colors)
            .map_err(Error::Mmcq),
        Algorithm::KMeans { max_iterations } => {
            let kmeans = kmeans::KMeans { max_iterations };
            kmeans
                .generate_palette(pixels, color_format, quality, max_colors)
                .map_err(Error::KMeans)
        }
    }
}

trait PaletteGenerator {
    type Error: std::error::Error;

    fn generate_palette(
        &self,
        pixels: &[u8],
        color_format: ColorFormat,
        quality: u8,
        max_colors: u8,
    ) -> Result<Vec<Color>, Self::Error>;
}
