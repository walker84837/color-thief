# color-thief-rs

A Rust implementation of the color-thief algorithm, forked from the archived [RazrFalcon/color-thief-rs](https://github.com/RazrFalcon/color-thief-rs) repository. This fork aims to carry the project forward by adding new algorithms, fixing bugs, and maintaining the library.

## Overview

*color-thief-rs* is a [color-thief](https://github.com/lokesh/color-thief) algorithm reimplementation in Rust.

## Table of Contents

- [Overview](#overview)
- [Differences](#differences)
- [Performance](#performance)
- [Changes and Migration Guide (from previous versions to 0.2.2)](#changes-and-migration-guide-from-previous-versions-to-022)
  - [Breaking changes](#breaking-changes)
  - [New Features](#new-features)
  - [Migration example](#migration-example)
- [Roadmap](#roadmap)
- [Usage](#usage)
- [License](#license)

### Differences

- There is no `getColor` method, since it's [just a shorthand](https://github.com/lokesh/color-thief/blob/b0115131476149500828b01db43ca701b099a315/src/color-thief.js#L76) for `getPalette`.
- Output colors are a bit different from JS version. See the [test suite](tests/test.rs) for details.

### Performance

About 150x faster than the JS version for MMCQ.

```text
test q1  ... bench:   1,429,800 ns/iter (+/- 21,987)
test q10 ... bench:     854,297 ns/iter (+/- 25,468)
```

| Algorithm | Speed | Accuracy | Best For |
|-----------|-------|----------|----------|
| **MMCQ**  | Fast  | Medium   | General purpose, extracting distinct colors. |
| **K-Means**| Slow  | High     | High-quality palettes, subtle gradients. |
| **Octree** | Fast  | High     | Memory-efficient, high-quality palettes. |

### Algorithm Selection

`color-thief-rs` provides several algorithms for color quantization:

- **MMCQ (Modified Median Cut Quantization)**: The original algorithm used in color-thief. It's fast and effective for most images.
- **K-Means**: An iterative algorithm that groups similar colors. It's more accurate for images with subtle gradients but slower. We use K-Means++ for initialization and mini-batching for performance.
- **Octree**: Builds a tree of color frequencies. It's very fast and often provides better quality than MMCQ.

#### Parallelization
K-Means uses `rayon` for parallel processing when the sample size exceeds 1000, improving performance on multi-core systems.

### Changes and migration guide (from previous versions to 0.2.2)

This fork introduces breaking API changes, including a new K-Means algorithm and a modified `get_palette` function signature.

#### Breaking changes

The `get_palette` function signature has changed to include an `algorithm` parameter.

**Old signature**:
```rust
pub fn get_palette(
    pixels: &[u8],
    color_format: ColorFormat,
    quality: u8,
    max_colors: u8
) -> Result<Vec<Color>, Error>
```

**New signature**:
```rust
pub fn get_palette(
    algorithm: Algorithm, <-- NEW parameter
    pixels: &[u8],
    color_format: ColorFormat,
    quality: u8,
    max_colors: u8
) -> Result<Vec<Color>, Error>
```

To migrate, you must now specify the algorithm to use. For the original behavior, use `Algorithm::Mmcq`.

#### New Features

* **K-Means algorithm**: A new `Algorithm` enum has been introduced, allowing you to choose between `Mmcq` (the original algorithm) and `KMeans`.
  * `Algorithm::KMeans { max_iterations: usize, seed: Option<u64> }`: Slower, better for subtle gradients. The `seed` field allows for reproducible results.
* **Octree algorithm**: A new `Algorithm::Octree { max_depth: Option<NonZeroU8> }` provides an efficient and high-quality alternative.
* **Expanded `ColorFormat`**: Support for `Rgb`, `Rgba`, `Argb`, `Bgr`, `Bgra`.
* **Improved error handling**: More descriptive error messages via `thiserror`.

#### Migration example

If you were previously calling `get_palette` like this:

```rust
let colors = color_thief::get_palette(&buffer, color_type, 10, 10).unwrap();
```

You should update it to explicitly use an algorithm:

```rust
use color_thief::Algorithm;

// For the original behavior (MMCQ)
let colors_mmcq = color_thief::get_palette(Algorithm::Mmcq, &buffer, color_type, 10, 10).unwrap();

// For the new K-Means algorithm with a seed
let colors_kmeans = color_thief::get_palette(
    Algorithm::KMeans {
        max_iterations: 100,
        seed: Some(42),
    },
    &buffer,
    color_type,
    10,
    10,
)
.unwrap();

// For the new Octree algorithm
let colors_octree = color_thief::get_palette(
    Algorithm::Octree { max_depth: None },
    &buffer,
    color_type,
    10,
    10,
)
.unwrap();
```

### Roadmap

- [x] Implement multiple algorithms to extract colors
- [ ] Create palette using color spaces
- [ ] Make library use generic types for convenience

### License

*color-thief-rs* is licensed under the MIT.
