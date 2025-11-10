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

About 150x faster that JS version.

```text
test q1  ... bench:   1,429,800 ns/iter (+/- 21,987)
test q10 ... bench:     854,297 ns/iter (+/- 25,468)
```

### Changes and migration guide (from previous versions to 0.2.2)

This fork introduces breaking API changes, including a new K-Means algorithm and a modified `get_palette` function signature.

#### Breaking changes

The `get_palette` function signature has changed to include an `algorithm` parameter.

**Old signature**:
```rust
pub fn get_palette(pixels: &[u8], color_format: ColorFormat, quality: u8, max_colors: u8) -> Result<Vec<Color>, Error>
```

**New signature**:
```rust
pub fn get_palette(algorithm: Algorithm, pixels: &[u8], color_format: ColorFormat, quality: u8, max_colors: u8) -> Result<Vec<Color>, Error>
```

To migrate, you must now specify the algorithm to use. For the original behavior, use `Algorithm::Mmcq`.

#### New Features

* **K-Means algorithm**: A new `Algorithm` enum has been introduced, allowing you to choose between `Mmcq` (the original Modified Median Cut Quantization algorithm) and `KMeans` (K-means clustering algorithm).
  * `Algorithm::Mmcq`: Fast, distinct colors, less accurate for subtle gradients.
  * `Algorithm::KMeans { max_iterations: usize }`: Slower, better for subtle gradients, more accurate overall. The `max_iterations` field allows you to specify the maximum number of iterations for the K-Means algorithm to run.
* **Expanded `ColorFormat`**: The `ColorFormat` enum now supports more formats: `Rgb`, `Rgba`, `Argb`, `Bgr`, `Bgra`.
* **Improved error handling**: The `Error` enum now provides more descriptive error messages.

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

// For the new K-Means algorithm
let colors_kmeans = color_thief::get_palette(
    Algorithm::KMeans {
        max_iterations: 100,
    },
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
