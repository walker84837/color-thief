use super::{Color, ColorFormat, PaletteGenerator};
use std::collections::HashMap;
use std::num::NonZeroU8;
use thiserror::Error;

const MAX_DEPTH: u8 = 8;

#[derive(Clone, Debug, Error)]
pub enum OctreeError {
    #[error("octree error")]
    OctreeError,
}

pub struct Octree {
    pub max_depth: Option<NonZeroU8>,
}

impl Octree {
    pub fn new(max_depth: Option<NonZeroU8>) -> Self {
        Octree { max_depth }
    }
}

impl PaletteGenerator for Octree {
    type Error = OctreeError;

    fn generate_palette(
        &self,
        pixels: &[u8],
        color_format: ColorFormat,
        quality: u8,
        max_colors: u8,
    ) -> Result<Vec<Color>, Self::Error> {
        let depth = self.max_depth.map_or(MAX_DEPTH, |d| d.get());
        debug_assert!((1..=8).contains(&depth));

        // Use a smaller effective depth for better color distribution
        // Depth 6 gives us 64^3 = 262K colors which is more manageable
        let effective_depth = depth.min(6);

        let mut color_map: HashMap<(u8, u8, u8), u32> = HashMap::new();

        let colors_count = color_format.channels();
        let step = quality as usize;

        for i in (0..pixels.len()).step_by(colors_count * step) {
            if i + colors_count > pixels.len() {
                break;
            }
            let (r, g, b, a) = color_format.color_parts(pixels, i);
            if a >= 125 && !(r > 250 && g > 250 && b > 250) {
                // Quantize to reduce distinct colors
                let qr = r >> (8 - effective_depth);
                let qg = g >> (8 - effective_depth);
                let qb = b >> (8 - effective_depth);
                *color_map.entry((qr, qg, qb)).or_insert(0) += 1;
            }
        }

        // Sort by count (most common first)
        let mut colors: Vec<_> = color_map.into_iter().collect();
        colors.sort_by(|a, b| b.1.cmp(&a.1));
        colors.truncate(max_colors as usize);

        let result: Vec<Color> = colors
            .into_iter()
            .map(|((r, g, b), _)| {
                // Scale back up
                let scale = 256u16 / (1 << effective_depth) as u16;
                Color::new(
                    (r as u16 * scale) as u8,
                    (g as u16 * scale) as u8,
                    (b as u16 * scale) as u8,
                )
            })
            .collect();

        Ok(result)
    }
}
