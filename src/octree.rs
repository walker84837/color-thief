use super::{Color, ColorFormat, PaletteGenerator, pixel};
use std::collections::BinaryHeap;
use std::num::NonZeroU8;
use thiserror::Error;

const MAX_DEPTH: u8 = 8;
const DEFAULT_DEPTH: u8 = 6;

#[derive(Clone, Debug, Error)]
#[error("octree error")]
pub struct OctreeError;

pub struct Octree {
    pub max_depth: Option<NonZeroU8>,
}

impl Octree {
    pub fn new(max_depth: Option<NonZeroU8>) -> Self {
        Octree { max_depth }
    }
}

struct Arena {
    nodes: Vec<Node>,
}

#[derive(Copy, Clone, Default)]
struct Node {
    r_sum: u64,
    g_sum: u64,
    b_sum: u64,
    pixel_count: u64,
    children: [Option<usize>; 8],
    is_leaf: bool,
    parent: Option<usize>,
}

impl Arena {
    fn new() -> Self {
        Arena {
            nodes: Vec::with_capacity(200_000),
        }
    }

    fn alloc(&mut self, parent: Option<usize>) -> usize {
        let node = Node {
            parent,
            ..Default::default()
        };
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    #[inline]
    const fn get_child_index(r: u8, g: u8, b: u8, depth: u8) -> u8 {
        let shift = 7 - depth;
        let r_bit = (r >> shift) & 1;
        let g_bit = (g >> shift) & 1;
        let b_bit = (b >> shift) & 1;
        (r_bit << 2) | (g_bit << 1) | b_bit
    }

    fn add_color(&mut self, root_idx: usize, r: u8, g: u8, b: u8, depth: u8, max_depth: u8) {
        let node = &mut self.nodes[root_idx];
        node.pixel_count += 1;

        if depth == max_depth {
            node.is_leaf = true;
            node.r_sum += r as u64;
            node.g_sum += g as u64;
            node.b_sum += b as u64;
            return;
        }

        let index = Self::get_child_index(r, g, b, depth);

        if self.nodes[root_idx].children[index as usize].is_none() {
            let child_idx = self.alloc(Some(root_idx));
            self.nodes[root_idx].children[index as usize] = Some(child_idx);
        }

        let child_idx = self.nodes[root_idx].children[index as usize].unwrap();
        self.add_color(child_idx, r, g, b, depth + 1, max_depth);
    }

    fn collect_leaves(&self, root_idx: usize, leaves: &mut Vec<(u64, Color)>) {
        let mut stack = vec![root_idx];

        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];

            if node.is_leaf {
                if node.pixel_count > 0 {
                    let r = (node.r_sum / node.pixel_count) as u8;
                    let g = (node.g_sum / node.pixel_count) as u8;
                    let b = (node.b_sum / node.pixel_count) as u8;
                    leaves.push((node.pixel_count, Color::new(r, g, b)));
                }
            } else {
                for &child_idx in &node.children {
                    if let Some(idx) = child_idx {
                        stack.push(idx);
                    }
                }
            }
        }
    }

    fn merge_node(&mut self, idx: usize) -> usize {
        let child_indices: Vec<usize> = (0..8)
            .filter_map(|i| self.nodes[idx].children[i].take())
            .collect();

        let child_data: Vec<(u64, u64, u64)> = child_indices
            .iter()
            .map(|&c| {
                let node = self.nodes[c];
                (node.r_sum, node.g_sum, node.b_sum)
            })
            .collect();

        let child_count = child_data.len();
        let (r_sum, g_sum, b_sum) = child_data
            .iter()
            .fold((0u64, 0u64, 0u64), |(r, g, b), (cr, cg, cb)| {
                (r + cr, g + cg, b + cb)
            });

        let node = &mut self.nodes[idx];
        node.r_sum = r_sum;
        node.g_sum = g_sum;
        node.b_sum = b_sum;
        node.is_leaf = true;

        if child_count > 0 { child_count - 1 } else { 0 }
    }

    fn has_children(&self, idx: usize) -> bool {
        self.nodes[idx].children.iter().any(|c| c.is_some())
    }

    fn pixel_count(&self, idx: usize) -> u64 {
        self.nodes[idx].pixel_count
    }
}

#[derive(Clone, Eq, PartialEq, PartialOrd, Ord)]
struct MergeCandidate {
    pixel_count: u64,
    node_idx: usize,
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
        let max_depth = self
            .max_depth
            .map_or(DEFAULT_DEPTH, |d| d.get())
            .clamp(1, MAX_DEPTH);

        let mut arena = Arena::new();
        let root_idx = arena.alloc(None);

        let channels = color_format.channels();
        let step = quality as usize;

        for i in (0..pixels.len()).step_by(channels * step) {
            if i + channels > pixels.len() {
                break;
            }

            let (r, g, b, a) = pixel::color_parts(pixels, color_format, i);
            if !pixel::should_skip_pixel(r, g, b, a) {
                arena.add_color(root_idx, r, g, b, 0, max_depth);
            }
        }

        let mut leaves = Vec::new();
        arena.collect_leaves(root_idx, &mut leaves);

        if leaves.len() <= max_colors as usize {
            leaves.sort_by(|a, b| b.0.cmp(&a.0));
            return Ok(leaves.into_iter().map(|(_, c)| c).collect());
        }

        let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();
        let mut stack = vec![root_idx];

        while let Some(idx) = stack.pop() {
            let node = &arena.nodes[idx];

            if node.is_leaf {
                continue;
            }

            let has_children = node.children.iter().any(|c| c.is_some());
            if has_children {
                heap.push(MergeCandidate {
                    pixel_count: node.pixel_count,
                    node_idx: idx,
                });
            }

            for &child_idx in &node.children {
                if let Some(c) = child_idx {
                    stack.push(c);
                }
            }
        }

        let mut leaf_count = leaves.len();

        while leaf_count > max_colors as usize {
            if let Some(candidate) = heap.pop() {
                let reduced = arena.merge_node(candidate.node_idx);
                leaf_count -= reduced;

                if let Some(parent_idx) = arena.nodes[candidate.node_idx].parent
                    && !arena.nodes[parent_idx].is_leaf
                    && arena.has_children(parent_idx)
                {
                    heap.push(MergeCandidate {
                        pixel_count: arena.pixel_count(parent_idx),
                        node_idx: parent_idx,
                    });
                }
            } else {
                break;
            }
        }

        leaves.clear();
        arena.collect_leaves(root_idx, &mut leaves);
        leaves.sort_by(|a, b| b.0.cmp(&a.0));

        Ok(leaves
            .into_iter()
            .take(max_colors as usize)
            .map(|(_, c)| c)
            .collect())
    }
}
