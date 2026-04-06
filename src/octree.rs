use super::{Color, ColorFormat, PaletteGenerator};
use std::num::NonZeroU8;
use thiserror::Error;

const MAX_DEPTH: u8 = 8;

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

#[derive(Default)]
struct Node {
    r_sum: u64,
    g_sum: u64,
    b_sum: u64,
    pixel_count: u64,
    children: [Option<Box<Node>>; 8],
    is_leaf: bool,
}

impl Node {
    fn new() -> Self {
        Node::default()
    }

    fn add_color(&mut self, r: u8, g: u8, b: u8, depth: u8, max_depth: u8) {
        self.pixel_count += 1;

        if depth == max_depth {
            self.is_leaf = true;
            self.r_sum += r as u64;
            self.g_sum += g as u64;
            self.b_sum += b as u64;
            return;
        }

        let index = self.get_child_index(r, g, b, depth);

        if self.children[index as usize].is_none() {
            self.children[index as usize] = Some(Box::new(Node::new()));
        }

        self.children[index as usize]
            .as_mut()
            .unwrap()
            .add_color(r, g, b, depth + 1, max_depth);
    }

    fn get_child_index(&self, r: u8, g: u8, b: u8, depth: u8) -> u8 {
        let shift = 7 - depth;
        let r_bit = (r >> shift) & 1;
        let g_bit = (g >> shift) & 1;
        let b_bit = (b >> shift) & 1;

        (r_bit << 2) | (g_bit << 1) | b_bit
    }

    fn get_leaf_nodes(&self, leaves: &mut Vec<(u64, Color)>) {
        if self.is_leaf {
            if self.pixel_count > 0 {
                let r = (self.r_sum / self.pixel_count) as u8;
                let g = (self.g_sum / self.pixel_count) as u8;
                let b = (self.b_sum / self.pixel_count) as u8;
                leaves.push((self.pixel_count, Color::new(r, g, b)));
            }
            return;
        }

        for child in self.children.iter().flatten() {
            child.get_leaf_nodes(leaves);
        }
    }

    fn merge_leaves(&mut self) -> usize {
        let mut count = 0;
        let mut r_sum = 0;
        let mut g_sum = 0;
        let mut b_sum = 0;

        for child_opt in &mut self.children {
            if let Some(child) = child_opt.take() {
                r_sum += child.r_sum;
                g_sum += child.g_sum;
                b_sum += child.b_sum;
                count += 1;
            }
        }

        self.r_sum = r_sum;
        self.g_sum = g_sum;
        self.b_sum = b_sum;
        self.is_leaf = true;

        if count > 0 { count - 1 } else { 0 }
    }

    fn find_node_at_path(&mut self, path: &[u8]) -> &mut Node {
        let mut current = self;

        for &index in path {
            current = current.children[index as usize].as_mut().unwrap();
        }

        current
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
        // Parameter initialization
        let max_depth = self.max_depth.map_or(MAX_DEPTH, |d| d.get()).min(MAX_DEPTH);
        let mut root = Node::new();

        let channels = color_format.channels();
        let step = quality as usize;

        // Pixel insertion loop
        for i in (0..pixels.len()).step_by(channels * step) {
            if i + channels > pixels.len() {
                break;
            }

            let (r, g, b, a) = color_format.color_parts(pixels, i);
            if a >= 125 && !(r > 250 && g > 250 && b > 250) {
                root.add_color(r, g, b, 0, max_depth);
            }
        }

        // Initial leaf counting
        let mut initial_leaves = Vec::new();
        root.get_leaf_nodes(&mut initial_leaves);
        let mut leaf_count = initial_leaves.len();

        if leaf_count <= max_colors as usize {
            initial_leaves.sort_by(|a, b| b.0.cmp(&a.0));
            return Ok(initial_leaves.into_iter().map(|(_, c)| c).collect());
        }

        // Tree reduction phase using safe paths
        for d in (0..max_depth).rev() {
            let mut parent_paths = Vec::new();
            let mut path = Vec::with_capacity(d as usize);

            get_paths_at_depth(&root, 0, d, &mut path, &mut parent_paths);

            // Sort by pixel count to merge nodes with fewer pixels first
            parent_paths.sort_by(|a, b| a.0.cmp(&b.0));

            for (_, path) in parent_paths {
                let node = root.find_node_at_path(&path);
                let reduced = node.merge_leaves();
                leaf_count -= reduced;

                if leaf_count <= max_colors as usize {
                    break;
                }
            }

            if leaf_count <= max_colors as usize {
                break;
            }
        }

        // Final extraction and sorting
        let mut final_leaves = Vec::new();
        root.get_leaf_nodes(&mut final_leaves);
        final_leaves.sort_by(|a, b| b.0.cmp(&a.0));

        Ok(final_leaves
            .into_iter()
            .take(max_colors as usize)
            .map(|(_, c)| c)
            .collect())
    }
}

fn get_paths_at_depth(
    node: &Node,
    current_depth: u8,
    target_depth: u8,
    path: &mut Vec<u8>,
    paths: &mut Vec<(u64, Vec<u8>)>,
) {
    if current_depth == target_depth {
        if node.children.iter().any(|c| c.is_some()) {
            paths.push((node.pixel_count, path.clone()));
        }
        return;
    }

    if node.is_leaf {
        return;
    }

    for (i, child_opt) in node.children.iter().enumerate() {
        if let Some(child) = child_opt {
            path.push(i as u8);
            get_paths_at_depth(child, current_depth + 1, target_depth, path, paths);
            path.pop();
        }
    }
}
