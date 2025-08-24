use std::{
    collections::{HashMap, HashSet},
    f64::consts::PI,
};

use geo::{ConvexHull, Distance as _, Euclidean, MinimumRotatedRect, MultiPoint, Point};
use interface_ocr::QuadrilateralInfo;
use interface_translator::{is_valuable_text, Detector, Language};
use itertools::Itertools as _;
use log::info;
use once_cell::sync::Lazy;
use ordered_float::OrderedFloat;
use petgraph::{
    algo::min_spanning_tree,
    data::Element,
    graph::{NodeIndex, UnGraph},
    Graph,
};
use util::text_direction::{connected_components_sets, quadrilateral_can_merge_region};

pub fn dispatch(
    textlines: &[QuadrilateralInfo],
    width: u16,
    height: u16,
    det: &dyn Detector,
) -> Vec<TextBlock> {
    merge_bboxes_text_region(textlines, width, height)
        .into_iter()
        .map(|(txtlns, fg_color, bg_color)| {
            let mut total_logprobs = 0.0;
            for txtln in &txtlns {
                total_logprobs += txtln.pos.score().ln() * txtln.pos.area();
            }

            total_logprobs /= textlines.iter().map(|v| v.pos.area()).sum::<f64>();
            let font_size = txtlns
                .iter()
                .map(|v| v.pos.font_size() as u64)
                .min()
                .unwrap_or_default();
            let mut angle = mean(txtlns.iter().map(|v| v.pos.angle())).unwrap()
                * (180.0 / std::f64::consts::PI)
                - 90.0;
            if angle.abs() < 3.0 {
                angle = 0.0;
            }
            let lines = txtlns
                .iter()
                .map(|v| v.pos.pts().clone())
                .collect::<Vec<_>>();
            let texts = txtlns
                .into_iter()
                .map(|v| v.text.clone())
                .collect::<Vec<_>>();
            let fg_color = match fg_color {
                (Some(r), Some(g), Some(b)) => Some((r as u8, g as u8, b as u8)),
                _ => None,
            };
            let bg_color = match bg_color {
                (Some(r), Some(g), Some(b)) => Some((r as u8, g as u8, b as u8)),
                _ => None,
            };
            TextBlock::new(
                lines,
                texts,
                font_size,
                angle,
                total_logprobs.exp(),
                fg_color,
                bg_color,
                det,
            )
        })
        .collect()
}

fn deg2rad(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.0
}

fn rotate_polygons(
    center: (i64, i64),
    polygons: Vec<[i64; 8]>,
    rotation: f32,
    new_center: Option<(i64, i64)>,
) -> Vec<[i64; 8]> {
    if rotation == 0.0 {
        return polygons;
    }

    let new_center = match new_center {
        Some(center) => center,
        None => center,
    };
    let rotation = deg2rad(rotation);
    let (s, c) = (rotation.sin(), rotation.cos());
    let (cx, cy) = (center.0 as f32, center.1 as f32);
    let (ncx, ncy) = (new_center.0 as f32, new_center.1 as f32);

    polygons
        .iter()
        .map(|poly| {
            let mut rotated = [0.0f32; 8];

            for i in 0..4 {
                let x = poly[i * 2] as f32 - cx;
                let y = poly[i * 2 + 1] as f32 - cy;

                let new_x = x * c - y * s + ncx;
                let new_y = x * s + y * c + ncy;

                rotated[i * 2] = new_x;
                rotated[i * 2 + 1] = new_y;
            }
            let rotated_i64: [i64; 8] = rotated.map(|x| x.round() as i64);
            rotated_i64
        })
        .collect()
}

pub struct OBB {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    /// radians
    pub theta: f64,
}

pub struct TextBlock {
    pub lines: Vec<[(i64, i64); 4]>,
    pub text: String,
    font_size: u64,
    pub angle: f64,
    prob: f64,
    fg_color: Option<(u8, u8, u8)>,
    bg_color: Option<(u8, u8, u8)>,
    pub skip_translate: bool,
    language: Option<Language>,
    pub translations: HashMap<String, String>,
}
impl TextBlock {
    pub fn obb(&self) -> Option<OBB> {
        let coords = MultiPoint::new(vec![Point::new(0.0, 0.0)])
            .convex_hull()
            .minimum_rotated_rect()?;

        let coords = &coords.exterior().0;
        let w = Euclidean.distance(coords[0], coords[1]);
        let h = Euclidean.distance(coords[1], coords[2]);
        let dx = coords[1].x - coords[0].x;
        let dy = coords[1].y - coords[0].y;
        let center_x = coords.iter().take(4).map(|c| c.x).sum::<f64>() / 4.0;
        let center_y = coords.iter().take(4).map(|c| c.y).sum::<f64>() / 4.0;
        let rotation = dy.atan2(dx);
        Some(OBB {
            x: center_x,
            y: center_y,
            w,
            h,
            theta: rotation,
        })
    }
    pub fn new(
        lines: Vec<[(i64, i64); 4]>,
        texts: Vec<String>,
        font_size: u64,
        angle: f64,
        prob: f64,
        fg_color: Option<(u8, u8, u8)>,
        bg_color: Option<(u8, u8, u8)>,
        det: &dyn Detector,
    ) -> Self {
        let mut iter = texts.iter();
        let mut result = match iter.next() {
            Some(first) => first.clone(),
            None => String::new(),
        };

        for txt in iter {
            let last_char_cjk = result
                .chars()
                .last()
                .is_some_and(|ch| matches!(ch, '\u{3000}'..='\u{9FFF}'));
            let first_char_cjk = txt
                .chars()
                .next()
                .is_some_and(|ch| matches!(ch, '\u{3000}'..='\u{9FFF}'));

            if last_char_cjk || first_char_cjk {
                result.push_str(txt);
            } else {
                result.push(' ');
                result.push_str(txt);
            }
        }
        Self {
            language: det.detect_language(&result),
            lines,
            text: result,
            font_size,
            angle,
            prob,
            fg_color,
            bg_color,
            translations: Default::default(),
            skip_translate: false,
        }
    }
}

fn compute_bounds(polygons: &[[i64; 8]]) -> Option<(i64, i64, i64, i64)> {
    if polygons.is_empty() {
        return None;
    }

    let mut min_x = i64::MAX;
    let mut min_y = i64::MAX;
    let mut max_x = i64::MIN;
    let mut max_y = i64::MIN;

    for poly in polygons {
        for (i, &val) in poly.iter().enumerate() {
            if i % 2 == 0 {
                min_x = min_x.min(val);
                max_x = max_x.max(val);
            } else {
                min_y = min_y.min(val);
                max_y = max_y.max(val);
            }
        }
    }

    Some((min_x, min_y, max_x, max_y))
}
impl TextBlock {
    fn center(&self) -> (i64, i64) {
        let xyxy = self.xyxy();
        ((xyxy.0 + xyxy.2) / 2, (xyxy.1 + xyxy.3) / 2)
    }

    fn unrotated_polygons(&self) -> Vec<[i64; 8]> {
        let reshaped: Vec<[i64; 8]> = self
            .lines
            .iter()
            .map(|quad| {
                [
                    quad[0].0, quad[0].1, quad[1].0, quad[1].1, quad[2].0, quad[2].1, quad[3].0,
                    quad[3].1,
                ]
            })
            .collect();
        if self.angle != 0.0 {
            rotate_polygons(self.center(), reshaped, self.angle as f32, None)
        } else {
            reshaped
        }
    }

    pub fn min_rect(&self) -> Vec<[(i64, i64); 4]> {
        let polygons = self.unrotated_polygons();
        let (min_x, min_y, max_x, max_y) = compute_bounds(&polygons).unwrap();
        let mut min_bbox = vec![[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]];
        if self.angle != 0.0 {
            min_bbox = rotate_polygons(self.center(), min_bbox, (-self.angle) as f32, None);
        }
        min_bbox
            .into_iter()
            .map(|bbox| {
                let clipped: Vec<i64> = bbox.iter().map(|&x| x.max(0)).collect();
                [
                    (clipped[0], clipped[1]),
                    (clipped[2], clipped[3]),
                    (clipped[4], clipped[5]),
                    (clipped[6], clipped[7]),
                ]
            })
            .collect()
    }
    pub fn xyxy(&self) -> (i64, i64, i64, i64) {
        let x = self
            .lines
            .iter()
            .flat_map(|v| v.into_iter().map(|v| v.0))
            .collect::<Vec<_>>();
        let y = self
            .lines
            .iter()
            .flat_map(|v| v.into_iter().map(|v| v.1))
            .collect::<Vec<_>>();

        (
            *x.iter().min().unwrap(),
            *y.iter().min().unwrap(),
            *x.iter().max().unwrap(),
            *y.iter().max().unwrap(),
        )
    }
}

fn merge_bboxes_text_region(
    bboxes: &[QuadrilateralInfo],
    width: u16,
    height: u16,
) -> Vec<(
    Vec<&QuadrilateralInfo>,
    (Option<f64>, Option<f64>, Option<f64>),
    (Option<f64>, Option<f64>, Option<f64>),
)> {
    let mut graph: Graph<QuadrilateralInfo, (), petgraph::Undirected> = Graph::new_undirected();

    // step 1: divide into multiple text region candidates
    for bbox in bboxes.iter() {
        graph.add_node(bbox.clone());
    }
    for ((u, ubox), (v, vbox)) in bboxes.iter().enumerate().tuple_combinations() {
        if quadrilateral_can_merge_region(&ubox.pos, &vbox.pos, 1.9, 2.0, 1.0, 3.0, 2.0, 1.3) {
            graph.add_edge((u as u32).into(), (v as u32).into(), ());
        }
    }

    // step 2: postprocess - further split each region
    let region_indices = connected_components_sets(&graph)
        .into_iter()
        .flat_map(|v| split_text_region(bboxes, v, width, height, 0.5, 2.0))
        .collect::<Vec<_>>();

    // step 3: return regions
    let v = region_indices
        .into_iter()
        .map(|node_set| {
            //TODO: should vertical or assigned_vertical be used?
            let mut nodes = node_set.into_iter().collect::<Vec<_>>();
            let txtlns = nodes.iter().map(|v| &bboxes[v.index()]).collect::<Vec<_>>();
            let fg_r = mean(txtlns.iter().filter_map(|v| v.fg.map(|v| v[0] as f64)));
            let fg_g = mean(txtlns.iter().filter_map(|v| v.fg.map(|v| v[1] as f64)));
            let fg_b = mean(txtlns.iter().filter_map(|v| v.fg.map(|v| v[2] as f64)));
            let bg_r = mean(txtlns.iter().filter_map(|v| v.bg.map(|v| v[0] as f64)));
            let bg_g = mean(txtlns.iter().filter_map(|v| v.bg.map(|v| v[1] as f64)));
            let bg_b = mean(txtlns.iter().filter_map(|v| v.bg.map(|v| v[2] as f64)));
            let vert = txtlns.iter().map(|v| v.pos.vertical() as u64).sum::<u64>();
            let count = txtlns.len() as u64;
            let vertical = if vert == count {
                true
            } else if vert * 2 == count {
                let mut max_aspect_ratio = -100.0;
                let mut lvert = true;
                for boxx in txtlns {
                    let baspect = boxx.pos.aspect_ratio();
                    if baspect > max_aspect_ratio {
                        max_aspect_ratio = baspect;
                        lvert = boxx.pos.vertical();
                    }
                    if 1.0 / baspect > max_aspect_ratio {
                        max_aspect_ratio = 1.0 / baspect;
                        lvert = boxx.pos.vertical();
                    }
                }
                lvert
            } else if vert * 2 > count {
                true
            } else {
                false
            };
            if vertical {
                nodes.sort_by_key(|a| OrderedFloat(-bboxes[a.index()].pos.centroid().0));
            } else {
                nodes.sort_by_key(|a| OrderedFloat(bboxes[a.index()].pos.centroid().1));
            }
            let txtlns = nodes.iter().map(|v| &bboxes[v.index()]).collect::<Vec<_>>();

            (txtlns, (fg_r, fg_g, fg_b), (bg_r, bg_g, bg_b))
        })
        .collect::<Vec<_>>();
    v
}

fn mean<I>(iter: I) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    let (sum, count) = iter.fold((0.0, 0), |(s, c), v| (s + v, c + 1));
    if count == 0 {
        return None;
    }
    Some(sum / count as f64)
}
fn stddev(values: &[f64]) -> Option<f64> {
    let len = values.len();
    if len == 0 {
        return None;
    }

    let mean = values.iter().sum::<f64>() / len as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / len as f64;

    Some(variance.sqrt())
}

fn split_text_region(
    bboxes: &[QuadrilateralInfo],
    connected_region_indices: Vec<NodeIndex>,
    width: u16,
    height: u16,
    gamma: f64,
    sigma: f64,
) -> Vec<HashSet<NodeIndex>> {
    if connected_region_indices.len() == 1 {
        return vec![connected_region_indices.into_iter().collect()];
    }

    if connected_region_indices.len() == 2 {
        let fb = &bboxes[connected_region_indices[0].index()];
        let sb = &bboxes[connected_region_indices[1].index()];
        let fs1 = fb.pos.font_size();
        let fs2 = sb.pos.font_size();
        let fs = fs1.max(fs2);

        if fb.pos.distance(&sb.pos, 0.5) < (1.0 + gamma) * fs
            && (fb.pos.angle() - sb.pos.angle()).abs() < 0.2 * PI
        {
            return vec![connected_region_indices.into_iter().collect()];
        } else {
            return vec![
                vec![connected_region_indices[0]].into_iter().collect(),
                vec![connected_region_indices[1]].into_iter().collect(),
            ];
        }
    }

    let mut graph: Graph<usize, f64, petgraph::Undirected> = UnGraph::new_undirected();
    let mut map = HashMap::new();
    for bbox in connected_region_indices.iter() {
        let idx = graph.add_node(bbox.index());
        map.insert(bbox.index(), idx);
    }
    for (u, v) in connected_region_indices.iter().tuple_combinations() {
        let weight = bboxes[u.index()].pos.distance(&bboxes[v.index()].pos, 0.5);
        graph.add_edge(
            *map.get(&u.index()).unwrap(),
            *map.get(&v.index()).unwrap(),
            weight,
        );
    }

    let edges: Vec<_> = min_spanning_tree(&graph).collect();

    let mut edges = edges
        .into_iter()
        .filter_map(|el| match el {
            Element::Edge {
                weight,
                source,
                target,
            } => Some((source, target, weight)),
            _ => None,
        })
        .collect::<Vec<_>>();
    edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let distances_sorted = edges.iter().map(|v| v.2).collect::<Vec<_>>();

    let fontsize = mean(
        connected_region_indices
            .iter()
            .map(|idx| (bboxes[idx.index()]).pos.font_size()),
    )
    .unwrap();

    let distances_mean = mean(distances_sorted.iter().cloned()).unwrap();
    let distances_std = stddev(&distances_sorted).unwrap();
    let std_threshold = f64::max(0.3 * fontsize + 5.0, 5.0);
    let (b1, b2) = (&bboxes[edges[0].0], &bboxes[edges[0].1]);
    let max_poly_distance = b1.pos.poly_distance(&b2.pos);
    let b1_centroid = b1.pos.centroid();
    let b2_centroid = b2.pos.centroid();
    let max_centroid_alignment = f64::max(
        (b1_centroid.0 - b2_centroid.0).abs(),
        (b1_centroid.1 - b2_centroid.1).abs(),
    );
    if (distances_sorted[0] <= distances_mean + distances_std * sigma
        || distances_sorted[0] <= fontsize * (1.0 + gamma))
        && (distances_std < std_threshold
            || max_poly_distance == 0.0 && max_centroid_alignment < 5.0)
    {
        vec![connected_region_indices.into_iter().collect()]
    } else {
        let mut graph: Graph<usize, (), petgraph::Undirected> = Graph::new_undirected();
        let mut map = HashMap::new();
        for idx in connected_region_indices {
            map.insert(idx.index(), graph.add_node(idx.index()));
        }
        // Split out the most deviating bbox
        for edge in &edges[1..] {
            graph.add_edge(*map.get(&edge.0).unwrap(), *map.get(&edge.1).unwrap(), ());
        }
        connected_components_sets(&graph)
            .into_iter()
            .flat_map(|node_set| split_text_region(bboxes, node_set, width, height, gamma, sigma))
            .collect()
    }
}

pub static BRACKET_PAIRS: Lazy<HashMap<char, char>> = Lazy::new(|| {
    HashMap::from([
        ('(', ')'),
        ('（', '）'),
        ('[', ']'),
        ('【', '】'),
        ('{', '}'),
        ('〔', '〕'),
        ('〈', '〉'),
        ('「', '」'),
        ('"', '"'),
        ('＂', '＂'),
        ('\'', '\''),
        ('“', '”'),
        ('《', '》'),
        ('『', '』'),
        ('〝', '〞'),
        ('﹁', '﹂'),
        ('﹃', '﹄'),
        ('⸂', '⸃'),
        ('⸄', '⸅'),
        ('⸉', '⸊'),
        ('⸌', '⸍'),
        ('⸜', '⸝'),
        ('⸠', '⸡'),
        ('‹', '›'),
        ('«', '»'),
        ('＜', '＞'),
        ('<', '>'),
    ])
});

pub static LEFT_SYMBOLS: Lazy<HashSet<char>> =
    Lazy::new(|| BRACKET_PAIRS.keys().cloned().collect());

pub static RIGHT_SYMBOLS: Lazy<HashSet<char>> =
    Lazy::new(|| BRACKET_PAIRS.values().cloned().collect());

fn has_brackets(stripped_text: &str) -> bool {
    stripped_text
        .chars()
        .any(|c| LEFT_SYMBOLS.contains(&c) || RIGHT_SYMBOLS.contains(&c))
}
pub fn dispatch_main(
    textlines: &[QuadrilateralInfo],
    width: u16,
    height: u16,
    min_text_length: usize,
    skip_languages: Vec<Language>,
    remove_text: &Vec<String>,
    det: &dyn Detector,
) -> Vec<TextBlock> {
    let text_regions = dispatch(textlines, width, height, det);
    text_regions
        .into_iter()
        .filter_map(|mut region| {
            let original_text = region.text;
            let stripped_text = original_text.trim();
            if original_text.len() != stripped_text.len() {
                info!("Removed leading characters from \"{original_text}\"");
            }
            let has_brackets = has_brackets(stripped_text);
            let stripped_text = if has_brackets {
                remove_leading_spaces_after_predict(stripped_text)
            } else {
                stripped_text.to_owned()
            };
            if stripped_text.len() < min_text_length {
                info!("Filtered out: {}", stripped_text);
                info!("Reason: Text length is less than the minimum required length.");
                None
            } else if !is_valuable_text(&stripped_text) {
                info!("Filtered out: {}", stripped_text);
                info!("Reason: Text is not considered valuable.");
                None
            } else if remove_text.contains(&stripped_text) {
                info!("Filtered out: {}", stripped_text);
                info!("Reason: Text contains unwanted content.");
                None
            } else {
                if let Some(language) = region.language {
                    if skip_languages.contains(&language) {
                        info!("Filtered out for translation: {}", stripped_text);
                        info!("Reason: Language should not be translated.");
                        region.skip_translate = true;
                    }
                }
                region.text = stripped_text;
                Some(region)
            }
        })
        .collect()
}

fn remove_leading_spaces_after_predict(stripped_text: &str) -> String {
    let mut stack = vec![];
    let mut result_chars = vec![];
    let mut to_skip = vec![];
    // First traversal: mark matching brackets
    for (i, char) in stripped_text.chars().enumerate() {
        if LEFT_SYMBOLS.contains(&char) {
            stack.push((i, char));
        } else if RIGHT_SYMBOLS.contains(&char) {
            if stack.is_empty() {
                // No corresponding left parenthesis, marked for deletion
                to_skip.push(i);
            } else {
                // There is a corresponding left bracket, pop the stack
                stack.pop();
            }
        }
    }

    // Mark unmatched left brackets as delete
    for (pos, _) in stack {
        to_skip.push(pos);
    }
    let mut stack = vec![];

    let has_removed_symbols = to_skip.len() > 0;
    for (i, char) in stripped_text.chars().enumerate() {
        if to_skip.contains(&i) {
            // Skip isolated parentheses
            continue;
        }
        if LEFT_SYMBOLS.contains(&char) {
            stack.push(char);
            result_chars.push(char);
        } else if RIGHT_SYMBOLS.contains(&char) {
            if let Some(left_bracket) = stack.pop() {
                let expected_right = *BRACKET_PAIRS.get(&left_bracket).unwrap();
                if char != expected_right {
                    //Replace mismatched right brackets with the correct right brackets corresponding to the left brackets
                    result_chars.push(expected_right);
                    info!(
                        "Fixed mismatched bracket: replaced \"{}\" with \"{}\"",
                        char, expected_right
                    );
                } else {
                    result_chars.push(char);
                }
            }
        } else {
            result_chars.push(char);
        }
    }
    let new_stripped_text = String::from_iter(result_chars);
    if has_removed_symbols {
        info!("Removed unpaired bracket from \"{}\"", stripped_text);
    }
    if new_stripped_text != stripped_text && !has_removed_symbols {
        info!(
            "Fixed brackets: \"{}\" → \"{}\"",
            stripped_text, new_stripped_text
        );
    }
    new_stripped_text
}
