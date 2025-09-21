use std::collections::HashMap;
use std::f64;
use std::sync::Arc;

use interface_detector::textlines::Quadrilateral;
use itertools::Itertools;
use petgraph::graph::NodeIndex;
use petgraph::unionfind::UnionFind;
use petgraph::visit::EdgeRef as _;
use petgraph::{Graph, Undirected};

pub fn connected_components_sets<N, E>(graph: &Graph<N, E, Undirected>) -> Vec<Vec<NodeIndex>> {
    let node_indices: Vec<_> = graph.node_indices().collect();
    let mut uf = UnionFind::new(node_indices.len());

    let node_to_idx: HashMap<_, _> = node_indices
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i))
        .collect();

    for edge in graph.edge_references() {
        let a = node_to_idx[&edge.source()];
        let b = node_to_idx[&edge.target()];
        uf.union(a, b);
    }

    let labels: Vec<_> = node_indices
        .iter()
        .enumerate()
        .map(|(i, _)| uf.find(i))
        .collect();

    let mut groups: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
    for &node_index in &node_indices {
        let idx = node_to_idx[&node_index];
        let label = labels[idx];
        groups.entry(label).or_default().push(node_index);
    }

    let mut comps: Vec<_> = groups.into_values().collect();
    comps.sort_by_key(|c| c[0]);
    comps
}

pub fn generate_text_direction(
    bboxes: Vec<Arc<parking_lot::Mutex<Quadrilateral>>>,
) -> impl Iterator<Item = (Arc<parking_lot::Mutex<Quadrilateral>>, bool)> {
    let mut graph: Graph<Arc<parking_lot::Mutex<Quadrilateral>>, (), petgraph::Undirected> =
        Graph::new_undirected();
    // allow:clone[arc]
    for bbox in bboxes.clone() {
        graph.add_node(bbox);
    }
    for ((u, ubox), (v, vbox)) in bboxes.iter().enumerate().tuple_combinations() {
        if quadrilateral_can_merge_region(
            &*ubox.lock(),
            &*vbox.lock(),
            1.9,
            2.0,
            0.6,
            1.5,
            1.5,
            1.0,
        ) {
            graph.add_edge((u as u32).into(), (v as u32).into(), ());
        }
    }
    let components = connected_components_sets(&graph);
    components.into_iter().flat_map(move |nodes| {
        let vertical_dirs: Vec<_> = nodes
            .iter()
            .map(|&node| graph[node].lock().vertical())
            .collect();
        let vertical_count = vertical_dirs.iter().filter(|&&v| v).count();

        let majority_vertical = vertical_count > vertical_dirs.len() / 2;

        if majority_vertical {
            nodes
                .into_iter()
                .sorted_by_key(|&node| {
                    let aabb = graph[node].lock().aabb();
                    -(aabb.x + aabb.w)
                })
                // allow:clone[arc]
                .map(|node| (graph[node].clone(), majority_vertical))
                .collect::<Vec<_>>()
                .into_iter()
        } else {
            nodes
                .into_iter()
                .sorted_by_key(|&node| {
                    let aabb = graph[node].lock().aabb();
                    aabb.y + aabb.h / 2
                })
                // allow:clone[arc]
                .map(|node| (graph[node].clone(), majority_vertical))
                .collect::<Vec<_>>()
                .into_iter()
        }
    })
}

pub fn quadrilateral_can_merge_region(
    a: &Quadrilateral,
    b: &Quadrilateral,
    ratio: f32,
    discard_connection_gap: f64,
    char_gap_tolerance: f64,
    char_gap_tolerance2: f64,
    font_size_ratio_tol: f64,
    aspect_ratio_tol: f64,
) -> bool {
    let b1 = a.aabb();
    let b2 = b.aabb();
    let char_size = a.font_size().min(b.font_size());
    let (x1, y1, w1, h1) = (b1.x, b1.y, b1.w, b1.h);
    let (x2, y2, w2, h2) = (b2.x, b2.y, b2.w, b2.h);
    let dist = a.poly_distance(&b);
    if dist > discard_connection_gap * char_size {
        return false;
    }

    if a.font_size().max(b.font_size()) / char_size > font_size_ratio_tol {
        return false;
    }
    if a.aspect_ratio() > aspect_ratio_tol && b.aspect_ratio() < 1. / aspect_ratio_tol {
        return false;
    }
    if b.aspect_ratio() > aspect_ratio_tol && a.aspect_ratio() < 1. / aspect_ratio_tol {
        return false;
    }
    let a_aa = a.is_approximate_axis_aligned();
    let b_aa = b.is_approximate_axis_aligned();
    if a_aa && b_aa {
        if dist < char_size * char_gap_tolerance {
            if (((x1 + w1 / 2) - (x2 + w2 / 2)).abs() as f64) < char_gap_tolerance2 {
                return true;
            }
            if w1 as f32 > h1 as f32 * ratio && h2 as f32 > w2 as f32 * ratio {
                return false;
            }
            if w2 as f32 > h2 as f32 * ratio && h1 as f32 > w1 as f32 * ratio {
                return false;
            }
            if w1 as f32 > h1 as f32 * ratio || w2 as f32 > h2 as f32 * ratio {
                return ((x1 - x2).abs() as f64) < char_size * char_gap_tolerance2
                    || ((x1 + w1 - (x2 + w2)).abs() as f64) < char_size * char_gap_tolerance2;
            } else if h1 as f32 > w1 as f32 * ratio || h2 as f32 > w2 as f32 * ratio {
                return ((y1 - y2).abs() as f64) < char_size * char_gap_tolerance2
                    || ((y1 + h1 - (y2 + h2)).abs() as f64) < char_size * char_gap_tolerance2;
            }
            return false;
        } else {
            return false;
        }
    }

    //if not a_aa and not b_aa:
    if (a.angle() - b.angle()).abs() < 15.0 * f64::consts::PI / 180.0 {
        let fs_a = a.font_size();
        let fs_b = b.font_size();
        let fs = fs_a.min(fs_b);
        if dist > fs * char_gap_tolerance2 {
            return false;
        }
        if (fs_a - fs_b).abs() / fs > 0.25 {
            return false;
        }
        return true;
    }

    false
}
