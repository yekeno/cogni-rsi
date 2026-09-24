//! Cognition tree operations: insert, prune, frontier extraction, merge,
//! and boundary mastery analysis.
//!
//! Mirrors Dream-RSI (2609.14858v1) and `docs/PROPOSAL.md` §3.3 & §7:
//! - Leaf/frontier extraction
//! - Tree merge with `TreeDelta { new_nodes, closed_nodes, reopened_nodes }`
//! - `goal_progress`: weighted mean of understanding across the frontier
//! - `boundary_mastery`: `|frontier ∩ boundary| / |frontier|`

pub mod boundary;
pub mod delta;

pub use boundary::{BoundaryDomain, BoundarySpec};
pub use delta::{is_closed, TreeDelta};

use cogni_core::{CognitionNode, CognitionTree, NodeId};

/// Frontier = leaves of the current tree.
pub fn frontier(tree: &CognitionTree) -> Vec<NodeId> {
    let mut has_child = std::collections::HashSet::new();
    for n in tree.nodes.values() {
        if let Some(p) = n.parent {
            has_child.insert(p);
        }
    }
    tree.nodes
        .keys()
        .filter(|id| !has_child.contains(id))
        .copied()
        .collect()
}

/// Open frontier: leaves of the current tree that are NOT closed.
pub fn open_frontier(tree: &CognitionTree) -> Vec<NodeId> {
    frontier(tree)
        .into_iter()
        .filter(|id| {
            tree.get(*id).map(|n| !is_closed(n)).unwrap_or(false)
        })
        .collect()
}

/// Insert a node into the cognition tree.
pub fn insert(tree: &mut CognitionTree, node: CognitionNode) {
    tree.insert(node);
}

/// Merge a `source` tree into `target`, returning the structural `TreeDelta`.
///
/// Tracks:
/// - `new_nodes`: nodes present in `source` but not in `target` before merge.
/// - `closed_nodes`: nodes newly transitioning into closed/failed status.
/// - `reopened_nodes`: nodes transitioning from closed back to valid and Ok.
pub fn merge(target: &mut CognitionTree, source: &CognitionTree) -> TreeDelta {
    let mut new_nodes = 0;
    let mut closed_nodes = 0;
    let mut reopened_nodes = 0;

    for (id, node) in &source.nodes {
        match target.nodes.get(id) {
            None => {
                new_nodes += 1;
                if is_closed(node) {
                    closed_nodes += 1;
                }
                target.nodes.insert(*id, node.clone());
            }
            Some(old) => {
                let was_closed = is_closed(old);
                let now_closed = is_closed(node);
                if was_closed && !now_closed {
                    reopened_nodes += 1;
                } else if !was_closed && now_closed {
                    closed_nodes += 1;
                }
                target.nodes.insert(*id, node.clone());
            }
        }
    }

    TreeDelta {
        new_nodes,
        closed_nodes,
        reopened_nodes,
    }
}

/// Goal progress: weighted mean of `Score::understanding` over the frontier leaves.
///
/// Weight per node is based on self-reported confidence (`confidence.max(0.1)`).
/// Nodes that are closed contribute 0.0 understanding.
/// Returns 0.0 if frontier is empty. Output is clamped to `[0.0, 1.0]`.
pub fn goal_progress(tree: &CognitionTree) -> f32 {
    let leaves = frontier(tree);
    if leaves.is_empty() {
        return 0.0;
    }

    let mut total_weight = 0.0;
    let mut weighted_sum = 0.0;

    for id in leaves {
        if let Some(node) = tree.get(id) {
            let weight = node.score.confidence.max(0.1);
            let attainment = if is_closed(node) {
                0.0
            } else {
                node.score.understanding.clamp(0.0, 1.0)
            };
            weighted_sum += weight * attainment;
            total_weight += weight;
        }
    }

    if total_weight <= 0.0 {
        0.0
    } else {
        (weighted_sum / total_weight).clamp(0.0, 1.0)
    }
}

/// Boundary mastery: ratio of frontier leaves that fall within declared boundary regions.
///
/// Corresponds to `|frontier ∩ boundary| / |frontier|` (0.0..=1.0).
/// Returns 0.0 if frontier is empty.
pub fn boundary_mastery(tree: &CognitionTree, boundary: &BoundarySpec) -> f32 {
    let leaves = frontier(tree);
    if leaves.is_empty() {
        return 0.0;
    }

    let matched = leaves
        .iter()
        .filter_map(|id| tree.get(*id))
        .filter(|n| boundary.contains_node(n))
        .count();

    (matched as f32 / leaves.len() as f32).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use cogni_core::{FailureClass, NodeKind, Score, Tag};
    use serde_json::json;
    use ulid::Ulid;

    fn n(parent: Option<NodeId>) -> CognitionNode {
        CognitionNode {
            id: Ulid::new(),
            parent,
            iteration: 0,
            kind: NodeKind::Concept,
            title: "x".into(),
            workspace: json!({}),
            observation: "".into(),
            score: Score::zero(),
            tags: vec![],
            evidence: vec![],
            failure_class: FailureClass::Ok,
            valid: true,
            no_total: false,
            created_at: Utc::now(),
        }
    }

    fn n_with_score(
        parent: Option<NodeId>,
        understanding: f32,
        confidence: f32,
        valid: bool,
        failure_class: FailureClass,
    ) -> CognitionNode {
        CognitionNode {
            id: Ulid::new(),
            parent,
            iteration: 0,
            kind: NodeKind::Concept,
            title: "node".into(),
            workspace: json!({}),
            observation: "".into(),
            score: Score {
                understanding,
                boundary_fit: 0.0,
                reuse_potential: 0.0,
                confidence,
            },
            tags: vec![],
            evidence: vec![],
            failure_class,
            valid,
            no_total: false,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn frontier_is_leaves_only() {
        let root = n(None);
        let c1 = n(Some(root.id));
        let c2 = n(Some(root.id));
        let gc = n(Some(c1.id));
        let c2_id = c2.id;
        let gc_id = gc.id;
        let mut tree = CognitionTree::new(root);
        tree.insert(c1);
        tree.insert(c2);
        tree.insert(gc);
        let mut f = frontier(&tree);
        f.sort();
        let mut expected = vec![c2_id, gc_id];
        expected.sort();
        assert_eq!(f, expected);
    }

    #[test]
    fn tree_merge_calculates_accurate_delta() {
        let root = n(None);
        let mut tree1 = CognitionTree::new(root.clone());

        // Prepare source tree with 2 new open nodes and 1 new failed node
        let mut tree2 = CognitionTree::new(root.clone());
        let child1 = n(Some(root.id));
        let child2 = n(Some(root.id));
        let mut child3 = n(Some(root.id));
        child3.failure_class = FailureClass::Error;

        let child1_id = child1.id;
        tree2.insert(child1);
        tree2.insert(child2);
        tree2.insert(child3);

        let delta = merge(&mut tree1, &tree2);
        assert_eq!(delta.new_nodes, 3);
        assert_eq!(delta.closed_nodes, 1);
        assert_eq!(delta.reopened_nodes, 0);
        assert_eq!(delta.net(), 2);
        assert_eq!(tree1.nodes.len(), 4);

        // Now update child1 in a new tree to be closed/failed
        let mut tree3 = CognitionTree::new(root.clone());
        let mut child1_mod = tree1.get(child1_id).unwrap().clone();
        child1_mod.failure_class = FailureClass::Invalid;
        tree3.insert(child1_mod);

        let delta2 = merge(&mut tree1, &tree3);
        assert_eq!(delta2.new_nodes, 0);
        assert_eq!(delta2.closed_nodes, 1);
        assert_eq!(delta2.reopened_nodes, 0);

        // Now repair child1 back to Ok
        let mut tree4 = CognitionTree::new(root);
        let mut child1_reopen = tree1.get(child1_id).unwrap().clone();
        child1_reopen.failure_class = FailureClass::Ok;
        child1_reopen.valid = true;
        tree4.insert(child1_reopen);

        let delta3 = merge(&mut tree1, &tree4);
        assert_eq!(delta3.new_nodes, 0);
        assert_eq!(delta3.closed_nodes, 0);
        assert_eq!(delta3.reopened_nodes, 1);
    }

    #[test]
    fn goal_progress_computes_weighted_understanding() {
        let root = n(None);
        let mut tree = CognitionTree::new(root.clone());

        // Leaf 1: understanding 0.8, confidence 0.5 -> weight 0.5, contrib 0.4
        let l1 = n_with_score(Some(root.id), 0.8, 0.5, true, FailureClass::Ok);
        // Leaf 2: understanding 0.4, confidence 0.5 -> weight 0.5, contrib 0.2
        let l2 = n_with_score(Some(root.id), 0.4, 0.5, true, FailureClass::Ok);

        tree.insert(l1);
        tree.insert(l2);

        // Expected: (0.5 * 0.8 + 0.5 * 0.4) / (0.5 + 0.5) = 0.6
        let gp = goal_progress(&tree);
        assert!((gp - 0.6).abs() < 1e-4);

        // If one leaf fails, its understanding drops to 0.0
        let l3 = n_with_score(Some(root.id), 0.9, 1.0, false, FailureClass::Error);
        tree.insert(l3);
        // Total weights: 0.5 + 0.5 + 1.0 = 2.0
        // Weighted sum: 0.4 + 0.2 + 0.0 = 0.6
        // Expected: 0.6 / 2.0 = 0.3
        let gp2 = goal_progress(&tree);
        assert!((gp2 - 0.3).abs() < 1e-4);
    }

    #[test]
    fn boundary_mastery_computes_frontier_intersection_ratio() {
        let root = n(None);
        let mut tree = CognitionTree::new(root.clone());

        // 3 matching frontier leaves and 1 non-matching leaf
        let mut m1 = n(Some(root.id));
        m1.tags.push(Tag("gl".into()));

        let mut m2 = n(Some(root.id));
        m2.title = "应付结账处理".into();

        let mut m3 = n(Some(root.id));
        m3.score.boundary_fit = 0.9;

        let mut non_match = n(Some(root.id));
        non_match.title = "random background task".into();

        tree.insert(m1);
        tree.insert(m2);
        tree.insert(m3);
        tree.insert(non_match);

        let boundary = BoundarySpec::default_saas();
        let mastery = boundary_mastery(&tree, &boundary);
        // 3 of 4 leaves match boundary -> 0.75
        assert!((mastery - 0.75).abs() < 1e-4);
    }
}
