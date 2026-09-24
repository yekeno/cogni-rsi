//! Tree delta tracking and node closure status.
//!
//! Mirrors `docs/PROPOSAL.md` §3.3 & §7.2:
//! Tracks `new_nodes`, `closed_nodes`, and `reopened_nodes` across
//! exploration and dreaming steps.

use cogni_core::{CognitionNode, FailureClass};
use serde::{Deserialize, Serialize};

/// Summary of structural changes during a tree merge or outer iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TreeDelta {
    /// Newly discovered or added nodes.
    pub new_nodes: u32,
    /// Nodes that transitioned into a closed/failed state.
    pub closed_nodes: u32,
    /// Nodes that transitioned from a closed/failed state back to open/valid.
    pub reopened_nodes: u32,
}

impl TreeDelta {
    pub fn new(new_nodes: u32, closed_nodes: u32, reopened_nodes: u32) -> Self {
        Self {
            new_nodes,
            closed_nodes,
            reopened_nodes,
        }
    }

    /// Net progress: `nodes_added - nodes_closed` (mirrors §3.3).
    pub fn net(&self) -> i64 {
        self.new_nodes as i64 - self.closed_nodes as i64
    }
}

/// Determine whether a node is considered "closed" (abandoned or failed).
/// A node is closed if it is marked invalid or has an error failure class.
pub fn is_closed(node: &CognitionNode) -> bool {
    !node.valid || !matches!(node.failure_class, FailureClass::Ok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use cogni_core::{NodeKind, Score};
    use serde_json::json;
    use ulid::Ulid;

    fn make_test_node(valid: bool, failure_class: FailureClass) -> CognitionNode {
        CognitionNode {
            id: Ulid::new(),
            parent: None,
            iteration: 0,
            kind: NodeKind::Concept,
            title: "test".into(),
            workspace: json!({}),
            observation: "".into(),
            score: Score::zero(),
            tags: vec![],
            evidence: vec![],
            failure_class,
            valid,
            no_total: false,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn is_closed_checks_valid_and_failure_class() {
        assert!(!is_closed(&make_test_node(true, FailureClass::Ok)));
        assert!(is_closed(&make_test_node(false, FailureClass::Ok)));
        assert!(is_closed(&make_test_node(true, FailureClass::Error)));
        assert!(is_closed(&make_test_node(true, FailureClass::Invalid)));
        assert!(is_closed(&make_test_node(true, FailureClass::DeltaVsBaseline)));
    }

    #[test]
    fn net_progress_calculation() {
        let delta = TreeDelta::new(8, 3, 1);
        assert_eq!(delta.net(), 5);
    }
}
