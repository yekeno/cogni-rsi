//! Domain types for cogni-rsi.
//!
//! These types mirror the entities described in `docs/PROPOSAL.md` and
//! correspond to the discovery tree / scoring primitives from
//! Dream-RSI (Zheng et al., 2609.14858v1).
//!
//! See also: `crates/cogni-replay` for the prefix-only simulator interface,
//! `crates/cogni-policy` for the policy trait.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

mod question;
pub use question::CognitionQuestion;

/// Stable identifier for a node in the cognition tree.
pub type NodeId = Ulid;

/// What a node represents in the cognition tree.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// A chunk of documentation (PDF / HTML / Markdown).
    Document,
    /// A chunk of source code (function / block).
    Code,
    /// A call to a read-only SaaS API.
    ApiCall,
    /// A concept / boundary / abstraction the agent has abstracted.
    Concept,
    /// A boundary marker (where the SaaS ends vs where the integrator begins).
    Boundary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Score {
    /// 0..=1 — how well the agent understands this node.
    pub understanding: f32,
    /// 0..=1 — does the node fit inside a declared boundary region?
    pub boundary_fit: f32,
    /// 0..=1 — how likely this node will be reused by future reasoning.
    pub reuse_potential: f32,
    /// 0..=1 — self-reported confidence on the score itself.
    pub confidence: f32,
}

impl Score {
    pub fn zero() -> Self {
        Self {
            understanding: 0.0,
            boundary_fit: 0.0,
            reuse_potential: 0.0,
            confidence: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    Ok,
    Error,
    DeltaVsBaseline,
    DeltaVsParent,
    Invalid,
    Nototal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRef {
    /// Source kind, e.g. "pdf:finance/ap.pdf", "git:src/gl/ap.rs", "api:/v1/bill".
    pub source: String,
    /// Page / line / range hint.
    pub locator: String,
}

/// A single node in the cognition tree.
///
/// Mirrors Dream-RSI's discovery-tree node: parent, workspace snapshot,
/// observation, evaluation diagnostics and score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionNode {
    pub id: NodeId,
    pub parent: Option<NodeId>,
    pub iteration: u32,
    pub kind: NodeKind,
    pub title: String,
    pub workspace: serde_json::Value,
    pub observation: String,
    pub score: Score,
    pub tags: Vec<Tag>,
    pub evidence: Vec<EvidenceRef>,
    pub failure_class: FailureClass,
    pub valid: bool,
    pub no_total: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionTree {
    pub root: NodeId,
    pub nodes: HashMap<NodeId, CognitionNode>,
}

impl CognitionTree {
    pub fn new(root: CognitionNode) -> Self {
        let id = root.id;
        let mut nodes = HashMap::new();
        nodes.insert(id, root);
        Self { root: id, nodes }
    }

    pub fn insert(&mut self, node: CognitionNode) {
        self.nodes.insert(node.id, node);
    }

    pub fn get(&self, id: NodeId) -> Option<&CognitionNode> {
        self.nodes.get(&id)
    }
}

/// "Direction declaration" — corresponds to Dream-RSI's `GridPlan`.
///
/// This is what the LLM writes to express: "next, I'll open these branches,
/// refine these existing frontiers, with this width/depth budget."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionPlan {
    pub iteration: u32,
    pub opened_branches: Vec<BranchTarget>,
    pub refine_targets: Vec<NodeId>,
    pub max_width: usize,
    pub max_depth: usize,
    pub stopping_rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchTarget {
    pub label: String,
    pub kind: NodeKind,
    pub locator: serde_json::Value,
    pub reason: String,
}

/// Aggregate progress snapshot for a given day / session.
///
/// Written once per outer iteration; TUI consumes it for the header strip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyProgress {
    pub day: u32,
    pub new_nodes: u32,
    pub closed_nodes: u32,
    pub reopened_nodes: u32,
    /// Weighted mean of `Score::understanding` over the frontier.
    pub goal_progress: f32,
    /// |frontier ∩ boundary| / |frontier|.
    pub boundary_mastery: f32,
    pub pareto_auc: f32,
    pub parallel_penalty: f32,
    pub current_beta: f32,
    pub selected_policy: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_zero_is_zero() {
        let z = Score::zero();
        assert_eq!(z.understanding, 0.0);
        assert_eq!(z.boundary_fit, 0.0);
    }

    #[test]
    fn cognition_tree_starts_with_root() {
        let root = CognitionNode {
            id: Ulid::new(),
            parent: None,
            iteration: 0,
            kind: NodeKind::Concept,
            title: "root".into(),
            workspace: serde_json::json!({}),
            observation: "".into(),
            score: Score::zero(),
            tags: vec![],
            evidence: vec![],
            failure_class: FailureClass::Ok,
            valid: true,
            no_total: false,
            created_at: Utc::now(),
        };
        let tree = CognitionTree::new(root.clone());
        assert_eq!(tree.root, root.id);
        assert!(tree.get(root.id).is_some());
    }
}