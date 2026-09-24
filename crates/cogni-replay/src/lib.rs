//! Replay simulator (mirrors Dream-RSI §3.2).
//!
//! The replay store turns the accumulated cognition history into a
//! "replayable world": the agent can `probe_batch` cells of the tree
//! and learn their recorded `observation` and `score`, but it may **not**
//! fabricate observations that were never recorded.
//!
//! This is the strict invariant that turns dreaming from "made-up training
//! data" into a faithful emulator of past online exploration.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use cogni_core::{CognitionNode, CognitionTree, FailureClass, NodeId, Score};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("probe refers to non-existent cell {0}")]
    UnknownCell(NodeId),
    #[error("probe batch must be non-empty")]
    EmptyBatch,
    #[error("duplicate cell {0} in batch")]
    DuplicateCell(NodeId),
    #[error("cell {0} is not legal in current prefix")]
    IllegalCell(NodeId),
    #[error("closed cell {0} cannot be probed without explicit reopen")]
    ClosedCell(NodeId),
}

pub type Result<T> = std::result::Result<T, ReplayError>;

/// Identifies a cell to reveal during a probe.
///
/// Mirrors Dream-RSI's `CellMeta(branch, attempt, parent_id, seq, tags)`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CellMeta {
    pub node_id: NodeId,
    pub tags: Vec<String>,
}

/// What a reveal returns. Strictly read-only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub node_id: NodeId,
    pub title: String,
    pub kind: cogni_core::NodeKind,
    pub observation: String,
    pub score: Score,
    pub failure_class: FailureClass,
    pub valid: bool,
    pub no_total: bool,
}

/// A read-only handle to the underlying history.
#[async_trait]
pub trait ReplayStore: Send + Sync {
    /// Construct a fresh, reset session — observes only the root node.
    async fn reset(&self) -> Result<ReplaySession>;
}

/// Internal lookup surface over the immutable history snapshot.
#[async_trait]
trait ReplayStoreInner: Send + Sync {
    async fn root(&self) -> CognitionNode;
    async fn lookup(&self, id: NodeId) -> Result<CognitionNode>;
    async fn children(&self, id: NodeId) -> Result<Vec<CognitionNode>>;
    async fn is_legal_root(&self, id: NodeId) -> Result<bool>;
    async fn baseline_score(&self) -> Result<Score>;
}

/// Live state of a single dreaming session.
///
/// Tracks three sets:
/// * `revealed` — cells already observed (visible to the policy)
/// * `closed`   — cells whose frontier has been deliberately abandoned
/// * `frontier` — revealed leaves that still have unrevealed recorded children
pub struct ReplaySession {
    inner: Arc<dyn ReplayStoreInner>,
    revealed: HashSet<NodeId>,
    closed: HashSet<NodeId>,
    revealed_nodes: HashMap<NodeId, CognitionNode>,
    /// Number of decision rounds taken in this session (incremented
    /// whenever `legal_actions()` runs).
    rounds: u32,
}

impl std::fmt::Debug for ReplaySession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplaySession")
            .field("revealed", &self.revealed)
            .field("closed", &self.closed)
            .field("revealed_count", &self.revealed_nodes.len())
            .field("rounds", &self.rounds)
            .finish()
    }
}

impl ReplaySession {
    pub(crate) fn new(inner: Arc<dyn ReplayStoreInner>) -> Self {
        Self {
            inner,
            revealed: HashSet::new(),
            closed: HashSet::new(),
            revealed_nodes: HashMap::new(),
            rounds: 0,
        }
    }

    /// Number of decision rounds taken so far in this session.
    pub fn decision_rounds(&self) -> u32 {
        self.rounds
    }

    /// Backwards-compat helper used by `cogni-dreamer`.
    pub fn rounds_field(&self) -> u32 {
        self.rounds
    }

    /// Subtree the policy has actually revealed.
    pub async fn observed(&self) -> Result<CognitionTree> {
        // Build a cognition tree from revealed nodes; if nothing revealed yet,
        // include just the root.
        let mut tree = if self.revealed_nodes.is_empty() {
            CognitionTree::new(self.inner.root().await)
        } else {
            let mut iter = self.revealed_nodes.values();
            let first = iter.next().expect("non-empty").clone();
            CognitionTree::new(first)
        };
        for n in self.revealed_nodes.values() {
            tree.insert(n.clone());
        }
        Ok(tree)
    }

    /// Actions legal from the current revealed prefix.
    pub async fn legal_actions(&mut self) -> Result<Vec<LegalAction>> {
        // Each call represents one policy decision round.
        let local_rounds = self.rounds;
        self.rounds = local_rounds + 1;
        let mut out = Vec::new();
        let root = self.inner.root().await;
        // 1. Roots that are still unopened (not in `revealed`, not closed).
        //    The root is *always* a legal opening target until revealed.
        if !self.revealed.contains(&root.id) && !self.closed.contains(&root.id) {
            out.push(LegalAction::Open {
                cell: CellMeta {
                    node_id: root.id,
                    tags: vec!["root".into()],
                },
            });
        }
        // 2. Frontiers: revealed leaves whose children are still unobserved.
        for v in self.revealed_nodes.values() {
            let kids = self.inner.children(v.id).await?;
            if kids.is_empty() && !self.closed.contains(&v.id) {
                continue;
            }
            for k in kids {
                if !self.revealed.contains(&k.id) && !self.closed.contains(&k.id) {
                    out.push(LegalAction::Refine {
                        cell: CellMeta {
                            node_id: k.id,
                            tags: vec!["frontier".into()],
                        },
                    });
                }
            }
        }
        // 3. Recoverable failures: revealed, failure_class != Ok, not closed.
        for v in self.revealed_nodes.values() {
            if !matches!(v.failure_class, FailureClass::Ok)
                && !self.closed.contains(&v.id)
            {
                out.push(LegalAction::Recover {
                    cell: CellMeta {
                        node_id: v.id,
                        tags: vec!["recover".into()],
                    },
                });
            }
        }
        // 4. Close actions on revealed leaves with failure or low attainment.
        for v in self.revealed_nodes.values() {
            if self.closed.contains(&v.id) {
                continue;
            }
            if !matches!(v.failure_class, FailureClass::Ok) || v.score.understanding < 0.2 {
                out.push(LegalAction::Close {
                    cell: CellMeta {
                        node_id: v.id,
                        tags: vec!["close".into()],
                    },
                });
            }
        }
        // 5. Stop is always legal once everything else is exhausted.
        if out.is_empty() {
            out.push(LegalAction::Stop);
        }
        Ok(out)
    }

    /// Roots not yet opened.
    pub async fn legal_roots(&self) -> Result<Vec<NodeId>> {
        let root = self.inner.root().await;
        let mut out = Vec::new();
        for r in self.roots(&root) {
            if !self.revealed.contains(&r) && !self.closed.contains(&r) {
                out.push(r);
            }
        }
        Ok(out)
    }

    /// Closed cells (closed branches that could be reopened).
    pub fn closed(&self) -> Vec<NodeId> {
        self.closed.iter().copied().collect()
    }

    /// Frontend: open a closed branch back up (Dream-RSI §3.2 "later successful
    /// result reopens the branch and cancels closure based only on an earlier
    /// failure").
    pub fn reopen(&mut self, id: NodeId) {
        self.closed.remove(&id);
    }

    /// Mark a node as deliberately closed.
    pub fn close(&mut self, id: NodeId) {
        if self.revealed.contains(&id) {
            self.closed.insert(id);
        }
    }

    /// Reveal a batch of cells, calling `on_reveal` once per newly-observed node.
    ///
    /// Strict invariant: cells in `cells` must be **legal** in the current
    /// prefix (root, frontier, or recoverable failure) and must not be in
    /// `closed` without an explicit `reopen`.
    pub async fn probe_batch<F>(
        &mut self,
        cells: &[CellMeta],
        mut on_reveal: F,
    ) -> Result<()>
    where
        F: FnMut(NodeId, &CognitionNode),
    {
        if cells.is_empty() {
            return Err(ReplayError::EmptyBatch);
        }
        let mut seen = HashSet::new();
        for c in cells {
            if !seen.insert(c.node_id) {
                return Err(ReplayError::DuplicateCell(c.node_id));
            }
            if self.closed.contains(&c.node_id) {
                return Err(ReplayError::ClosedCell(c.node_id));
            }
            let legal = self.is_legal(c.node_id).await?;
            if !legal {
                return Err(ReplayError::IllegalCell(c.node_id));
            }
            let node = self.inner.lookup(c.node_id).await?;
            self.revealed.insert(c.node_id);
            self.revealed_nodes.insert(c.node_id, node.clone());
            on_reveal(c.node_id, &node);
        }
        Ok(())
    }

    pub async fn baseline_score(&self) -> Result<Score> {
        self.inner.baseline_score().await
    }

    /// Best score observed so far in this session.
    pub fn best_so_far(&self) -> Score {
        self.revealed_nodes
            .values()
            .map(|n| n.score.clone())
            .max_by(|a, b| a.understanding.partial_cmp(&b.understanding).unwrap())
            .unwrap_or_else(Score::zero)
    }

    /// Count of probe calls (cells observed) in this session.
    pub fn probes(&self) -> u32 {
        self.revealed_nodes.len() as u32
    }

    async fn is_legal(&self, id: NodeId) -> Result<bool> {
        // Root is always legal on first reveal.
        let root = self.inner.root().await;
        if id == root.id {
            return Ok(true);
        }
        // Legal root (multi-root histories): not revealed, not closed.
        if self.inner.is_legal_root(id).await?
            && !self.revealed.contains(&id)
            && !self.closed.contains(&id)
        {
            return Ok(true);
        }
        // Child of any *revealed* node whose parent is also revealed
        // (we walk strictly through the revealed prefix).
        for (parent_id, parent_node) in self.revealed_nodes.iter() {
            for c in self.inner.children(*parent_id).await? {
                if c.id == id && !self.revealed.contains(&id) {
                    // Also ensure the parent is *itself* revealed, which it
                    // is by construction. Done.
                    let _ = parent_node;
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn roots(&self, root: &CognitionNode) -> Vec<NodeId> {
        // In a single-root history the only legal root is the root node.
        // Multi-root histories are not modelled here.
        let _ = root;
        vec![]
    }
}

/// Helpers for building simple legal-action sets in tests and concrete impls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LegalAction {
    Probe { cell: CellMeta },
    Open { cell: CellMeta },
    Close { cell: CellMeta },
    Refine { cell: CellMeta },
    Recover { cell: CellMeta },
    Stop,
}

/// A local, in-memory replay store seeded from a known tree.
///
/// Used by `cogni-dreamer` for offline dreaming.
#[derive(Clone)]
pub struct LocalReplayStore {
    nodes: HashMap<NodeId, CognitionNode>,
    root: NodeId,
    children: HashMap<NodeId, Vec<NodeId>>,
    roots: Vec<NodeId>,
    baseline: Score,
}

impl LocalReplayStore {
    pub fn from_tree(tree: &CognitionTree) -> Self {
        let mut children: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut roots = Vec::new();
        for n in tree.nodes.values() {
            if let Some(p) = n.parent {
                children.entry(p).or_default().push(n.id);
            } else {
                roots.push(n.id);
            }
        }
        let baseline = tree
            .get(tree.root)
            .map(|n| n.score.clone())
            .unwrap_or_else(Score::zero);
        Self {
            nodes: tree.nodes.clone(),
            root: tree.root,
            children,
            roots,
            baseline,
        }
    }

    /// Snapshot the replay store's underlying cognition tree.
    pub fn to_tree(&self) -> CognitionTree {
        CognitionTree {
            root: self.root,
            nodes: self.nodes.clone(),
        }
    }
}

#[async_trait]
impl ReplayStoreInner for LocalReplayStore {
    async fn root(&self) -> CognitionNode {
        self.nodes.get(&self.root).cloned().expect("root exists")
    }

    async fn lookup(&self, id: NodeId) -> Result<CognitionNode> {
        self.nodes
            .get(&id)
            .cloned()
            .ok_or(ReplayError::UnknownCell(id))
    }

    async fn children(&self, id: NodeId) -> Result<Vec<CognitionNode>> {
        Ok(self
            .children
            .get(&id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|c| self.nodes.get(&c).cloned())
            .collect())
    }

    async fn is_legal_root(&self, id: NodeId) -> Result<bool> {
        Ok(self.roots.contains(&id))
    }

    async fn baseline_score(&self) -> Result<Score> {
        Ok(self.baseline.clone())
    }
}

#[async_trait]
impl ReplayStore for LocalReplayStore {
    async fn reset(&self) -> Result<ReplaySession> {
        Ok(ReplaySession::new(Arc::new(self.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogni_core::{CognitionTree, NodeKind};
    use chrono::Utc;
    use serde_json::json;
    use ulid::Ulid;

    fn make_node(parent: Option<NodeId>, title: &str) -> CognitionNode {
        CognitionNode {
            id: Ulid::new(),
            parent,
            iteration: 0,
            kind: NodeKind::Concept,
            title: title.into(),
            workspace: json!({}),
            observation: format!("obs-{title}"),
            score: Score::zero(),
            tags: vec![],
            evidence: vec![],
            failure_class: FailureClass::Ok,
            valid: true,
            no_total: false,
            created_at: Utc::now(),
        }
    }

    fn make_node_with(parent: Option<NodeId>, title: &str, score: Score, fail: FailureClass) -> CognitionNode {
        CognitionNode {
            id: Ulid::new(),
            parent,
            iteration: 0,
            kind: NodeKind::Concept,
            title: title.into(),
            workspace: json!({}),
            observation: format!("obs-{title}"),
            score,
            tags: vec![],
            evidence: vec![],
            failure_class: fail,
            valid: true,
            no_total: false,
            created_at: Utc::now(),
        }
    }

    fn seed_tree() -> CognitionTree {
        let root = make_node(None, "root");
        let c1 = make_node(Some(root.id), "c1");
        let c2 = make_node(Some(root.id), "c2");
        let gc = make_node(Some(c1.id), "gc");
        let mut tree = CognitionTree::new(root);
        tree.insert(c1);
        tree.insert(c2);
        tree.insert(gc);
        tree
    }

    #[tokio::test]
    async fn empty_batch_is_rejected() {
        let store = LocalReplayStore::from_tree(&seed_tree());
        let mut session = store.reset().await.unwrap();
        let err = session.probe_batch(&[], |_, _| {}).await.unwrap_err();
        matches!(err, ReplayError::EmptyBatch);
    }

    #[tokio::test]
    async fn duplicate_cells_in_batch_are_rejected() {
        let store = LocalReplayStore::from_tree(&seed_tree());
        let mut session = store.reset().await.unwrap();
        let id = store.root;
        let cells = vec![
            CellMeta {
                node_id: id,
                tags: vec![],
            },
            CellMeta {
                node_id: id,
                tags: vec![],
            },
        ];
        let err = session.probe_batch(&cells, |_, _| {}).await.unwrap_err();
        matches!(err, ReplayError::DuplicateCell(_));
    }

    #[tokio::test]
    async fn probe_returns_recorded_observation_not_fabrication() {
        let store = LocalReplayStore::from_tree(&seed_tree());
        let mut session = store.reset().await.unwrap();
        let mut seen: Vec<String> = vec![];
        session
            .probe_batch(
                &[CellMeta {
                    node_id: store.root,
                    tags: vec![],
                }],
                |_, n| seen.push(n.observation.clone()),
            )
            .await
            .unwrap();
        assert_eq!(seen, vec!["obs-root".to_string()]);
    }

    #[tokio::test]
    async fn probing_unrevealed_grandchild_is_illegal_until_parent_revealed() {
        let tree = seed_tree();
        let store = LocalReplayStore::from_tree(&tree);
        let root_id = tree.root;
        let c1_id = tree
            .nodes
            .values()
            .find(|n| n.title == "c1")
            .unwrap()
            .id;
        let gc_id = tree
            .nodes
            .values()
            .find(|n| n.title == "gc")
            .unwrap()
            .id;
        let mut session = store.reset().await.unwrap();
        // Without revealing root or c1, gc is illegal.
        let err = session
            .probe_batch(
                &[CellMeta {
                    node_id: gc_id,
                    tags: vec![],
                }],
                |_, _| {},
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ReplayError::IllegalCell(_)));
        // Reveal root — c1 becomes legal, but gc still isn't.
        session
            .probe_batch(
                &[CellMeta {
                    node_id: root_id,
                    tags: vec![],
                }],
                |_, _| {},
            )
            .await
            .unwrap();
        let err = session
            .probe_batch(
                &[CellMeta {
                    node_id: gc_id,
                    tags: vec![],
                }],
                |_, _| {},
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ReplayError::IllegalCell(_)));
        // Reveal c1 — gc now legal.
        session
            .probe_batch(
                &[CellMeta {
                    node_id: c1_id,
                    tags: vec![],
                }],
                |_, _| {},
            )
            .await
            .unwrap();
        session
            .probe_batch(
                &[CellMeta {
                    node_id: gc_id,
                    tags: vec![],
                }],
                |_, _| {},
            )
            .await
            .unwrap();
        assert_eq!(session.probes(), 3);
    }

    #[tokio::test]
    async fn closed_branch_cannot_be_probed_until_reopened() {
        let store = LocalReplayStore::from_tree(&seed_tree());
        let mut session = store.reset().await.unwrap();
        // reveal root, then close it
        session.probe_batch(
            &[CellMeta {
                node_id: store.root,
                tags: vec![],
            }],
            |_, _| {},
        )
        .await
        .unwrap();
        session.close(store.root);
        let err = session
            .probe_batch(
                &[CellMeta {
                    node_id: store.root,
                    tags: vec![],
                }],
                |_, _| {},
            )
            .await
            .unwrap_err();
        matches!(err, ReplayError::ClosedCell(_));
        session.reopen(store.root);
        session
            .probe_batch(
                &[CellMeta {
                    node_id: store.root,
                    tags: vec![],
                }],
                |_, _| {},
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn best_so_far_tracks_revealed_max() {
        let tree = seed_tree();
        let store = LocalReplayStore::from_tree(&tree);
        let mut session = store.reset().await.unwrap();
        let root_score = tree.get(store.root).unwrap().score.clone();
        session
            .probe_batch(
                &[CellMeta {
                    node_id: store.root,
                    tags: vec![],
                }],
                |_, _| {},
            )
            .await
            .unwrap();
        assert_eq!(session.best_so_far().understanding, root_score.understanding);
        assert_eq!(session.probes(), 1);
    }

    #[tokio::test]
    async fn legal_actions_includes_recover_for_failed_nodes() {
        let mut root = make_node(None, "r");
        root.failure_class = FailureClass::Error;
        let id = root.id;
        let tree = CognitionTree::new(root);
        let store = LocalReplayStore::from_tree(&tree);
        let mut session = store.reset().await.unwrap();
        session
            .probe_batch(
                &[CellMeta {
                    node_id: id,
                    tags: vec![],
                }],
                |_, _| {},
            )
            .await
            .unwrap();
        let actions = session.legal_actions().await.unwrap();
        assert!(actions
            .iter()
            .any(|a| matches!(a, LegalAction::Recover { .. })));
        assert!(actions
            .iter()
            .any(|a| matches!(a, LegalAction::Close { .. })));
    }

    #[tokio::test]
    async fn legal_actions_eventually_returns_stop_when_exhausted() {
        let mut root = make_node_with(None, "r", Score::zero(), FailureClass::Ok);
        root.score.understanding = 0.9;
        let id = root.id;
        let tree = CognitionTree::new(root);
        let store = LocalReplayStore::from_tree(&tree);
        let mut session = store.reset().await.unwrap();
        session
            .probe_batch(
                &[CellMeta {
                    node_id: id,
                    tags: vec![],
                }],
                |_, _| {},
            )
            .await
            .unwrap();
        let actions = session.legal_actions().await.unwrap();
        assert!(actions.iter().any(|a| matches!(a, LegalAction::Stop)));
    }
}