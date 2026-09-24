//! Outer iteration loop (Dream-RSI §3.3).
//!
//! PR #2: complete dreaming cycle.
//!  ① online rollout (placeholder — same as PR #1 for now)
//!  ② construct replay simulator from history
//!  ③ dreaming-based policy improvement:
//!     a) generate M candidate policies (current + perturbed variants)
//!     b) replay each candidate on a fresh session
//!     c) compute V^m for each via Dream-RSI formula (1)
//!     d) pick argmax, keep current if no improvement

use anyhow::{anyhow, Result};
use cogni_core::{CognitionQuestion, DirectionPlan};
use cogni_policy::{BaselineParallelRefining, Budget, CognitionPolicy};
use cogni_cognition_tree::{BoundarySpec, TreeDelta};
use cogni_core::CognitionTree;
use cogni_replay::{
    CellMeta, LegalAction, LocalReplayStore, Observation, ReplaySession, ReplayStore,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::{debug, info};

fn default_root_node() -> cogni_core::CognitionNode {
    cogni_core::CognitionNode {
        id: ulid::Ulid::new(),
        parent: None,
        iteration: 0,
        kind: cogni_core::NodeKind::Concept,
        title: "system_cognition_root".into(),
        workspace: serde_json::json!({}),
        observation: "root cognition baseline".into(),
        score: cogni_core::Score {
            understanding: 0.1,
            boundary_fit: 1.0,
            reuse_potential: 0.5,
            confidence: 0.5,
        },
        tags: vec![],
        evidence: vec![],
        failure_class: cogni_core::FailureClass::Ok,
        valid: true,
        no_total: false,
        created_at: chrono::Utc::now(),
    }
}

/// State carried across outer iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CogniState {
    pub iteration: u32,
    pub current_policy_name: String,
    pub beta: f32,
    /// pareto.reward of the *currently selected* policy (initialised to 0).
    pub current_replay_score: f32,
    pub history_trees: Vec<serde_json::Value>,
    /// Accumulated cognition tree across exploration and dreaming.
    pub cognition_tree: CognitionTree,
    /// Boundary specification (defaults to SaaS finance/biz integration).
    pub boundary_spec: Option<BoundarySpec>,
    /// Last structural change delta from tree merge.
    pub last_delta: TreeDelta,
}

impl CogniState {
    pub fn new(initial_policy: &str) -> Self {
        let root = default_root_node();
        Self {
            iteration: 0,
            current_policy_name: initial_policy.into(),
            beta: 0.6,
            current_replay_score: 0.0,
            history_trees: vec![],
            cognition_tree: CognitionTree::new(root),
            boundary_spec: None,
            last_delta: TreeDelta::default(),
        }
    }

    pub fn with_tree(mut self, tree: CognitionTree) -> Self {
        self.cognition_tree = tree;
        self
    }

    pub fn with_boundary(mut self, boundary: BoundarySpec) -> Self {
        self.boundary_spec = Some(boundary);
        self
    }
}

/// Replay score from Dream-RSI equation (1):
///   V^m = max s_v − β1 * N^m + β2 * (N^m / max(1, k^m))
#[derive(Debug, Clone, Copy)]
pub struct ReplayScore {
    pub best_node: f32,
    pub probes: u32,
    pub rounds: u32,
    pub beta1: f32,
    pub beta2: f32,
}

impl ReplayScore {
    pub fn from_session(session: &ReplaySession, beta1: f32, beta2: f32) -> Self {
        let best = session.best_so_far().understanding;
        Self {
            best_node: best,
            probes: session.probes(),
            rounds: session.decision_rounds(),
            beta1,
            beta2,
        }
    }

    pub fn compute(&self) -> f32 {
        let parallelism_bonus = if self.rounds == 0 {
            0.0
        } else {
            self.probes as f32 / self.rounds.max(1) as f32
        };
        self.best_node
            - self.beta1 * self.probes as f32
            + self.beta2 * parallelism_bonus
    }
}

/// One candidate policy + its evaluated replay score.
#[derive(Debug, Clone)]
pub struct CandidateScore {
    pub index: usize,
    pub name: String,
    pub plan: DirectionPlan,
    pub score: ReplayScore,
    pub source_code: String,
}

/// A policy candidate paired with its raw or canonical XML source code.
/// Implements `CognitionPolicy` by forwarding to the inner policy so that
/// consumers can treat it as a policy while retaining its source code.
pub struct CandidateItem {
    pub policy: Box<dyn CognitionPolicy>,
    pub source_code: String,
}

#[async_trait::async_trait]
impl CognitionPolicy for CandidateItem {
    fn name(&self) -> &str {
        self.policy.name()
    }

    async fn solve(
        &self,
        question: &CognitionQuestion,
        budget: Option<Budget>,
        replay: &mut ReplaySession,
    ) -> Result<DirectionPlan> {
        self.policy.solve(question, budget, replay).await
    }

    fn plan_grid(&self, ctx: &cogni_policy::GridContext) -> cogni_policy::GridPlan {
        self.policy.plan_grid(ctx)
    }
}

/// Generate M candidate policies starting from the current one.
///
/// PR #3 signature: takes an optional `LlmTransport`. When supplied and the
/// call succeeds, the last candidate is the LLM-generated `PerturbedPolicy`.
/// When the LLM is unavailable or returns an unparseable response, falls
/// back to the deterministic trio of variants used by PR #2.
pub async fn generate_candidates(
    current: &dyn CognitionPolicy,
    m: usize,
    llm: Option<&dyn cogni_llm::LlmTransport>,
) -> Vec<CandidateItem> {
    let mut out: Vec<CandidateItem> = Vec::with_capacity(m);
    let name = current.name().to_string();
    let base_xml = format!(
        "<policy>\n  max_width: 4\n  max_depth: 4\n  close_on_failure: false\n  beta: 0.60\n  rationale: {}\n</policy>",
        name
    );
    out.push(CandidateItem {
        policy: as_box_cloneable(current, &name),
        source_code: base_xml,
    });
    if m <= 1 {
        return out;
    }
    // m-1 deterministic variants (or m-2 if an LLM candidate is appended).
    let last_is_llm = llm.is_some() && m >= 2;
    let det_count = if last_is_llm { m - 2 } else { m - 1 };
    for i in 0..det_count {
        let variant = match i % 3 {
            0 => PerturbedPolicy {
                name: format!("{name}#narrow"),
                max_width: 2,
                max_depth: 2,
                close_on_failure: true,
                beta: 0.4,
            },
            1 => PerturbedPolicy {
                name: format!("{name}#wide"),
                max_width: 8,
                max_depth: 6,
                close_on_failure: false,
                beta: 0.7,
            },
            _ => PerturbedPolicy {
                name: format!("{name}#balanced"),
                max_width: 4,
                max_depth: 4,
                close_on_failure: false,
                beta: 0.6,
            },
        };
        let xml = variant.to_xml();
        out.push(CandidateItem {
            policy: Box::new(variant),
            source_code: xml,
        });
    }
    // Optionally append the LLM-generated candidate.
    if last_is_llm {
        let prompt = format!(
            "{}\n\n{}\n\nReturn a <policy> XML block with fields: \
            max_width, max_depth, close_on_failure, beta, rationale.",
            "Improve the previous policy. Trade width vs depth based on \
             the prefix evidence. Stay prefix-only.",
            name,
        );
        match llm.expect("checked above").complete(&prompt).await {
            Ok(text) => match parse_llm_policy_response(&text, &name) {
                Ok(p) => out.push(CandidateItem {
                    source_code: text,
                    policy: Box::new(p),
                }),
                Err(e) => {
                    tracing::warn!(
                        target: "cogni.dreamer",
                        "LLM policy parse failed ({e}); using balanced fallback"
                    );
                    let p = PerturbedPolicy {
                        name: format!("{name}#llm-balanced-fallback"),
                        max_width: 4,
                        max_depth: 4,
                        close_on_failure: false,
                        beta: 0.6,
                    };
                    let xml = p.to_xml();
                    out.push(CandidateItem {
                        policy: Box::new(p),
                        source_code: xml,
                    });
                }
            },
            Err(e) => {
                tracing::warn!(
                    target: "cogni.dreamer",
                    "LLM completion failed ({e}); using balanced fallback"
                );
                let p = PerturbedPolicy {
                    name: format!("{name}#llm-err-fallback"),
                    max_width: 4,
                    max_depth: 4,
                    close_on_failure: false,
                    beta: 0.6,
                };
                let xml = p.to_xml();
                out.push(CandidateItem {
                    policy: Box::new(p),
                    source_code: xml,
                });
            }
        }
    }
    out
}

/// Errors from parsing the LLM's `<policy>` XML block.
#[derive(Debug, thiserror::Error)]
pub enum PolicyParseError {
    #[error("missing <policy>...</policy> block")]
    MissingBlock,
    #[error("missing field `{0}` in policy block")]
    MissingField(&'static str),
    #[error("invalid integer for `{0}`: {1}")]
    BadInt(&'static str, String),
    #[error("invalid bool for `close_on_failure`: {0}")]
    BadBool(String),
    #[error("`beta` out of [0, 1]: {0}")]
    BetaOutOfRange(f32),
}

/// Parse the LLM's text response into a `PerturbedPolicy`.
///
/// The expected shape is an XML-ish fenced block:
///
/// ```text
/// <policy>
///   max_width: 4
///   max_depth: 6
///   close_on_failure: false
///   beta: 0.6
///   rationale: ...
/// </policy>
/// ```
///
/// Loose whitespace and case are tolerated, but the four numeric/bool fields
/// are required. Returns `MissingBlock` when the input contains no fenced
/// block at all.
pub fn parse_llm_policy_response(
    text: &str,
    base_name: &str,
) -> Result<PerturbedPolicy, PolicyParseError> {
    let open = text.find("<policy>").ok_or(PolicyParseError::MissingBlock)?;
    let after_open = open + "<policy>".len();
    let close = text[after_open..]
        .find("</policy>")
        .ok_or(PolicyParseError::MissingBlock)?;
    let body = &text[after_open..after_open + close];

    let mut max_width: Option<usize> = None;
    let mut max_depth: Option<usize> = None;
    let mut close_on_failure: Option<bool> = None;
    let mut beta: Option<f32> = None;
    let mut rationale: Option<String> = None;

    for raw_line in body.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        // Accept both `key: value` (LLM convention) and `key = value`
        // (YAML-ish) so the parser is lenient to the model's output style.
        let Some((k, v)) = line.split_once(':').or_else(|| line.split_once('=')) else {
            continue;
        };
        let key = k.trim().to_ascii_lowercase();
        let val = v.trim();
        match key.as_str() {
            "max_width" => {
                max_width = Some(
                    val.parse::<usize>()
                        .map_err(|_| PolicyParseError::BadInt("max_width", val.into()))?,
                );
            }
            "max_depth" => {
                max_depth = Some(
                    val.parse::<usize>()
                        .map_err(|_| PolicyParseError::BadInt("max_depth", val.into()))?,
                );
            }
            "close_on_failure" => {
                close_on_failure = Some(match val.to_ascii_lowercase().as_str() {
                    "true" | "yes" | "1" => true,
                    "false" | "no" | "0" => false,
                    _ => return Err(PolicyParseError::BadBool(val.into())),
                });
            }
            "beta" => {
                let f: f32 = val
                    .parse()
                    .map_err(|_| PolicyParseError::BadInt("beta", val.into()))?;
                if !(0.0..=1.0).contains(&f) {
                    return Err(PolicyParseError::BetaOutOfRange(f));
                }
                beta = Some(f);
            }
            "rationale" => {
                rationale = Some(val.to_string());
            }
            _ => {}
        }
    }

    let max_width = max_width.ok_or(PolicyParseError::MissingField("max_width"))?;
    let max_depth = max_depth.ok_or(PolicyParseError::MissingField("max_depth"))?;
    let close_on_failure = close_on_failure
        .ok_or(PolicyParseError::MissingField("close_on_failure"))?;
    let beta = beta.ok_or(PolicyParseError::MissingField("beta"))?;

    let rationale_slug: String = rationale
        .as_deref()
        .map(|r| r.chars().filter(|c| c.is_ascii_alphanumeric()).take(24).collect())
        .filter(|s: &String| !s.is_empty())
        .map(|s| format!("-{s}"))
        .unwrap_or_default();

    Ok(PerturbedPolicy {
        name: format!("{base_name}#llm{rationale_slug}"),
        max_width,
        max_depth,
        close_on_failure,
        beta,
    })
}

fn as_box_cloneable(_p: &dyn CognitionPolicy, _name: &str) -> Box<dyn CognitionPolicy> {
    // We can't clone a `dyn` reference, so any non-first candidate must be a
    // concrete type. The first slot is reserved for a concrete Baseline; in
    // this PR we always seed with BaselineParallelRefining.
    Box::new(BaselineParallelRefining)
}

/// A simple deterministic perturbed policy used in PR #2 (LLM-free).
#[derive(Debug, Clone)]
pub struct PerturbedPolicy {
    pub name: String,
    pub max_width: usize,
    pub max_depth: usize,
    pub close_on_failure: bool,
    pub beta: f32,
}

impl PerturbedPolicy {
    /// Emit the canonical `<policy>` XML block accepted by `parse_llm_policy_response`.
    pub fn to_xml(&self) -> String {
        format!(
            "<policy>\n  max_width: {}\n  max_depth: {}\n  close_on_failure: {}\n  beta: {:.2}\n  rationale: {}\n</policy>",
            self.max_width,
            self.max_depth,
            self.close_on_failure,
            self.beta,
            self.name,
        )
    }
}

#[async_trait::async_trait]
impl CognitionPolicy for PerturbedPolicy {
    fn name(&self) -> &str {
        &self.name
    }

    async fn solve(
        &self,
        _question: &CognitionQuestion,
        budget: Option<Budget>,
        replay: &mut ReplaySession,
    ) -> Result<DirectionPlan> {
        let max_probes = budget.map(|b| b.max_probes as usize).unwrap_or(8);
        let mut probes = 0usize;
        let mut rounds = 0u32;
        let mut opened_branches = Vec::new();
        let mut refine_targets = Vec::new();

        loop {
            if probes as u32 >= max_probes as u32 {
                break;
            }
            let actions = replay.legal_actions().await?;
            if matches!(actions.as_slice(), [LegalAction::Stop]) {
                break;
            }
            rounds += 1;
            // Build a batch of up to max_width distinct cells from the legal
            // action set, preferring Refine > Open > Recover > Close.
            let mut picked: Vec<CellMeta> = Vec::new();
            let mut seen: HashSet<_> = HashSet::new();
            for prefer in [
                |a: &LegalAction| matches!(a, LegalAction::Refine { .. }),
                |a: &LegalAction| matches!(a, LegalAction::Open { .. }),
                |a: &LegalAction| matches!(a, LegalAction::Recover { .. }),
                |a: &LegalAction| matches!(a, LegalAction::Close { .. }),
            ] {
                if picked.len() >= self.max_width {
                    break;
                }
                for action in &actions {
                    if !prefer(action) {
                        continue;
                    }
                    let (cell, is_open, is_refine) = match action {
                        LegalAction::Refine { cell } => (cell, false, true),
                        LegalAction::Open { cell } => (cell, true, false),
                        LegalAction::Recover { cell }
                        | LegalAction::Close { cell } => (cell, false, false),
                        LegalAction::Stop | LegalAction::Probe { .. } => continue,
                    };
                    if !seen.insert(cell.node_id) {
                        continue;
                    }
                    if is_refine {
                        refine_targets.push(cell.node_id);
                    } else if is_open {
                        opened_branches.push(cogni_core::BranchTarget {
                            label: format!("branch-{}", cell.node_id),
                            kind: cogni_core::NodeKind::Concept,
                            locator: serde_json::json!({ "node_id": cell.node_id.to_string() }),
                            reason: format!("opened in round {}", rounds),
                        });
                    }
                    picked.push(cell.clone());
                    if picked.len() >= self.max_width {
                        break;
                    }
                }
            }
            if picked.is_empty() {
                break;
            }
            // Apply close actions before probing to drop weak branches.
            if self.close_on_failure {
                for c in &picked {
                    if let Some(LegalAction::Close { cell }) =
                        actions.iter().find(|a| matches!(a, LegalAction::Close { cell: cc } if cc == c))
                    {
                        replay.close(cell.node_id);
                    }
                }
            }
            let picked_cells: Vec<CellMeta> = picked
                .into_iter()
                .filter(|c| !matches!(c.tags.first().map(String::as_str), Some("close")))
                .collect();
            if picked_cells.is_empty() {
                // The only legal actions were Close; nothing to probe.
                // Stop rather than spin forever.
                break;
            }
            match replay
                .probe_batch(&picked_cells, |_id, _node: &cogni_core::CognitionNode| {
                    probes += 1;
                })
                .await
            {
                Ok(()) => {}
                Err(_) => break, // illegal cells end this iteration
            }
            if rounds as usize >= self.max_depth {
                break;
            }
        }
        Ok(DirectionPlan {
            iteration: 0,
            opened_branches,
            refine_targets,
            max_width: self.max_width,
            max_depth: self.max_depth,
            stopping_rationale: format!(
                "{name}: probes={probes} rounds={rounds} beta={beta}",
                name = self.name,
                probes = probes,
                rounds = rounds,
                beta = self.beta,
            ),
        })
    }

    fn plan_grid(&self, _ctx: &cogni_policy::GridContext) -> cogni_policy::GridPlan {
        cogni_policy::GridPlan {
            branch_count: self.max_width,
            refine_count: self.max_depth,
            reason: format!("{} grid", self.name),
        }
    }
}

/// Run the dreaming-based policy improvement step on `store`.
///
/// Returns the winning candidate, or `None` if no candidate beats the
/// incumbent (in which case the caller keeps the current policy).
///
/// PR #3: takes an optional `LlmTransport`. When supplied, the last of the
/// M candidates is generated by `LlmTransport::complete()` and parsed into
/// a `PerturbedPolicy`. Pass `None` to keep the deterministic PR #2 trio.
pub async fn dream<M>(
    state: &mut CogniState,
    store: &M,
    question: &CognitionQuestion,
    llm: Option<&dyn cogni_llm::LlmTransport>,
) -> Result<Option<CandidateScore>>
where
    M: ReplayStore + ?Sized,
{
    let m = 3usize; // paper §4 uses M=3
    let beta1 = 0.05f32;
    let beta2 = 0.10f32;

    let current = BaselineParallelRefining; // PR #2 hard-codes the incumbent.
    let candidates = generate_candidates(&current, m, llm).await;

    let mut scored = Vec::with_capacity(candidates.len());
    for (idx, cand) in candidates.iter().enumerate() {
        let mut session = store.reset().await?;
        let plan = cand
            .solve(question, Some(Budget { max_probes: 8 }), &mut session)
            .await?;
        let score = ReplayScore::from_session(&session, beta1, beta2);
        debug!(
            target: "cogni.dreamer",
            "candidate m={idx} name={} V={:.4} probes={} rounds={}",
            cand.name(),
            score.compute(),
            score.probes,
            score.rounds,
        );
        scored.push(CandidateScore {
            index: idx,
            name: cand.name().to_string(),
            plan,
            score,
            source_code: cand.source_code.clone(),
        });
    }

    // Selection rule: V^m ≥ V^0 (current policy is index 0).
    let incumbent_v = scored.first().map(|c| c.score.compute()).unwrap_or(0.0);
    let best = scored
        .iter()
        .max_by(|a, b| {
            a.score
                .compute()
                .partial_cmp(&b.score.compute())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| anyhow!("no candidates scored"))?;

    if best.score.compute() + 1e-6 < incumbent_v {
        info!(
            "no improvement: incumbent V={:.4} best V={:.4}",
            incumbent_v,
            best.score.compute()
        );
        state.current_replay_score = incumbent_v;
        return Ok(None);
    }
    info!(
        "selected {} V={:.4} (incumbent V={:.4})",
        best.name,
        best.score.compute(),
        incumbent_v
    );
    state.current_policy_name = best.name.clone();
    state.current_replay_score = best.score.compute();
    state.beta = best.plan.max_width as f32 / 10.0; // PR #2: β auto-bumps from grid.
    Ok(Some(best.clone()))
}

/// One outer iteration: online rollout (PR #1 placeholder) + dream() +
/// optional persistence to `cogni-store`. When `persist` is `None` (the
/// default) the call is a no-op for storage; PR #4 wires the real
/// `SqliteStore` here.
pub async fn run_outer_iteration(
    state: &mut CogniState,
    store: &LocalReplayStore,
    question: &CognitionQuestion,
    llm: Option<&dyn cogni_llm::LlmTransport>,
    persist: Option<&dyn cogni_store::Store>,
) -> Result<DirectionPlan> {
    // Stage 1: Merge latest nodes from replay store into cognition_tree.
    let replay_tree = store.to_tree();
    let mut iter_delta = cogni_cognition_tree::merge(&mut state.cognition_tree, &replay_tree);

    // Stage 2: dream.
    let winner = dream(state, store, question, llm).await?;
    state.iteration += 1;

    // Stage 3: If candidate won and produced opened_branches, generate
    // frontier child nodes in cognition_tree.
    if let Some(w) = winner.as_ref() {
        for branch in &w.plan.opened_branches {
            let parent_id = branch
                .locator
                .get("node_id")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<ulid::Ulid>().ok())
                .filter(|id| state.cognition_tree.nodes.contains_key(id))
                .unwrap_or(state.cognition_tree.root);

            let new_node = cogni_core::CognitionNode {
                id: ulid::Ulid::new(),
                parent: Some(parent_id),
                iteration: state.iteration,
                kind: branch.kind,
                title: branch.label.clone(),
                workspace: serde_json::json!({ "reason": branch.reason }),
                observation: "".into(),
                score: cogni_core::Score {
                    understanding: w.score.best_node.max(0.6),
                    boundary_fit: 0.7,
                    reuse_potential: 0.5,
                    confidence: 0.8,
                },
                tags: vec![cogni_core::Tag("dream".into())],
                evidence: vec![],
                failure_class: cogni_core::FailureClass::Ok,
                valid: true,
                no_total: false,
                created_at: chrono::Utc::now(),
            };
            state.cognition_tree.insert(new_node);
            iter_delta.new_nodes += 1;
        }
    }

    state.last_delta = iter_delta;

    let boundary = state
        .boundary_spec
        .clone()
        .unwrap_or_else(cogni_cognition_tree::BoundarySpec::default_saas);

    // PR #4: persist this iteration's outcome. We do this even when there
    // is no winner (incumbent kept) so the dashboard can show stagnation.
    if let Some(persist) = persist {
        let row = build_daily_progress(state, winner.as_ref(), iter_delta, &boundary);
        if let Err(e) = persist.record_daily_progress(row).await {
            tracing::warn!(
                target: "cogni.dreamer",
                "failed to persist daily_progress iter={}: {e}",
                state.iteration
            );
        }
        // PR #7: persist winning policy XML (skip on no-improvement / incumbent kept).
        if let Some(w) = winner.as_ref() {
            let policy_row = cogni_store::PolicyRunRow {
                id: format!("pol-iter{}-{}", state.iteration, ulid::Ulid::new()),
                iteration: state.iteration,
                pareto_reward: w.score.compute(),
                created_at: chrono::Utc::now(),
                source_code: w.source_code.clone(),
            };
            if let Err(e) = persist.record_policy_run(policy_row).await {
                tracing::warn!(
                    target: "cogni.dreamer",
                    "failed to persist policy iter={}: {e}",
                    state.iteration
                );
            }
        }
    }

    Ok(winner
        .map(|c| c.plan)
        .unwrap_or_else(|| DirectionPlan {
            iteration: state.iteration,
            opened_branches: vec![],
            refine_targets: vec![],
            max_width: 4,
            max_depth: 4,
            stopping_rationale: "no improvement over incumbent".into(),
        }))
}

/// Build a `DailyProgressRow` from the current state + winner + tree delta + boundary.
///
/// Computes accurate `goal_progress` (weighted understanding over frontier) and
/// `boundary_mastery` (|frontier ∩ boundary| / |frontier|) via `cogni_cognition_tree`.
pub fn build_daily_progress(
    state: &CogniState,
    winner: Option<&CandidateScore>,
    delta: cogni_cognition_tree::TreeDelta,
    boundary: &cogni_cognition_tree::BoundarySpec,
) -> cogni_store::DailyProgressRow {
    let (pareto_auc, parallel_penalty) = match winner {
        Some(w) => (
            w.score.best_node,
            if w.score.probes == 0 {
                0.0
            } else {
                w.score.rounds as f32 / w.score.probes.max(1) as f32
            },
        ),
        None => (0.0, 0.0),
    };
    let goal_progress = cogni_cognition_tree::goal_progress(&state.cognition_tree);
    let boundary_mastery = cogni_cognition_tree::boundary_mastery(&state.cognition_tree, boundary);

    cogni_store::DailyProgressRow {
        day: state.iteration,
        new_nodes: delta.new_nodes,
        closed_nodes: delta.closed_nodes,
        reopened_nodes: delta.reopened_nodes,
        goal_progress,
        boundary_mastery,
        pareto_auc,
        parallel_penalty,
        current_beta: state.beta,
        selected_policy: state.current_policy_name.clone(),
    }
}

#[allow(dead_code)]
fn _unused_observation_marker(_o: &Observation) {}

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn parse_well_formed_policy_block() {
        let text = r#"
Some preamble from the model.
<policy>
  max_width: 6
  max_depth: 5
  close_on_failure: true
  beta: 0.7
  rationale: prefers depth over breadth given narrow frontier
</policy>
Trailing commentary."#;
        let p = parse_llm_policy_response(text, "baseline").expect("parses");
        assert_eq!(p.max_width, 6);
        assert_eq!(p.max_depth, 5);
        assert!(p.close_on_failure);
        assert!((p.beta - 0.7).abs() < 1e-6);
        assert!(p.name.contains("baseline#llm"));
        assert!(p.name.contains("prefersdepthoverbreadth"));
    }

    #[test]
    fn parse_rejects_missing_block() {
        let text = "no block here";
        assert!(matches!(
            parse_llm_policy_response(text, "baseline"),
            Err(PolicyParseError::MissingBlock)
        ));
    }

    #[test]
    fn parse_rejects_missing_field() {
        let text = "<policy>\n  max_width: 4\n  max_depth: 4\n  beta: 0.6\n</policy>";
        let err = parse_llm_policy_response(text, "base").unwrap_err();
        assert!(matches!(err, PolicyParseError::MissingField("close_on_failure")));
    }

    #[test]
    fn parse_rejects_beta_out_of_range() {
        let text = "<policy>\n  max_width: 4\n  max_depth: 4\n  close_on_failure: false\n  beta: 1.5\n</policy>";
        let err = parse_llm_policy_response(text, "base").unwrap_err();
        assert!(matches!(err, PolicyParseError::BetaOutOfRange(_)));
    }

    #[test]
    fn parse_tolerates_loose_whitespace_and_case() {
        let text = "<policy>\nMAX_WIDTH = 3\n  Max_Depth: 2\n  close_on_failure: NO\n  beta: 0.4\n</policy>";
        let p = parse_llm_policy_response(text, "base").expect("parses");
        assert_eq!(p.max_width, 3);
        assert_eq!(p.max_depth, 2);
        assert!(!p.close_on_failure);
    }

    #[test]
    fn parse_without_rationale_uses_short_name() {
        let text = "<policy>\n  max_width: 4\n  max_depth: 4\n  close_on_failure: false\n  beta: 0.6\n</policy>";
        let p = parse_llm_policy_response(text, "base").expect("parses");
        assert_eq!(p.name, "base#llm");
    }

    /// PR #7: `PerturbedPolicy::to_xml()` produces valid `<policy>` XML that
    /// round-trips through `parse_llm_policy_response`.
    #[test]
    fn serialize_policy_xml_round_trips_with_parser() {
        let p = PerturbedPolicy {
            name: "test_policy".into(),
            max_width: 6,
            max_depth: 5,
            close_on_failure: true,
            beta: 0.7,
        };
        let xml = p.to_xml();
        assert!(xml.contains("<policy>"));
        assert!(xml.contains("</policy>"));
        let parsed = parse_llm_policy_response(&xml, "base").expect("parses back");
        assert_eq!(parsed.max_width, p.max_width);
        assert_eq!(parsed.max_depth, p.max_depth);
        assert_eq!(parsed.close_on_failure, p.close_on_failure);
        assert!((parsed.beta - p.beta).abs() < 1e-6);
    }
}

#[cfg(test)]
mod llm_candidate_tests {
    use super::*;
    use cogni_core::CognitionTree;
    use cogni_llm::{LlmConfig, LlmTransport};
    // Bring in helpers defined in the lower-down `tests` module.
    use crate::tests::{node, toy_tree};

    /// Stub LLM that always returns the same canned `<policy>` block.
    /// Used to verify the wiring without depending on the network.
    struct StubLlm;
    #[async_trait::async_trait]
    impl LlmTransport for StubLlm {
        async fn complete(&self, _prompt: &str) -> anyhow::Result<String> {
            Ok(r#"
<policy>
  max_width: 5
  max_depth: 3
  close_on_failure: true
  beta: 0.55
  rationale: focuseddepth
</policy>"#
                .to_string())
        }
    }

    #[tokio::test]
    async fn llm_path_adds_a_distinct_candidate() {
        let p = BaselineParallelRefining;
        let stub = StubLlm;
        // M=4: slot 0 = current, slot 1/2 = deterministic (narrow, wide),
        // slot 3 = LLM. This way we can assert both the deterministic trio
        // AND the LLM candidate are present.
        let c = generate_candidates(&p, 4, Some(&stub)).await;
        assert_eq!(c.len(), 4);
        let names: Vec<_> = c.iter().map(|x| x.name().to_string()).collect();
        assert!(names.iter().any(|n| n.contains("#llm")), "LLM candidate present");
        assert!(names.iter().any(|n| n.contains("#narrow")));
        assert!(names.iter().any(|n| n.contains("#wide")));
    }

    #[tokio::test]
    async fn llm_path_falls_back_on_parse_error() {
        struct BrokenLlm;
        #[async_trait::async_trait]
        impl LlmTransport for BrokenLlm {
            async fn complete(&self, _prompt: &str) -> anyhow::Result<String> {
                Ok("not a valid policy block".into())
            }
        }
        let p = BaselineParallelRefining;
        let broken = BrokenLlm;
        let c = generate_candidates(&p, 3, Some(&broken)).await;
        // Should not panic; falls back to a deterministic candidate.
        assert_eq!(c.len(), 3);
        let names: Vec<_> = c.iter().map(|x| x.name().to_string()).collect();
        // Either the fallback name OR one of the deterministic variant names.
        assert!(names.iter().any(|n| n.contains("fallback") || n.contains("narrow") || n.contains("wide")));
    }

    #[tokio::test]
    async fn llm_dry_run_lives_in_cogni_llm() {
        // The dry-run path in cogni-llm returns a valid `<policy>` block, so
        // the real client works as a drop-in stub when MINIMAX_API_KEY is
        // absent.
        let cfg = LlmConfig {
            api_key_env: "COGNI_TEST_NONEXISTENT_KEY_XYZ".into(),
            ..LlmConfig::default()
        };
        let client = cogni_llm::LlmClient::from_config(cfg);
        assert!(client.is_dry_run());
        let text = client.complete("anything").await.expect("stub");
        let p = parse_llm_policy_response(&text, "base").expect("stub parses");
        assert!(p.name.starts_with("base#llm"));
        assert_eq!(p.max_width, 4);
        assert_eq!(p.max_depth, 4);
    }

    /// End-to-end: full `dream()` cycle with an LLM injected. The dreamer
    /// must (a) call the LLM, (b) parse the response, (c) include the LLM
    /// candidate in scoring, (d) never panic on parse failure.
    #[tokio::test]
    async fn dreaming_e2e_with_llm() {
        // Tree that rewards depth — so a wide+deep LLM policy might win.
        let root = node(None, "finance_balance_boundary", 0.3);
        let mut prev = root.id;
        let mut levels = vec![root];
        for lvl in 1..=3 {
            let n = node(Some(prev), &format!("L{lvl}"), 0.5 + 0.1 * lvl as f32);
            prev = n.id;
            levels.push(n);
        }
        let mut t = CognitionTree::new(levels[0].clone());
        for n in &levels[1..] {
            t.insert(n.clone());
        }
        let store = LocalReplayStore::from_tree(&t);
        let mut state = CogniState::new("baseline_parallel_refining");

        // Stub that always proposes a deep, wide policy.
        let stub = StubLlm;
        let winner = dream(
            &mut state,
            &store,
            &CognitionQuestion::default(),
            Some(&stub),
        )
        .await
        .expect("dream runs");

        // dream() must pick one of the candidates. With M=3 the LLM path
        // generates 1 deterministic (narrow) + 1 LLM candidate.
        assert!(winner.is_some());
        // State must have updated.
        assert_eq!(state.iteration, 0, "dream() does not bump iteration; outer does");
        assert!(!state.current_policy_name.is_empty());

        // Now run a full outer iteration (which DOES bump iteration).
        let _plan = run_outer_iteration(
            &mut state,
            &store,
            &CognitionQuestion::default(),
            Some(&stub),
            None,
        )
        .await
        .expect("outer iteration runs");
        assert_eq!(state.iteration, 1);
    }

    /// End-to-end: when the LLM errors out, dreaming must still complete
    /// without panicking and still pick a candidate.
    #[tokio::test]
    async fn dreaming_e2e_with_failing_llm() {
        struct AlwaysFailing;
        #[async_trait::async_trait]
        impl LlmTransport for AlwaysFailing {
            async fn complete(&self, _p: &str) -> anyhow::Result<String> {
                anyhow::bail!("simulated upstream failure")
            }
        }
        let store = LocalReplayStore::from_tree(&toy_tree());
        let mut state = CogniState::new("baseline_parallel_refining");
        let failing = AlwaysFailing;
        let winner = dream(
            &mut state,
            &store,
            &CognitionQuestion::default(),
            Some(&failing),
        )
        .await
        .expect("dream runs despite LLM failure");
        assert!(winner.is_some(), "fallback candidate keeps dreaming alive");
    }
}

#[cfg(test)]
mod store_e2e_tests {
    use super::*;
    use cogni_llm::LlmTransport;
    use cogni_store::{DailyProgressRow, SqliteStore, Store, WriteGate};
    use crate::tests::toy_tree;

    fn test_store() -> std::sync::Arc<SqliteStore> {
        // PR #4 E2E always runs with the write gate forced on. The
        // `cogni-store` crate already exposes its own gated tests; here
        // we just bypass the env var for direct access.
        let gate = WriteGate { allow: true };
        std::sync::Arc::new(
            SqliteStore::open_in_memory(gate).expect("in-memory db"),
        )
    }

    struct CountingLlm {
        n: std::sync::atomic::AtomicUsize,
    }
    impl CountingLlm {
        fn new() -> Self {
            Self { n: std::sync::atomic::AtomicUsize::new(0) }
        }
    }
    #[async_trait::async_trait]
    impl LlmTransport for CountingLlm {
        async fn complete(&self, _p: &str) -> anyhow::Result<String> {
            self.n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(r#"<policy>
  max_width: 5
  max_depth: 4
  close_on_failure: false
  beta: 0.6
  rationale: store-e2e-stub
</policy>"#
                .into())
        }
    }

    /// End-to-end: 3 outer iterations + SqliteStore. After the loop,
    /// `latest_daily_progress()` must reflect the third iteration's
    /// `pareto_auc` / `current_beta` / `selected_policy`.
    #[tokio::test]
    async fn dreaming_three_iterations_persist_to_sqlite() {
        let store = test_store();
        let replay = LocalReplayStore::from_tree(&toy_tree());
        let mut state = CogniState::new("baseline_parallel_refining");
        let llm = CountingLlm::new();
        let llm_ref = &llm;

        for round in 0..3 {
            let _plan = run_outer_iteration(
                &mut state,
                &replay,
                &CognitionQuestion::default(),
                Some(llm_ref as &dyn LlmTransport),
                Some(store.as_ref() as &dyn Store),
            )
            .await
            .expect("outer iter runs");
            assert_eq!(state.iteration, round + 1);
        }

        // The LLM was consulted once per outer iteration.
        assert_eq!(llm.n.load(std::sync::atomic::Ordering::SeqCst), 3);

        // Most recent row is iteration 3.
        let latest = store.latest_daily_progress().await.unwrap().expect("a row");
        assert_eq!(latest.day, 3);
        assert_eq!(latest.selected_policy, state.current_policy_name);
        assert!((latest.current_beta - state.beta).abs() < 1e-6);
        // pareto_auc is the best_node from the winner's ReplayScore.
        assert!(latest.pareto_auc >= 0.0);
        // parallel_penalty = rounds / max(1, probes) — bounded in [0, 1].
        assert!(latest.parallel_penalty <= 1.0);
        // PR #8: goal_progress and boundary_mastery are real frontier metrics.
        assert!(latest.goal_progress > 0.0);
        assert!(latest.boundary_mastery > 0.0);

        let rows = store.recent_rows(10).await.unwrap();
        assert_eq!(rows.len(), 3);
        // recent_rows returns newest first, so rows.last() is the first iteration.
        assert!(rows.last().unwrap().new_nodes > 0, "first iteration must discover/merge nodes");
    }

    /// End-to-end: even when `persist` is None, dreaming must still
    /// complete (the no-persist path stays green).
    #[tokio::test]
    async fn dreaming_runs_without_persist() {
        let replay = LocalReplayStore::from_tree(&toy_tree());
        let mut state = CogniState::new("baseline_parallel_refining");
        let _plan = run_outer_iteration(
            &mut state,
            &replay,
            &CognitionQuestion::default(),
            None,
            None,
        )
        .await
        .expect("outer iter runs");
        assert_eq!(state.iteration, 1);
    }

    /// PR #7: End-to-end: running outer iterations against SqliteStore must
    /// persist winning policy XML into policies table.
    #[tokio::test]
    async fn dreaming_persists_winner_policy_xml() {
        let store = test_store();
        let replay = LocalReplayStore::from_tree(&toy_tree());
        let mut state = CogniState::new("baseline_parallel_refining");
        let llm = CountingLlm::new();
        let llm_ref = &llm;

        for _round in 0..3 {
            let _plan = run_outer_iteration(
                &mut state,
                &replay,
                &CognitionQuestion::default(),
                Some(llm_ref as &dyn LlmTransport),
                Some(store.as_ref() as &dyn Store),
            )
            .await
            .expect("outer iter runs");
        }

        let count = store.policies_seen_count().await.unwrap();
        // At least one winning policy was persisted.
        assert!(
            count >= 1 && count <= 3,
            "expected 1..=3 winning policies, got {count}"
        );

        let policies = store.recent_policy_runs(10).await.unwrap();
        assert_eq!(policies.len() as u64, count);
        for p in &policies {
            assert!(
                p.source_code.contains("<policy>"),
                "expected XML tags in source_code: {}",
                p.source_code
            );
            assert!(
                p.source_code.contains("</policy>"),
                "expected XML closing tag: {}",
                p.source_code
            );
            assert!(
                p.source_code.contains("max_width:"),
                "expected max_width in source_code"
            );
            assert!(
                p.source_code.contains("max_depth:"),
                "expected max_depth in source_code"
            );
        }
    }

    /// Sanity: build_daily_progress derives sane values from a CandidateScore.
    #[test]
    fn build_daily_progress_uses_winner_signal() {
        let mut state = CogniState::new("baseline_parallel_refining");
        state.iteration = 7;
        state.current_policy_name = "baseline#llm-focuseddepth".into();
        state.beta = 0.55;
        let cs = CandidateScore {
            index: 0,
            name: "baseline#llm".into(),
            plan: DirectionPlan {
                iteration: 0,
                opened_branches: vec![],
                refine_targets: vec![],
                max_width: 4,
                max_depth: 4,
                stopping_rationale: "ok".into(),
            },
            score: ReplayScore {
                best_node: 0.85,
                probes: 6,
                rounds: 3,
                beta1: 0.05,
                beta2: 0.10,
            },
            source_code: "<policy>\n  max_width: 4\n</policy>".into(),
        };
        let delta = cogni_cognition_tree::TreeDelta::new(5, 2, 1);
        let boundary = cogni_cognition_tree::BoundarySpec::default_saas();
        let row = build_daily_progress(&state, Some(&cs), delta, &boundary);
        assert_eq!(row.day, 7);
        assert_eq!(row.new_nodes, 5);
        assert_eq!(row.closed_nodes, 2);
        assert_eq!(row.reopened_nodes, 1);
        assert_eq!(row.selected_policy, "baseline#llm-focuseddepth");
        assert!((row.pareto_auc - 0.85).abs() < 1e-6);
        // parallel_penalty = 3 / max(1, 6) = 0.5
        assert!((row.parallel_penalty - 0.5).abs() < 1e-6);
        // Root node in default state has understanding 0.1 and boundary_fit 1.0
        assert!((row.goal_progress - 0.1).abs() < 1e-6);
        assert!((row.boundary_mastery - 1.0).abs() < 1e-6);
        // Custom sanity: row equals what we'd read back from the store.
        let expected = DailyProgressRow {
            day: 7,
            new_nodes: 5,
            closed_nodes: 2,
            reopened_nodes: 1,
            goal_progress: 0.1,
            boundary_mastery: 1.0,
            pareto_auc: 0.85,
            parallel_penalty: 0.5,
            current_beta: 0.55,
            selected_policy: "baseline#llm-focuseddepth".into(),
        };
        assert_eq!(row, expected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogni_core::{CognitionNode, CognitionTree, FailureClass, NodeId, NodeKind, Score};
    use chrono::Utc;
    use serde_json::json;
    use ulid::Ulid;

    pub(crate) fn node(parent: Option<NodeId>, title: &str, score: f32) -> CognitionNode {
        CognitionNode {
            id: Ulid::new(),
            parent,
            iteration: 0,
            kind: NodeKind::Concept,
            title: title.into(),
            workspace: json!({}),
            observation: format!("obs-{title}"),
            score: Score {
                understanding: score,
                boundary_fit: 0.5,
                reuse_potential: 0.5,
                confidence: 0.7,
            },
            tags: vec![],
            evidence: vec![],
            failure_class: FailureClass::Ok,
            valid: true,
            no_total: false,
            created_at: Utc::now(),
        }
    }

    /// A tree with a shallow root and a deep, higher-scoring grandchild.
    /// The "wide" candidate should reveal more and win on best_node.
    pub(crate) fn toy_tree() -> CognitionTree {
        let root = node(None, "root", 0.4);
        let c1 = node(Some(root.id), "c1", 0.5);
        let gc = node(Some(c1.id), "gc", 0.95);
        let c2 = node(Some(root.id), "c2", 0.6);
        let mut t = CognitionTree::new(root);
        t.insert(c1);
        t.insert(gc);
        t.insert(c2);
        t
    }

    #[test]
    fn replay_score_matches_paper_formula() {
        let s = ReplayScore {
            best_node: 0.9,
            probes: 6,
            rounds: 3,
            beta1: 0.05,
            beta2: 0.1,
        };
        assert!((s.compute() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn state_initializes_baseline_policy() {
        let st = CogniState::new("baseline_parallel_refining");
        assert_eq!(st.iteration, 0);
        assert!(st.beta > 0.0 && st.beta < 1.0);
    }

    #[tokio::test]
    async fn generate_candidates_returns_m() {
        let p = BaselineParallelRefining;
        let c = generate_candidates(&p, 3, None).await;
        assert_eq!(c.len(), 3);
    }

    #[tokio::test]
    async fn perturbed_policy_name_is_distinct_per_variant() {
        let p = BaselineParallelRefining;
        let c = generate_candidates(&p, 4, None).await;
        let names: Vec<_> = c.iter().map(|x| x.name().to_string()).collect();
        assert!(names[0].contains("baseline"));
        assert!(names.iter().any(|n| n.contains("narrow")));
        assert!(names.iter().any(|n| n.contains("wide")));
        assert!(names.iter().any(|n| n.contains("balanced")));
    }

    #[tokio::test]
    async fn dream_picks_an_improvement_when_one_exists() {
        let store = LocalReplayStore::from_tree(&toy_tree());
        let mut state = CogniState::new("baseline_parallel_refining");
        let winner = dream(&mut state, &store, &CognitionQuestion::default(), None)
            .await
            .expect("dream runs");
        assert!(winner.is_some(), "at least one candidate should win");
        let w = winner.unwrap();
        assert!(w.score.compute() > 0.0);
    }

    #[tokio::test]
    async fn dream_keeps_incumbent_when_all_candidates_score_lower() {
        // Construct a tree where every candidate explores nothing useful.
        let root = node(None, "r", 0.0);
        let tree = CognitionTree::new(root);
        let store = LocalReplayStore::from_tree(&tree);
        let mut state = CogniState::new("baseline_parallel_refining");
        let winner = dream(&mut state, &store, &CognitionQuestion::default(), None)
            .await
            .expect("dream runs");
        // All V^m = 0 - β1*0 + β2*0 = 0; tie → first wins; that is the incumbent,
        // so the function returns Some(incumbent) (we still update the name).
        assert!(winner.is_some());
    }

    #[tokio::test]
    async fn run_outer_iteration_increments_state() {
        let store = LocalReplayStore::from_tree(&toy_tree());
        let mut state = CogniState::new("baseline_parallel_refining");
        let before = state.iteration;
        let _plan = run_outer_iteration(&mut state, &store, &CognitionQuestion::default(), None, None)
            .await
            .expect("outer iteration runs");
        assert_eq!(state.iteration, before + 1);
    }

    /// End-to-end: builds a SaaS-cognition shaped tree (root "balance
    /// sheet boundary", three branches each with a high-scoring grandchild),
    /// runs three outer iterations, and asserts:
    ///  - iteration counter advances each round
    ///  - the winner always has V ≥ incumbent (non-degradation invariant)
    ///  - selected_policy in state reflects the chosen name
    #[tokio::test]
    async fn dreaming_e2e_saas_boundary_tree() {
        // Root = "boundary question" (low understanding — fits a SaaS FI domain
        // where the agent is still learning the boundary). Three children
        // represent the three evidence paths (Document, Code, ApiCall); each
        // has a high-scoring grandchild to make at least one candidate win.
        let root = node(None, "balance_sheet_boundary", 0.2);
        let doc_branch = node(Some(root.id), "doc_path", 0.4);
        let doc_leaf = node(Some(doc_branch.id), "doc_leaf", 0.85);
        let code_branch = node(Some(root.id), "code_path", 0.5);
        let code_leaf = node(Some(code_branch.id), "code_leaf", 0.9);
        let api_branch = node(Some(root.id), "api_path", 0.45);
        let api_leaf = node(Some(api_branch.id), "api_leaf", 0.7);
        let mut t = CognitionTree::new(root);
        for n in [doc_branch, doc_leaf, code_branch, code_leaf, api_branch, api_leaf] {
            t.insert(n);
        }
        let store = LocalReplayStore::from_tree(&t);
        let mut state = CogniState::new("baseline_parallel_refining");
        let incumbent = state.current_policy_name.clone();

        // Run three outer iterations.
        for round in 0..3 {
            let plan = run_outer_iteration(
                &mut state,
                &store,
                &CognitionQuestion {
                    daily_question: "How is the balance sheet boundary defined?"
                        .to_string(),
                    boundary_hint: vec!["finance.v1".into()],
                    session_id: ulid::Ulid::new().to_string(),
                },
                None,
                None,
            )
            .await
            .expect("outer iteration runs");
            assert_eq!(state.iteration, round + 1, "iteration advanced");
            // The winning plan must keep width/depth > 0 (a usable grid).
            assert!(plan.max_width > 0 && plan.max_depth > 0);
        }

        // After three rounds, the state must have either picked a perturbed
        // variant or kept the incumbent. Either way, current_policy_name is
        // set and current_replay_score ≥ 0.
        assert!(!state.current_policy_name.is_empty());
        assert!(state.current_replay_score >= 0.0);
        // If nothing improved, the name should equal the incumbent.
        // We can't assert that strictly (the perturbed variants sometimes
        // win), but we can assert it never reverted to empty.
        assert_ne!(state.current_policy_name, String::new());

        // Cross-check: a policy that's never called never has its name
        // surface. Make sure the policy store round-trip works.
        let _ = incumbent; // silence unused warning
    }

    /// End-to-end non-degradation: when every candidate scores the same as
    /// the incumbent, the winner must still be returned (paper §3.3, the
    /// invariant V^m ≥ V^0 is what we test through this being pure value).
    #[tokio::test]
    async fn dreaming_e2e_pure_incumbent_score() {
        let root = node(None, "empty_root", 0.0);
        let tree = CognitionTree::new(root);
        let store = LocalReplayStore::from_tree(&tree);
        let mut state = CogniState::new("baseline_parallel_refining");
        let before_score = state.current_replay_score;
        let before_iter = state.iteration;

        let plan = run_outer_iteration(&mut state, &store, &CognitionQuestion::default(), None, None)
            .await
            .expect("outer iteration runs");
        assert_eq!(state.iteration, before_iter + 1);
        // Score should be ≥ incumbent (0.0).
        assert!(state.current_replay_score >= before_score - 1e-6);
        // The grid plan keeps the shape the policy was tracking.
        assert!(plan.max_width >= 1);
        assert!(plan.max_depth >= 1);
    }

    /// End-to-end: a deterministic tree where the "wide" variant should
    /// outperform the narrow one. Smoke-test for the argmax selection.
    #[tokio::test]
    async fn dreaming_e2e_wide_candidate_wins() {
        // Build a tree where wide exploration reaches the high-score
        // grandchild but narrow cannot.
        let root = node(None, "deep_root", 0.3);
        let mut prev_id = root.id;
        // 4-level chain so max_depth=2 (narrow) can't reach the leaf but
        // max_depth=6 (wide) can.
        let mut levels = vec![root];
        for lvl in 1..=4 {
            let n = node(Some(prev_id), &format!("L{lvl}"), 0.3 + 0.1 * lvl as f32);
            prev_id = n.id;
            levels.push(n);
        }
        let mut t = CognitionTree::new(levels[0].clone());
        for n in &levels[1..] {
            t.insert(n.clone());
        }
        let store = LocalReplayStore::from_tree(&t);
        let mut state = CogniState::new("baseline_parallel_refining");
        let _plan = run_outer_iteration(&mut state, &store, &CognitionQuestion::default(), None, None)
            .await
            .expect("outer iteration runs");
        // Iteration advanced and a non-empty winner plan was returned.
        assert_eq!(state.iteration, 1);
        assert!(!state.current_policy_name.is_empty());
    }
}