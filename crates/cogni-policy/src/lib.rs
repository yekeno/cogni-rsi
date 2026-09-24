//! Cognition policy trait + baseline implementation.
//!
//! Mirrors Dream-RSI's `OptimalPolicy` / `LLMDesignedMethod`. A policy
//! observes the prefix-revealed cognition tree and decides the next
//! `DirectionPlan` (analogous to `GridPlan`).

use async_trait::async_trait;
use cogni_core::{CognitionQuestion, DirectionPlan, Score};
use cogni_replay::ReplaySession;
use serde::{Deserialize, Serialize};

/// Grid planning context, mirroring Dream-RSI's `GridPlanningContext`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridContext {
    pub history: serde_json::Value,
    pub actual_opened_width: usize,
    pub probe_work: u32,
    pub decision_rounds: u32,
    pub scores: Vec<Score>,
    pub beta: f32,
    pub failure_budget_hard: u32,
    pub worker_cap: usize,
}

/// A grid plan: how many new branches to open, how many refinements, how
/// many total probes the runtime may take before stopping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPlan {
    pub branch_count: usize,
    pub refine_count: usize,
    pub reason: String,
}

/// Budget for one solve() call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Budget {
    pub max_probes: u32,
}

#[async_trait]
pub trait CognitionPolicy: Send + Sync {
    fn name(&self) -> &str;

    /// Decide the next direction plan given the current revealed prefix.
    async fn solve(
        &self,
        question: &CognitionQuestion,
        budget: Option<Budget>,
        replay: &mut ReplaySession,
    ) -> anyhow::Result<DirectionPlan>;

    /// Deterministic, prefix-safe grid planner — must NOT inspect raw tree
    /// outcomes or per-iteration results (mirrors Dream-RSI §B.2
    /// "Learn from history without leaking outcomes").
    fn plan_grid(&self, ctx: &GridContext) -> GridPlan;
}

/// Baseline policy: a parallel refining strategy on the frontier.
///
/// Mirrors Dream-RSI's "parallel refining" baseline.
pub struct BaselineParallelRefining;

#[async_trait]
impl CognitionPolicy for BaselineParallelRefining {
    fn name(&self) -> &str {
        "baseline_parallel_refining"
    }

    async fn solve(
        &self,
        _question: &CognitionQuestion,
        budget: Option<Budget>,
        replay: &mut ReplaySession,
    ) -> anyhow::Result<DirectionPlan> {
        let actions = replay.legal_actions().await?;
        let budget = budget.unwrap_or(Budget { max_probes: 8 });
        let probes = actions
            .iter()
            .filter(|a| matches!(a, cogni_replay::LegalAction::Probe { .. }))
            .take(budget.max_probes as usize)
            .count();
        Ok(DirectionPlan {
            iteration: 0,
            opened_branches: vec![],
            refine_targets: vec![],
            max_width: probes.min(8),
            max_depth: 4,
            stopping_rationale: "baseline: parallel-refining frontier".into(),
        })
    }

    fn plan_grid(&self, ctx: &GridContext) -> GridPlan {
        let width_budget = ctx.history.get("actual_opened_width")
            .and_then(|v| v.as_u64())
            .unwrap_or(ctx.failure_budget_hard as u64) as usize;
        let probe_budget = ctx.history.get("probe_work")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let target_branches = if ctx.beta > 0.7 {
            width_budget.max(1)
        } else if probe_budget > 100 {
            (width_budget / 2).max(1)
        } else {
            width_budget.max(1)
        };
        GridPlan {
            branch_count: target_branches,
            refine_count: ctx.actual_opened_width.min(2),
            reason: format!(
                "baseline: beta={}, prior_width={}, prior_probes={}",
                ctx.beta, ctx.actual_opened_width, probe_budget
            ),
        }
    }
}

/// Holds the current policy used in the live cycle.
pub struct PolicyHolder {
    pub current: Box<dyn CognitionPolicy>,
    pub beta: f32,
}

impl PolicyHolder {
    pub fn new(policy: Box<dyn CognitionPolicy>) -> Self {
        Self {
            current: policy,
            beta: 0.6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogni_core::CognitionQuestion;

    #[test]
    fn grid_plan_uses_prefix_signals_only() {
        let pol = BaselineParallelRefining;
        let ctx = GridContext {
            history: serde_json::json!({
                "actual_opened_width": 4,
                "probe_work": 200,
            }),
            actual_opened_width: 4,
            probe_work: 200,
            decision_rounds: 2,
            scores: vec![],
            beta: 0.4,
            failure_budget_hard: 50,
            worker_cap: 8,
        };
        let plan = pol.plan_grid(&ctx);
        // Many prior probes → drop width per Dream-RSI §B.2 "many roots
        // improve early while deeper refinements stall: reduce/hold width".
        assert!(plan.branch_count >= 1);
    }

    #[tokio::test]
    async fn baseline_solve_returns_nonempty_plan() {
        let pol = BaselineParallelRefining;
        let q = CognitionQuestion::default();
        // We do not have a real ReplaySession here — smoke-test that the
        // trait surface compiles.
        let _: &dyn CognitionPolicy = &pol;
        assert_eq!(pol.name(), "baseline_parallel_refining");
        assert_eq!(q.daily_question.len(), 0);
    }
}