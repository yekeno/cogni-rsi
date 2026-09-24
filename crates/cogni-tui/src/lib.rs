//! Ratatui-based terminal UI for the thinking loop.
//!
//! PR #5 split the crate into two layers:
//!  * the **block layer** (`Block` + `render_block`) — pure-string renderer
//!    for the four §3.1 dialogue blocks; used by tests and `cogni-cli show`.
//!  * the **app layer** (`app` module) — a stateful `App` struct that holds
//!    `cogni_store::DailyProgressRow`s and renders a multi-pane ratatui
//!    frame.
//!
//! The block layer is unchanged so existing tests stay green.

pub mod app;

use cogni_core::DailyProgress;

/// Block kinds the TUI emits. Mirrors §3.1.
#[derive(Debug, Clone)]
pub enum Block {
    DayHeader(DailyProgress),
    /// A single `[ k ]` observation block.
    Obs {
        index: u32,
        title: String,
        score: (f32, f32),
        evidence: String,
        child_link: Option<String>,
    },
    /// A `[ thinking ]` block summarising the dreaming step.
    Dreaming {
        candidates: u32,
        winner: usize,
        pareto_reward: f32,
        beta: f32,
    },
    /// A `[ stop ]` block with three boolean reasons.
    Stop {
        no_frontier: bool,
        no_root: bool,
        budget_exhausted: bool,
    },
}

pub fn render_block(b: &Block) -> String {
    match b {
        Block::DayHeader(d) => format!(
            "──────────\n[ day {:03} ]  t = {}   β = {:.2}\n──────────\n\
             goal_progress {:.2}   boundary_mastery {:.2}\n\
             daily_delta   +{} / -{} / reopen {}\n",
            d.day,
            d.day,
            d.current_beta,
            d.goal_progress,
            d.boundary_mastery,
            d.new_nodes,
            d.closed_nodes,
            d.reopened_nodes,
        ),
        Block::Obs {
            index,
            title,
            score,
            evidence,
            child_link,
        } => {
            let link = child_link
                .as_deref()
                .map(|s| format!("\n   ↳ next: {s}"))
                .unwrap_or_default();
            format!(
                "[ {index:02} ] {title}  → ({:.2}, {:.2})\n   ↳ evidence: {evidence}{link}",
                score.0, score.1
            )
        }
        Block::Dreaming {
            candidates,
            winner,
            pareto_reward,
            beta,
        } => format!(
            "[ thinking ] evaluated {} policies; winner = m={winner}, pareto={:.3}, β={:.2}",
            candidates, pareto_reward, beta
        ),
        Block::Stop {
            no_frontier,
            no_root,
            budget_exhausted,
        } => format!(
            "[ stop ] no_frontier={no_frontier}  no_root={no_root}  budget_exhausted={budget_exhausted}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_header_includes_required_fields() {
        let p = DailyProgress {
            day: 7,
            new_nodes: 3,
            closed_nodes: 1,
            reopened_nodes: 0,
            goal_progress: 0.61,
            boundary_mastery: 0.74,
            pareto_auc: 0.83,
            parallel_penalty: 0.3,
            current_beta: 0.62,
            selected_policy: "p2".into(),
        };
        let s = render_block(&Block::DayHeader(p));
        assert!(s.contains("[ day 007 ]"));
        assert!(s.contains("goal_progress"));
        assert!(s.contains("β = 0.62"));
    }

    #[test]
    fn stop_block_lists_three_conditions() {
        let s = render_block(&Block::Stop {
            no_frontier: true,
            no_root: false,
            budget_exhausted: false,
        });
        assert!(s.contains("no_frontier=true"));
        assert!(s.contains("no_root=false"));
    }
}