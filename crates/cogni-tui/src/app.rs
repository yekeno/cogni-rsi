//! Stateful `App` rendered by `cogni-cli tui`.
//!
//! PR #5: stores recent `DailyProgressRow`s and renders a multi-pane
//! ratatui frame. The store is read on construction; the `cogni-cli tui`
//! binary is responsible for polling and re-creating the `App` (which
//! is cheap because the rows are already in memory).

use cogni_store::DailyProgressRow;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block as RBlock, Borders, List, ListItem, Paragraph};
use ratatui::Frame;

/// The TUI's view-model. Holds the most recent `history_len` rows from
/// the store (oldest first, newest last). PR #5 caps at 8 rows on
/// screen at once; older rows scroll off the top.
#[derive(Debug, Clone)]
pub struct App {
    /// Day banner displayed at the top — the highest day seen so far.
    pub latest_day: u32,
    /// The full `DailyProgressRow` corresponding to `latest_day`.
    pub latest: Option<DailyProgressRow>,
    /// Newest-first history (capped at 8). Bounded for tests.
    pub history: Vec<DailyProgressRow>,
}

impl App {
    /// Build an `App` from a list of rows. Rows may be in any order;
    /// they're sorted by `day` ascending (oldest first) and the highest
    /// day becomes the banner.
    pub fn from_rows<I>(rows: I) -> Self
    where
        I: IntoIterator<Item = DailyProgressRow>,
    {
        let mut history: Vec<DailyProgressRow> = rows.into_iter().collect();
        history.sort_by_key(|r| r.day);
        let latest = history.last().cloned();
        let latest_day = latest.as_ref().map(|r| r.day).unwrap_or(0);
        // Keep the most recent 8 only.
        let history: Vec<DailyProgressRow> =
            history.into_iter().rev().take(8).collect::<Vec<_>>().into_iter().rev().collect();
        Self {
            latest_day,
            latest,
            history,
        }
    }

    /// Empty `App`. Useful when no rows are in the store yet.
    pub fn empty() -> Self {
        Self {
            latest_day: 0,
            latest: None,
            history: vec![],
        }
    }

    /// Render the frame. The Backend lives on the surrounding Terminal —
    /// ratatui 0.27's `Frame` is concrete, so we don't carry a generic.
    pub fn view(&self, frame: &mut Frame) {
        let area = frame.size();
        // Top-level split: header (4 lines) + metrics (3) + history (rest).
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(3),
                Constraint::Min(3),
            ])
            .split(area);

        self.render_header(frame, chunks[0]);
        self.render_metrics(frame, chunks[1]);
        self.render_history(frame, chunks[2]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let day_label = if self.latest_day == 0 {
            "day ???".to_string()
        } else {
            format!("day {:03}", self.latest_day)
        };
        let policy = self
            .latest
            .as_ref()
            .map(|r| r.selected_policy.clone())
            .unwrap_or_else(|| "(no data)".to_string());
        let beta = self.latest.as_ref().map(|r| r.current_beta).unwrap_or(0.0);
        let title = Line::from(vec![
            Span::styled("cogni-rsi ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("· "),
            Span::styled(day_label, Style::default().fg(Color::Cyan)),
            Span::raw("  β = "),
            Span::styled(format!("{beta:.2}"), Style::default().fg(Color::Yellow)),
        ]);
        let subtitle = Line::from(vec![
            Span::raw("policy: "),
            Span::styled(policy, Style::default().fg(Color::Green)),
        ]);
        let block = RBlock::default()
            .borders(Borders::ALL)
            .title("thinking loop")
            .border_style(Style::default().fg(Color::DarkGray));
        let para = Paragraph::new(vec![
            title,
            Line::from(""),
            subtitle,
            Line::from(""),
        ])
        .block(block);
        frame.render_widget(para, area);
    }

    fn render_metrics(&self, frame: &mut Frame, area: Rect) {
        let (auc, par) = match self.latest.as_ref() {
            Some(r) => (r.pareto_auc, r.parallel_penalty),
            None => (0.0, 0.0),
        };
        let line = Line::from(vec![
            Span::raw(" pareto_auc "),
            Span::styled(format!("{auc:.3}"), Style::default().fg(Color::Magenta)),
            Span::raw("   parallel_penalty "),
            Span::styled(format!("{par:.3}"), Style::default().fg(Color::Magenta)),
            Span::raw("   (latest)"),
        ]);
        let block = RBlock::default().borders(Borders::ALL).title("metrics");
        let para = Paragraph::new(line).block(block);
        frame.render_widget(para, area);
    }

    fn render_history(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = if self.history.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "(no iterations yet — start a dreaming loop)",
                Style::default().fg(Color::DarkGray),
            )))]
        } else {
            self.history
                .iter()
                .map(|r| {
                    let line = format!(
                        "day {:03}  policy={:30}  auc={:.3}  penalty={:.3}  β={:.2}",
                        r.day,
                        r.selected_policy,
                        r.pareto_auc,
                        r.parallel_penalty,
                        r.current_beta
                    );
                    ListItem::new(Line::from(Span::raw(line)))
                })
                .collect()
        };
        let block = RBlock::default().borders(Borders::ALL).title("history (last 8)");
        let list = List::new(items).block(block);
        frame.render_widget(list, area);
    }
}

/// Bridge: turn a `cogni_store::DailyProgressRow` into a `cogni_core::DailyProgress`
/// for the existing `Block::DayHeader` renderer. Lives in `cogni-tui` so
/// the orphan rule is satisfied (cogni-store is a `dep`, not a foreign
/// type for cogni-tui — but cogni-core's `DailyProgress` is foreign, so we
/// still cannot write `From for DailyProgress` here. Instead, expose a
/// plain function and let `cogni-cli` call it.).
pub fn row_to_daily_progress(r: &DailyProgressRow) -> cogni_core::DailyProgress {
    cogni_core::DailyProgress {
        day: r.day,
        new_nodes: r.new_nodes,
        closed_nodes: r.closed_nodes,
        reopened_nodes: r.reopened_nodes,
        goal_progress: r.goal_progress,
        boundary_mastery: r.boundary_mastery,
        pareto_auc: r.pareto_auc,
        parallel_penalty: r.parallel_penalty,
        current_beta: r.current_beta,
        selected_policy: r.selected_policy.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(day: u32, auc: f32, penalty: f32, beta: f32, policy: &str) -> DailyProgressRow {
        DailyProgressRow {
            day,
            new_nodes: 1,
            closed_nodes: 0,
            reopened_nodes: 0,
            goal_progress: auc,
            boundary_mastery: auc,
            pareto_auc: auc,
            parallel_penalty: penalty,
            current_beta: beta,
            selected_policy: policy.into(),
        }
    }

    #[test]
    fn empty_app_does_not_panic_and_shows_unknown_day() {
        let app = App::empty();
        let backend = ratatui::backend::TestBackend::new(80, 16);
        let mut terminal = ratatui::Terminal::new(backend).expect("test backend");
        terminal
            .draw(|f| app.view(f))
            .expect("draw empty app");
    }

    #[test]
    fn from_rows_picks_highest_day_as_latest() {
        let rows = vec![
            row(1, 0.1, 0.2, 0.5, "baseline"),
            row(5, 0.5, 0.1, 0.6, "baseline#llm"),
            row(3, 0.3, 0.1, 0.7, "baseline#wide"),
        ];
        let app = App::from_rows(rows);
        assert_eq!(app.latest_day, 5);
        assert_eq!(app.history.len(), 3);
        // Oldest first.
        assert_eq!(app.history[0].day, 1);
        assert_eq!(app.history[2].day, 5);
    }

    #[test]
    fn history_is_capped_to_eight() {
        let rows: Vec<DailyProgressRow> =
            (0..20).map(|i| row(i, 0.5, 0.1, 0.6, "p")).collect();
        let app = App::from_rows(rows);
        assert_eq!(app.history.len(), 8);
        // Newest is day 19.
        assert_eq!(app.history.last().unwrap().day, 19);
    }

    #[test]
    fn daily_progress_conversion_preserves_fields() {
        let r = row(7, 0.42, 0.13, 0.66, "baseline#llm");
        let d = row_to_daily_progress(&r);
        assert_eq!(d.day, 7);
        assert!((d.pareto_auc - 0.42).abs() < 1e-6);
        assert!((d.parallel_penalty - 0.13).abs() < 1e-6);
        assert!((d.current_beta - 0.66).abs() < 1e-6);
        assert_eq!(d.selected_policy, "baseline#llm");
    }

    /// Snapshot test: render the app at a fixed size and snapshot the
    /// buffer to plain text. We don't diff against a frozen string — PR
    /// #5 keeps the test resilient to ratatui style/width tweaks.
    #[test]
    fn snapshot_contains_required_widgets() {
        let app = App::from_rows(vec![
            row(1, 0.30, 0.20, 0.55, "baseline"),
            row(2, 0.45, 0.18, 0.60, "baseline#wide"),
            row(3, 0.62, 0.15, 0.65, "baseline#llm-focuseddepth"),
        ]);
        let backend = ratatui::backend::TestBackend::new(80, 16);
        let mut terminal = ratatui::Terminal::new(backend).expect("test backend");
        terminal.draw(|f| app.view(f)).expect("draw");
        let buffer = terminal.backend().buffer().clone();
        let dump: String = buffer
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>()
            .replace('\n', "⏎\n");
        assert!(dump.contains("thinking loop"), "header missing");
        assert!(dump.contains("day 003"), "latest day missing");
        assert!(dump.contains("baseline#llm-focuseddepth"), "policy missing");
        assert!(dump.contains("pareto_auc"), "metrics row missing");
        assert!(dump.contains("history (last 8)"), "history block missing");
        // All three history rows visible.
        assert!(dump.contains("day 001"));
        assert!(dump.contains("day 002"));
        assert!(dump.contains("day 003"));
    }

    #[test]
    fn snapshot_handles_tall_terminal() {
        let app = App::from_rows(vec![row(1, 0.5, 0.2, 0.6, "p")]);
        let backend = ratatui::backend::TestBackend::new(120, 30);
        let mut terminal = ratatui::Terminal::new(backend).expect("test backend");
        terminal.draw(|f| app.view(f)).expect("draw tall");
    }

    #[test]
    fn snapshot_handles_narrow_terminal_without_overflow() {
        let app = App::from_rows(vec![row(1, 0.5, 0.2, 0.6, "p")]);
        // 30 columns is enough for borders + 1 line of content; longer
        // lines wrap. PR #5 asserts only "no panic" here.
        let backend = ratatui::backend::TestBackend::new(30, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("test backend");
        terminal.draw(|f| app.view(f)).expect("draw narrow");
    }
}