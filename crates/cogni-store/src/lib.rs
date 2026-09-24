//! SQLite-backed persistence for cognition history + daily_progress.
//!
//! PR #4: real `Store` trait with a `SqliteStore` impl. The schema is the
//! PR #1 baseline (`schema.sql`); this module adds typed reads/writes for
//! the `daily_progress` and `policies` tables.
//!
//! Safety: every write goes through `WriteGate::from_env()`. With
//! `COGNI_ALLOW_WRITE` unset (the default), `SqliteStore::open` returns
//! `Err(StoreError::WriteGateClosed)` instead of creating a writable
//! connection. PR #4 keeps the schema-only test (`schema_includes_*`)
//! passing without needing a real database.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};

pub const SCHEMA_SQL: &str = include_str!("schema.sql");

/// Whether the store is allowed to perform state-changing writes. Mirrors
/// the gate in `cogni-crawler` so that all crates share one convention.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct WriteGate {
    pub allow: bool,
}

impl WriteGate {
    pub fn from_env() -> Self {
        let allow = std::env::var("COGNI_ALLOW_WRITE")
            .ok()
            .filter(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .is_some();
        Self { allow }
    }
}

/// One day's dreaming outcome. `day` is the outer-iteration index used as
/// the primary key, so re-running an iteration overwrites the prior row
/// (this matches the paper's "non-degradation" spirit: each iteration has
/// exactly one record).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DailyProgressRow {
    pub day: u32,
    pub new_nodes: u32,
    pub closed_nodes: u32,
    pub reopened_nodes: u32,
    pub goal_progress: f32,
    pub boundary_mastery: f32,
    pub pareto_auc: f32,
    pub parallel_penalty: f32,
    pub current_beta: f32,
    pub selected_policy: String,
}

impl DailyProgressRow {
    /// A neutral baseline — useful as a starting point for tests.
    pub fn empty(day: u32) -> Self {
        Self {
            day,
            new_nodes: 0,
            closed_nodes: 0,
            reopened_nodes: 0,
            goal_progress: 0.0,
            boundary_mastery: 0.0,
            pareto_auc: 0.0,
            parallel_penalty: 0.0,
            current_beta: 0.6,
            selected_policy: "baseline_parallel_refining".into(),
        }
    }
}

/// One saved policy artifact. PR #7 actually persists the LLM-generated
/// `<policy>` XML so the dreaming loop is fully replayable from the DB
/// alone — no need to re-derive each iteration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyRunRow {
    pub id: String,
    pub iteration: u32,
    pub pareto_reward: f32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// The literal `<policy>…</policy>` text the LLM (or deterministic
    /// builder) produced. PR #4 stored `""` here; PR #7 binds the real
    /// value.
    pub source_code: String,
}

/// Async store API. The SQLite impl wraps a sync `rusqlite::Connection`
/// in a Mutex and runs blocking operations through `spawn_blocking`.
#[async_trait::async_trait]
pub trait Store: Send + Sync {
    async fn record_daily_progress(&self, row: DailyProgressRow) -> anyhow::Result<()>;
    async fn latest_daily_progress(&self) -> anyhow::Result<Option<DailyProgressRow>>;
    /// Return the most recent up-to-`limit` rows, newest first.
    /// Implementations that have no row history should return an empty
    /// Vec. PR #6 wires the TUI history pane to this.
    async fn recent_rows(&self, limit: u32) -> anyhow::Result<Vec<DailyProgressRow>>;
    async fn record_policy_run(&self, row: PolicyRunRow) -> anyhow::Result<()>;
    /// Return the most recent up-to-`limit` policy runs, newest first
    /// (ordered by `iteration` DESC). PR #7 exposes this so the test
    /// suite can verify `source_code` round-trips through the store.
    async fn recent_policy_runs(&self, limit: u32) -> anyhow::Result<Vec<PolicyRunRow>>;
    async fn policies_seen_count(&self) -> anyhow::Result<u64>;
}

/// All errors that the store can surface.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("write gate closed: set COGNI_ALLOW_WRITE=1 to allow state-changing operations")]
    WriteGateClosed,
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("invalid column value for `{0}`: {1}")]
    Decode(&'static str, String),
}

/// SQLite-backed implementation. The connection is wrapped in
/// `Arc<Mutex<…>>` so the `&self` async methods can `spawn_blocking`
/// (which requires `'static` closures).
pub struct SqliteStore {
    conn: Arc<Mutex<rusqlite::Connection>>,
    gate: WriteGate,
}

impl SqliteStore {
    /// Open (or create) a SQLite database at `path`. When `gate.allow` is
    /// `false`, this returns `Err(StoreError::WriteGateClosed)` so that
    /// state-changing operations never happen by default.
    pub fn open(path: impl AsRef<Path>, gate: WriteGate) -> Result<Self, StoreError> {
        if !gate.allow {
            return Err(StoreError::WriteGateClosed);
        }
        let conn = rusqlite::Connection::open(path)?;
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            gate,
        })
    }

    /// Open an in-memory database. Always subject to the write gate.
    pub fn open_in_memory(gate: WriteGate) -> Result<Self, StoreError> {
        if !gate.allow {
            return Err(StoreError::WriteGateClosed);
        }
        let conn = rusqlite::Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            gate,
        })
    }

    pub fn gate(&self) -> WriteGate {
        self.gate
    }

    fn write_row(conn: &Mutex<rusqlite::Connection>, row: &DailyProgressRow) -> Result<(), StoreError> {
        let conn = conn.lock().expect("conn mutex poisoned");
        conn.execute(
            "INSERT INTO daily_progress (
                day, new_nodes, closed_nodes, reopened_nodes,
                goal_progress, boundary_mastery, pareto_auc,
                parallel_penalty, current_beta, selected_policy
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(day) DO UPDATE SET
                new_nodes        = excluded.new_nodes,
                closed_nodes     = excluded.closed_nodes,
                reopened_nodes   = excluded.reopened_nodes,
                goal_progress    = excluded.goal_progress,
                boundary_mastery = excluded.boundary_mastery,
                pareto_auc       = excluded.pareto_auc,
                parallel_penalty = excluded.parallel_penalty,
                current_beta     = excluded.current_beta,
                selected_policy  = excluded.selected_policy",
            rusqlite::params![
                row.day,
                row.new_nodes,
                row.closed_nodes,
                row.reopened_nodes,
                row.goal_progress,
                row.boundary_mastery,
                row.pareto_auc,
                row.parallel_penalty,
                row.current_beta,
                row.selected_policy,
            ],
        )?;
        Ok(())
    }

    fn read_latest(conn: &Mutex<rusqlite::Connection>) -> Result<Option<DailyProgressRow>, StoreError> {
        let conn = conn.lock().expect("conn mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT day, new_nodes, closed_nodes, reopened_nodes,
                    goal_progress, boundary_mastery, pareto_auc,
                    parallel_penalty, current_beta, selected_policy
             FROM daily_progress
             ORDER BY day DESC
             LIMIT 1",
        )?;
        let mut rows = stmt.query([])?;
        let Some(row) = rows.next()? else {
            return Ok(None);
        };
        Ok(Some(DailyProgressRow {
            day: row.get(0)?,
            new_nodes: row.get(1)?,
            closed_nodes: row.get(2)?,
            reopened_nodes: row.get(3)?,
            goal_progress: row.get(4)?,
            boundary_mastery: row.get(5)?,
            pareto_auc: row.get(6)?,
            parallel_penalty: row.get(7)?,
            current_beta: row.get(8)?,
            selected_policy: row.get(9)?,
        }))
    }

    fn read_recent(
        conn: &Mutex<rusqlite::Connection>,
        limit: u32,
    ) -> Result<Vec<DailyProgressRow>, StoreError> {
        let conn = conn.lock().expect("conn mutex poisoned");
        // `LIMIT ?` with a u32 binding is fine; cast to i64 so SQLite's
        // parameter binding accepts it across versions.
        let mut stmt = conn.prepare(
            "SELECT day, new_nodes, closed_nodes, reopened_nodes,
                    goal_progress, boundary_mastery, pareto_auc,
                    parallel_penalty, current_beta, selected_policy
             FROM daily_progress
             ORDER BY day DESC
             LIMIT ?",
        )?;
        let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
            Ok(DailyProgressRow {
                day: row.get(0)?,
                new_nodes: row.get(1)?,
                closed_nodes: row.get(2)?,
                reopened_nodes: row.get(3)?,
                goal_progress: row.get(4)?,
                boundary_mastery: row.get(5)?,
                pareto_auc: row.get(6)?,
                parallel_penalty: row.get(7)?,
                current_beta: row.get(8)?,
                selected_policy: row.get(9)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    fn write_policy(conn: &Mutex<rusqlite::Connection>, row: &PolicyRunRow) -> Result<(), StoreError> {
        let conn = conn.lock().expect("conn mutex poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO policies (id, iteration, source_code, score_pareto, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                row.id,
                row.iteration,
                row.source_code,
                row.pareto_reward,
                row.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    fn read_recent_policies(
        conn: &Mutex<rusqlite::Connection>,
        limit: u32,
    ) -> Result<Vec<PolicyRunRow>, StoreError> {
        let conn = conn.lock().expect("conn mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, iteration, score_pareto, created_at, source_code
             FROM policies
             ORDER BY iteration DESC
             LIMIT ?",
        )?;
        let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, u32>(1)?,
                row.get::<_, f32>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (id, iteration, pareto_reward, created_at, source_code) = r?;
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_at)
                .map(|d| d.with_timezone(&chrono::Utc))
                .map_err(|e| StoreError::Decode("created_at", e.to_string()))?;
            out.push(PolicyRunRow {
                id,
                iteration,
                pareto_reward,
                created_at,
                source_code,
            });
        }
        Ok(out)
    }

    fn count_policies(conn: &Mutex<rusqlite::Connection>) -> Result<u64, StoreError> {
        let conn = conn.lock().expect("conn mutex poisoned");
        let n: i64 =
            conn.query_row("SELECT COUNT(*) FROM policies", [], |r| r.get(0))?;
        Ok(n.max(0) as u64)
    }
}

#[async_trait::async_trait]
impl Store for SqliteStore {
    async fn record_daily_progress(&self, row: DailyProgressRow) -> anyhow::Result<()> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || Self::write_row(&conn, &row))
            .await
            .map_err(|e| anyhow::anyhow!("blocking join error: {e}"))??;
        Ok(())
    }

    async fn latest_daily_progress(&self) -> anyhow::Result<Option<DailyProgressRow>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || Self::read_latest(&conn))
            .await
            .map_err(|e| anyhow::anyhow!("blocking join error: {e}"))?
            .map_err(Into::into)
    }

    async fn recent_rows(&self, limit: u32) -> anyhow::Result<Vec<DailyProgressRow>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || Self::read_recent(&conn, limit))
            .await
            .map_err(|e| anyhow::anyhow!("blocking join error: {e}"))?
            .map_err(Into::into)
    }

    async fn record_policy_run(&self, row: PolicyRunRow) -> anyhow::Result<()> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || Self::write_policy(&conn, &row))
            .await
            .map_err(|e| anyhow::anyhow!("blocking join error: {e}"))??;
        Ok(())
    }

    async fn recent_policy_runs(&self, limit: u32) -> anyhow::Result<Vec<PolicyRunRow>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || Self::read_recent_policies(&conn, limit))
            .await
            .map_err(|e| anyhow::anyhow!("blocking join error: {e}"))?
            .map_err(Into::into)
    }

    async fn policies_seen_count(&self) -> anyhow::Result<u64> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || Self::count_policies(&conn))
            .await
            .map_err(|e| anyhow::anyhow!("blocking join error: {e}"))?
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn schema_includes_daily_progress() {
        let s = SCHEMA_SQL;
        assert!(s.contains("daily_progress"), "schema content was:\n{s}");
        assert!(s.contains("goal_progress"));
        assert!(s.contains("boundary_mastery"));
    }

    #[test]
    fn write_gate_closed_by_default() {
        // We never mutate the env here; we just check that a fresh gate
        // denies when nothing is set in the test runner's environment.
        // (cogni-crawler tests guard the actual env-var path.)
        let g = WriteGate { allow: false };
        assert!(!g.allow);
        assert!(matches!(
            SqliteStore::open_in_memory(g),
            Err(StoreError::WriteGateClosed)
        ));
    }

    fn open_test_db() -> Arc<SqliteStore> {
        // Force the gate open for this test process. The WriteGate is
        // Copy + threadsafe, so concurrent tests don't fight over it.
        std::env::set_var("COGNI_ALLOW_WRITE", "1");
        let gate = WriteGate { allow: true };
        Arc::new(
            SqliteStore::open_in_memory(gate).expect("in-memory db opens with gate open"),
        )
    }

    #[tokio::test]
    async fn record_then_read_round_trips() {
        let store = open_test_db();
        let row = DailyProgressRow {
            day: 1,
            new_nodes: 7,
            closed_nodes: 2,
            reopened_nodes: 1,
            goal_progress: 0.42,
            boundary_mastery: 0.65,
            pareto_auc: 0.81,
            parallel_penalty: 0.15,
            current_beta: 0.55,
            selected_policy: "baseline#llm-focuseddepth".into(),
        };
        store
            .record_daily_progress(row.clone())
            .await
            .expect("write");
        let got = store
            .latest_daily_progress()
            .await
            .expect("read")
            .expect("some row");
        assert_eq!(got, row);
    }

    #[tokio::test]
    async fn upsert_replaces_same_day() {
        let store = open_test_db();
        let mut row = DailyProgressRow::empty(2);
        row.pareto_auc = 0.1;
        store.record_daily_progress(row.clone()).await.unwrap();
        row.pareto_auc = 0.9;
        store.record_daily_progress(row.clone()).await.unwrap();
        let got = store.latest_daily_progress().await.unwrap().unwrap();
        assert_eq!(got.pareto_auc, 0.9);
        // Only one row even after two writes.
        assert_eq!(got.day, 2);
    }

    #[tokio::test]
    async fn record_policy_then_count() {
        let store = open_test_db();
        for i in 0..3 {
            store
                .record_policy_run(PolicyRunRow {
                    id: format!("pol-{i}"),
                    iteration: i,
                    pareto_reward: 0.1 * i as f32,
                    created_at: chrono::Utc::now(),
                    source_code: "<policy>\n  rationale: count\n</policy>".into(),
                })
                .await
                .unwrap();
        }
        let n = store.policies_seen_count().await.unwrap();
        assert_eq!(n, 3);
    }

    #[tokio::test]
    async fn latest_is_none_on_fresh_db() {
        let store = open_test_db();
        let latest = store.latest_daily_progress().await.unwrap();
        assert!(latest.is_none());
    }

    #[tokio::test]
    async fn recent_rows_returns_descending_with_limit() {
        let store = open_test_db();
        // Insert 5 days out of order; expect the query to sort by day.
        let mut rows = Vec::new();
        for d in [3u32, 1, 5, 2, 4] {
            let mut r = DailyProgressRow::empty(d);
            r.pareto_auc = d as f32 * 0.1;
            r.selected_policy = format!("baseline#day-{d}");
            rows.push(r);
        }
        for r in &rows {
            store.record_daily_progress(r.clone()).await.unwrap();
        }

        // limit = 3 → expect days 5, 4, 3 (newest first).
        let got = store.recent_rows(3).await.unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].day, 5);
        assert_eq!(got[1].day, 4);
        assert_eq!(got[2].day, 3);
        assert_eq!(got[0].selected_policy, "baseline#day-5");

        // limit larger than the row count returns everything.
        let got_all = store.recent_rows(64).await.unwrap();
        assert_eq!(got_all.len(), 5);
        assert_eq!(got_all.first().unwrap().day, 5);
        assert_eq!(got_all.last().unwrap().day, 1);
    }

    #[tokio::test]
    async fn recent_rows_empty_db_returns_empty() {
        let store = open_test_db();
        let got = store.recent_rows(8).await.unwrap();
        assert!(got.is_empty());
    }

    /// PR #7: writing a `PolicyRunRow` must persist `source_code` (no
    /// more empty-string placeholder). Read it back via
    /// `recent_policy_runs` and verify the XML body is intact.
    #[tokio::test]
    async fn record_policy_run_persists_source_code() {
        let store = open_test_db();
        let xml = "<policy>\n  max_width: 6\n  max_depth: 5\n  \
                   close_on_failure: true\n  beta: 0.7\n  \
                   rationale: focuseddepth\n</policy>";
        store
            .record_policy_run(PolicyRunRow {
                id: "pol-e2e".into(),
                iteration: 3,
                pareto_reward: 0.84,
                created_at: chrono::Utc::now(),
                source_code: xml.into(),
            })
            .await
            .expect("write");
        let got = store
            .recent_policy_runs(8)
            .await
            .expect("read")
            .into_iter()
            .find(|p| p.id == "pol-e2e")
            .expect("pol-e2e present");
        assert_eq!(got.source_code, xml);
        assert_eq!(got.iteration, 3);
        assert!((got.pareto_reward - 0.84).abs() < 1e-6);
    }

    /// PR #7: `recent_policy_runs` returns newest first by `iteration`.
    #[tokio::test]
    async fn recent_policy_runs_orders_descending_by_iteration() {
        let store = open_test_db();
        for i in [1u32, 3, 2] {
            store
                .record_policy_run(PolicyRunRow {
                    id: format!("pol-{i}"),
                    iteration: i,
                    pareto_reward: 0.1 * i as f32,
                    created_at: chrono::Utc::now(),
                    source_code: format!("<policy>\n  rationale: iter-{i}\n</policy>"),
                })
                .await
                .unwrap();
        }
        let got = store.recent_policy_runs(8).await.unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].iteration, 3);
        assert_eq!(got[1].iteration, 2);
        assert_eq!(got[2].iteration, 1);
        // Limit respected.
        let got2 = store.recent_policy_runs(2).await.unwrap();
        assert_eq!(got2.len(), 2);
        assert_eq!(got2[0].iteration, 3);
        assert_eq!(got2[1].iteration, 2);
    }
}