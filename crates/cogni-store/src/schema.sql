-- cogni-store schema (PR #1 baseline; PR #2+ will add embedding tables)

CREATE TABLE IF NOT EXISTS cognition_nodes (
    id              TEXT PRIMARY KEY,
    parent          TEXT,
    iteration       INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    title           TEXT NOT NULL,
    workspace_json  TEXT NOT NULL,
    observation     TEXT NOT NULL,
    score_json      TEXT NOT NULL,
    tags_json       TEXT NOT NULL DEFAULT '[]',
    evidence_json   TEXT NOT NULL DEFAULT '[]',
    failure_class   TEXT NOT NULL,
    valid           INTEGER NOT NULL,
    no_total        INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_parent ON cognition_nodes(parent);
CREATE INDEX IF NOT EXISTS idx_nodes_iter   ON cognition_nodes(iteration);

CREATE TABLE IF NOT EXISTS daily_progress (
    day              INTEGER PRIMARY KEY,
    new_nodes        INTEGER NOT NULL DEFAULT 0,
    closed_nodes     INTEGER NOT NULL DEFAULT 0,
    reopened_nodes   INTEGER NOT NULL DEFAULT 0,
    goal_progress    REAL    NOT NULL DEFAULT 0.0,
    boundary_mastery REAL    NOT NULL DEFAULT 0.0,
    pareto_auc       REAL    NOT NULL DEFAULT 0.0,
    parallel_penalty REAL    NOT NULL DEFAULT 0.0,
    current_beta     REAL    NOT NULL DEFAULT 0.6,
    selected_policy  TEXT    NOT NULL DEFAULT 'baseline_parallel_refining'
);

CREATE TABLE IF NOT EXISTS policies (
    id              TEXT PRIMARY KEY,
    iteration       INTEGER NOT NULL,
    source_code     TEXT NOT NULL,
    score_pareto    REAL    NOT NULL DEFAULT 0.0,
    created_at      TEXT NOT NULL
);