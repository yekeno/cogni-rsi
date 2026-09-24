# CLAUDE.md

## What this repo is

`cogni-rsi` — a Rust implementation of the Dream-RSI outer loop, applied to
building a cognition tree over a fixed target SaaS (financial + business
integrated suite). The agent's goal is to master the SaaS's boundaries,
documents, code, and APIs over recursive daily sessions.

## Required reading before coding

* `docs/PROPOSAL.md` — full design (mapping table, three-stage loop, data model).
* `docs/prompts/online_explore.md`, `docs/prompts/replay_policy_improve.md`,
  `docs/prompts/grid_plan.md` — verbatim prompt contracts.
* The original paper: `../book/dream-rsi/2609.14858v1.pdf` (Dream-RSI).

## Hard invariants

* **Prefix-only replay.** `cogni-replay` MUST NOT fabricate observations.
  The unit tests in `crates/cogni-replay/src/lib.rs` enforce this.
* **Read-only by default.** `cogni-crawler` denies write calls unless
  `COGNI_ALLOW_WRITE=1`.
* **No secrets in source.** All credentials via env vars loaded by `dotenvy`.

## Workflow

1. Read `~/.claude/rules/common/development-workflow.md` first.
2. Run `cargo check --workspace` after every change.
3. Add a unit test alongside every new function (TDD).
4. Use the `code-reviewer` agent before pushing.

## Layout

```
crates/
  cogni-core            domain types: Node, Tree, Score, DirectionPlan, DailyProgress
  cogni-llm             LLM client (minimax-m3) + 3 prompt templates
  cogni-store           SQLite + sqlite-vec, daily_progress table
  cogni-replay          prefix-only simulator (LocalReplayStore)
  cogni-policy          CognitionPolicy trait + BaselineParallelRefining
  cogni-crawler         SourceAdapter trait (doc/code/api), WriteGate
  cogni-cognition-tree  frontier/merge/insert
  cogni-dreamer         outer iteration loop + replay score formula
  cogni-tui             ratatui renderers for the four block kinds
  cogni-cli             clap entry point
```

## Useful commands

* `cargo test -p cogni-replay` — verify the prefix-only invariant.
* `cargo run -p cogni-cli -- smoke` — run one dummy outer iteration.
* `cargo run -p cogni-cli -- show-day --day 3` — preview the day header.
* `cargo run -p cogni-cli -- prompts` — list the three prompt files.