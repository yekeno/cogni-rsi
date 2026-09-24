# grid_plan.md — Dream-RSI Listing 3 / `plan_grid` constraint

This file is a thin wrapper that references `replay_policy_improve.md` for the
full policy-improvement contract. cogni-rsa adopts the same prefix-only,
deterministic grid planner as the paper; concrete grid values are emitted by
`cogni_policy::CognitionPolicy::plan_grid` (see `crates/cogni-policy/src/lib.rs`).

Key rules the planner must honour:

1. Read only prefix-safe fields from `GridContext` (no raw trace outcomes).
2. Return `GridPlan { branch_count, refine_count, reason }` on every code path.
3. Honour `failure_budget_hard`, `worker_cap`, and `beta`.
4. Apply the "width vs depth from evidence" heuristic from
   `replay_policy_improve.md` §Required next-cycle grid planning.