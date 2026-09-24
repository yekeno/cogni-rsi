# cogni-rsi:面向 SaaS 财务/业务一体化系统认知的递归自我改进系统

> 基于 Dream-RSI(Zheng et al., 2609.14858v1)的核心机制,把"把历史当作可回放的模拟器,在里面做梦以改进策略"的思想迁移到 **对一个 SaaS 财务业务一体化系统的认知学习** 上。

---

## 1. 目标与核心机制

### 1.1 你想让系统做的事

对一个**固定目标**(某 SaaS 财务/业务一体化系统的功能边界、文档、代码、API 行为),由 LLM 驱动的认知 Agent:

1. **持续思考** 你给出的问题(每天/每个 session 多次);
2. 在对话框中**逐步展开思考过程**,显式给出
   - 当前的思维方向(`current_direction`),
   - 目标达成度(`goal_progress`)与边界掌握度(`boundary_mastery`),
   - 每一天相对前一天的进展(`daily_delta`),
   - 已建立的认知图谱节点(`cognition_nodes`);
3. 通过"**做梦**"机制,在**不反复调用真实 SaaS 接口**的前提下,演练并迭代自己的提问策略;
4. 当"梦"里的策略稳定后,再发起一次真正的"在线探索"(读文档/读代码/调用真实系统只读接口),把新数据写回历史。

### 1.2 与 Dream-RSI 的概念对齐

| Dream-RSI 概念 | cogni-rsi 中的映射 |
|---|---|
| **exploration policy**(策略 π) | 当前对 SaaS 系统的"提问方向 / 调查方向",由 LLM 维护 |
| **discovery tree** T | **cognition tree** —— 文档节点/代码节点/功能节点/边界节点的树状历史 |
| **candidate**(尝试 w_k) | 一次"思考块"或一次"实操查询"(运行提案/响应) |
| **evaluation / score** s_v | 对该节点的达成度评分:理解度 0–1、引用证据数、覆盖维度数 |
| **replay simulator** | 从已沉淀的历史节点中重放,**不重新调用真实 SaaS 接口** |
| **online rollout** | 在真实文档/代码/系统上发起一次新的探索(读 PDF / 读源码 / 调用只读 API) |
| **dreaming** | 让 LLM 在 simulator 上反复演练,输出思考过程 |
| **outer iteration t** | **day t**(每一天或一个用户 session) |
| **policy-development agent** | 单独一个 LLM Agent,负责改写提问策略 |
| **LLMDesignedMethod / OptimalPolicy** | Rust trait `CognitionPolicy`,由 LLM 通过 prompt + 模板代码生成实现 |
| **GridPlan / beta / batch** | "我下一轮要看哪几个分支、看多深、并行几条" 的显式策略声明 |
| **concurrent calls, W workers** | 真实系统的并行只读调用(读多个模块/多份文档) |

### 1.3 三阶段循环(完全对齐 Dream-RSI 图 1)

```
┌────────────────────────────────────────────────────────────────────┐
│                       outer iteration t (= day / session)          │
│                                                                    │
│  ① Online Explore    ──► 用当前策略 π_t 在真实系统上展开新节点       │
│        │                 (读文档/读代码/调用只读API)               │
│        ▼                                                       │
│  history H_t = H_{t-1} ∪ {T_t}                                │
│        │                                                       │
│  ② Construct Replay Simulator                                    │
│        │   把 history 转成可"重走"的 ReplayStore:                │
│        │     - 每个节点 v 记录:父节点、workspace、score、tag、证据  │
│        │     - 提供 prefix-only 查询接口(只返回已观察到的子树)    │
│        ▼                                                       │
│  ③ Dreaming-based Policy Improvement                             │
│        │   对 M 个候选策略版本 π_t^0 = π_t, π_t^1, ..., π_t^{M-1}│
│        │   全部在 ReplayStore 上 reproduce(无需真实调用),       │
│        │   得到各自 replay score V^m。                           │
│        │   选中 V 最高的作为 π_{t+1}。                           │
│        ▼                                                       │
│  π_{t+1} ──► 进入 day t+1 的 Online Explore                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. 系统架构

### 2.1 模块划分(Rust crate workspace)

```
cogni-rsi/
├── Cargo.toml                    # workspace
├── crates/
│   ├── cogni-core/               # 领域类型:Node, Edge, Score, Tag
│   ├── cogni-llm/                # LLM 客户端(minimax-m3)+ prompt 编排
│   ├── cogni-store/              # SQLite/Postgres 历史持久化 + 向量检索
│   ├── cogni-replay/             # ReplayStore + prefix-only 查询
│   ├── cogni-policy/             # CognitionPolicy trait + 默认实现
│   ├── cogni-crawler/           # 真实系统适配层:文档/代码/只读 API
│   ├── cogni-cognition-tree/    # 认知树构造、剪枝、版本管理
│   ├── cogni-dreamer/           # dreaming 主循环:批策略生成、replay 评分
│   ├── cogni-tui/               # 对话框 + 思考过程可视化(基于 ratatui)
│   └── cogni-cli/               # CLI 入口
└── docs/
    ├── PROPOSAL.md              # 本文档
    └── prompts/                 # 在线/做梦/策略改写三类 prompt
```

### 2.2 关键依赖(均使用 Rust 生态已有的成熟 crate)

| 依赖 | 用途 | 备注 |
|---|---|---|
| `tokio` | 异步运行时 | 1.x,feature-full |
| `axum` + `tower` | 本地 HTTP API(给 TUI / 第三方调用) | 0.7 |
| `rig-core` | LLM 客户端抽象(支持 OpenAI 兼容 API) | minimax-m3 可作为 OpenAI 兼容 endpoint |
| `reqwest` | 直接 HTTP 调用 LLM | 作为 rig 的 backend 备选 |
| `serde` / `serde_json` | 序列化 | — |
| `rusqlite` 或 `sqlx`(sqlite) | 本地历史持久化 | 离线优先,易调试 |
| `rusqlite` + `sqlite-vec` 或 `qdrant-client` | 节点 embedding 向量检索 | 二选一,默认 sqlite-vec |
| `tree-sitter` + 多语言 grammar | 解析 SaaS 文档/代码为节点 | 复用现成 grammar |
| `ratatui` + `crossterm` | TUI 思考展示 | 风格可控,无需 GUI |
| `anyhow` / `thiserror` | 错误处理 | — |
| `tracing` / `tracing-subscriber` | 结构化日志 | — |
| `clap` | CLI | 4.x |
| `chrono` | 时间戳 | day t 标识 |
| `rust_decimal` | 业务字段(如金额)校验 | 财务场景需要 |

> 之所以选 `rig-core`:它是 Rust 生态中抽象度最合适的 LLM 客户端库,自带 tool calling / structured output,可直接对接 OpenAI 兼容协议(minimax-m3 通过 OpenAI 兼容 endpoint 暴露)。

### 2.3 核心数据模型(节选)

```rust
// crates/cogni-core/src/lib.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionNode {
    pub id: NodeId,                      // ULID
    pub parent: Option<NodeId>,          // 树的父节点
    pub iteration: u32,                  // outer iteration t (day)
    pub kind: NodeKind,                  // Document | Code | ApiCall | Concept | Boundary
    pub title: String,
    pub workspace: serde_json::Value,    // 当时的工作空间快照(页码/函数名/URL...)
    pub observation: String,             // 该节点的实质内容(文档摘录 / 代码摘录 / API 响应)
    pub score: Score,                    // see below
    pub tags: Vec<Tag>,
    pub evidence: Vec<EvidenceRef>,      // 引用到的 PDF 页码 / 源码行号 / API 字段
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub understanding: f32,     // 0..=1, 我对该节点理解到几分
    pub boundary_fit: f32,      // 0..=1, 它是否落在已声明的认知边界上
    pub reuse_potential: f32,   // 0..=1, 它对未来节点有多大的可重用价值
    pub confidence: f32,        // 0..=1, 评估自身置信度
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionTree {
    pub root: NodeId,
    pub nodes: HashMap<NodeId, CognitionNode>,
}

// 一个"direction declaration"对应论文里的 GridPlan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionPlan {
    pub iteration: u32,
    pub opened_branches: Vec<BranchTarget>, // 我接下来要看哪些新方向
    pub refine_targets: Vec<NodeId>,        // 我要继续深入哪些旧节点
    pub max_width: usize,                   // 并发上限 W
    pub max_depth: usize,                   // 单条链最大深度
    pub stopping_rationale: String,         // 为什么停
}
```

### 2.4 Replay Simulator 接口(prefix-only,完全对齐论文 §3)

```rust
// crates/cogni-replay/src/lib.rs
#[async_trait]
pub trait ReplayStore: Send + Sync {
    async fn reset(&self) -> ReplaySession;
}

pub struct ReplaySession {
    store: Arc<dyn ReplayStore>,
    revealed: HashSet<NodeId>,
    closed: HashSet<NodeId>,
    frontier: VecDeque<NodeId>,
}

impl ReplaySession {
    /// prefix-only 观察
    pub async fn observed(&self) -> CognitionTree { ... }

    /// 论文里 legal_actions() 的对应:可揭示的合法下一步
    pub async fn legal_actions(&self) -> Vec<LegalAction> { ... }

    /// 论文里 legal_roots() 的对应:未打开过的根
    pub async fn legal_roots(&self) -> Vec<NodeId> { ... }

    /// 揭示 cell_id,推进区间
    pub async fn probe_batch(
        &mut self,
        cells: &[CellMeta],
        on_reveal: impl FnMut(NodeId, &CognitionNode),
    ) -> Result<()> { ... }

    pub async fn baseline_score(&self) -> Score { ... }
}
```

> 关键不变式:**probe_batch 不允许凭空生成新节点的 observation**;如果该 child 在 history 中不存在,只能返回"unobserved"标记。这保证 dreaming 阶段不会偷跑真实数据。

### 2.5 CognitionPolicy(对应论文里的 `LLMDesignedMethod` / `OptimalPolicy`)

```rust
// crates/cogni-policy/src/lib.rs
#[async_trait]
pub trait CognitionPolicy: Send + Sync {
    fn name(&self) -> &str;

    /// 论文里 OptimalPolicy.solve(self, question, budget=None) 的对应
    async fn solve(
        &self,
        question: &CognitionQuestion,
        budget: Option<Budget>,
        replay: &mut ReplaySession,
    ) -> Result<DirectionPlan>;

    /// 论文里 GridPlan.plan_grid 的对应(确定性、可重放)
    fn plan_grid(&self, ctx: &GridContext) -> GridPlan;
}
```

* **默认实现**:`BaselineScriptFix` — 论文中"parallel refining"的 Rust 版,只看根→并行打 W 个子节点→递归。它是所有策略的起点 π_0,也是固定基线。
* **LLM 生成的策略**:由 policy-development agent 在 dreaming 阶段生成 Rust 源代码(只允许改 `solve`/`plan_grid` 两个方法),落地后由 `cogni-policy::DynamicPolicy::load` 装载。

---

## 3. 思考循环的对话协议

### 3.1 思考块结构(在 TUI/对话框中逐块显示)

```text
──────────────────────────────────────────
[ day 003 ]  iteration t = 3   β = 0.62
──────────────────────────────────────────
▌ direction        收口"应付票据"在结账→总账→报表三层如何走
▌ goal_progress    0.58  (+0.07 vs day 002)
▌ boundary_mastery 0.71  (+0.04)
▌ daily_delta      8 new nodes, 3 closures, 1 reopened
▌ plan             open 2 roots · refine 3 frontiers · W=3 · K2=6
──────────────────────────────────────────

[ 01 ] 已读: docs/finance/ap-bill.md §3.2 → node#a91c (0.82, 0.90)
   ↳ 证据: PDF p.42 §应付票据子系统边界
   ↳ 与昨日 node#4f3(locked)冲突 → re-org 修正

[ 02 ] 已读: src-tauri/src/gl/ap.rs 2 in package ap → node#7c12 (0.74, 0.85)
   ↳ 发现 "结账→总账过账"接口 payables.post_to_gl()
   ↳ 复用候选: → 将进入 day 004 的 refine_targets

[ 03 ] API 调用: GET /api/v1/finance/bill?status=open (W3 并发) → node#3e01 (0.61, 0.79)
   ↳ 实际数据与文档 §3.2 一致 → confidence +0.05

[ thinking ] 现在能不能改写策略?
   ↳ 评估 m=0..3 四个候选策略在 replay 上的 V^m
   ↳ m=2(pareto: 收口应付 + 拓宽预算)得分最高 → π_{t+1} = π_t^2

[ stop ] 仅当三条件同时满足才停:无合法 frontier + 无 legal root + budget 耗尽
──────────────────────────────────────────
```

### 3.2 触发频率

* **per block**:每读到一个文档/代码/API 节点就写一个 `[ k ]` 块。
* **per dream**:每完成一次 dreaming(评估完 M 个候选策略)写一个 `[ thinking ]` 块。
* **per day**:每天 session 结束写一个 `[ day NNN ]` 头部汇总。
* **per stop**:满足停止条件时写一个 `[ stop ]` 块,并解释三条触发条件各占多少比重。

### 3.3 可见的"思考过程"内容(7 项,与论文 LLM prompt §B.2 对齐)

| 字段 | 来源 | 例子 |
|---|---|---|
| 方向调整 | `direction_plan.opened_branches + refine_targets` | "从应付票据切到'结账过账'接口" |
| 目标达成度 | `Score::understanding` 在 frontier 上的加权平均 | 0.58 (+0.07) |
| 边界掌握度 | frontier 与已声明 `boundary.yaml` 的交集比例 | 0.71 (+0.04) |
| 每日进展 | `nodes_added - nodes_closed` (按 day t) | +8 / −3 / +1 reopen |
| 候选评估 | `M` 个策略在 replay 上的 pareto 曲线 | m=2: pareto_reward=0.83, β=0.62 |
| 失败归因 | 每个 closed 节点的 `failure_class` | `hard-unrecoverable` / `repairable` / `unpromising` |
| 明日计划 | 下一天的 `DirectionPlan` | open 2 roots · refine 3 · W=3 |

---

## 4. 与真实 SaaS 系统的对接(`cogni-crawler`)

适配层是唯一可能变化的部分,根据目标 SaaS 替换实现。`cogni-core` 与 `cogni-replay` 不动。

```rust
// crates/cogni-crawler/src/lib.rs
#[async_trait]
pub trait SourceAdapter: Send + Sync {
    /// 读一份 PDF/HTML/Markdown 文档
    async fn read_doc(&self, locator: DocLocator) -> Result<Vec<TextChunk>>;
    /// 读一段源码(给定路径 + 行号区间)
    async fn read_code(&self, locator: CodeLocator) -> Result<Vec<CodeChunk>>;
    /// 调一个只读 API endpoint
    async fn call_api(&self, locator: ApiLocator) -> Result<ApiResponse>;
}
```

内置三个 adapter:

* `LocalFsAdapter`:扫描 `SLAW_FILE_ROOT`(用户把 PDF/HTML/Markdown 放在本地)。
* `GitRepoAdapter`:基于 `git2` 读代码仓(给路径+range 拿内容)。
* `HttpReadOnlyAdapter`:对 SaaS 暴露的只读 OpenAPI 做 GET 调用,带 rate limit + dry-run 模式。

每个 adapter 都必须能 **dry-run**(只读、不写),并且把每次调用的完整响应写入 `CognitionNode::observation`,这样下次 dreaming 就能在 simulator 上重放。

---

## 5. LLM 接入(`cogni-llm`,minimax-m3)

* 使用 `rig-core`(OpenAI 兼容 provider),配置:
  ```toml
  [llm]
  provider = "openai_compat"
  base_url = "https://api.minimax.chat/v1"   # 换成实际 minimax-m3 endpoint
  model = "minimax-m3"
  ```
* 三个 prompt 模板放在 `docs/prompts/`:
  1. `online_explore.md` — 对应论文 Listing 1
  2. `replay_policy_improve.md` — 对应论文 Listing 2
  3. `grid_plan.md` — 对应论文 Listing 3 的 `plan_grid` 约束
* 使用 `rig::agent::AgentBuilder` + `tool!` 宏把 `cogni-replay` 的 prefix-only API 暴露给 LLM(工具签名完全照搬论文 `question.reset / legal_actions / probe_batch / ...`)。

---

## 6. dreaming(循环)主流程(`cogni-dreamer`)

对齐论文 §3.3 `outer iteration t` 的描述:

```rust
pub async fn run_outer_iteration(
    state: &mut CogniState,
    llm: &LlmClient,
) -> Result<()> {
    let t = state.iteration;

    // === ① Online Explore ===
    let policy = state.current_policy.clone();      // π_t
    let online_tree = run_online_rollout(&policy, &state.history, t, K1, W).await?;
    state.history.append_tree(online_tree.clone());
    state.cognition_tree.merge(online_tree);

    // === ② Construct Replay Simulator ===
    let replay = LocalReplayStore::new(state.history.clone());
    let mut session = replay.reset().await;

    // === ③ Dreaming-based Policy Improvement ===
    let candidates = generate_policy_candidates(&llm, &state, M).await?;
    // 评估 π_t^0 = π_t, π_t^1..M-1
    let mut scored = Vec::with_capacity(candidates.len());
    for (m, cand) in candidates.iter().enumerate() {
        let mut s = replay.reset().await;
        let plan = cand.solve(&state.question, Some(K2.into()), &mut s).await?;
        let v = evaluate_plan(&plan, &s).await?;
        scored.push((cand.clone(), v));
    }
    // 选 argmax(V^m),且 V^m ≥ V^0(不退化)
    let best = scored.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    if best.1 >= scored[0].1 {
        state.current_policy = best.0.clone();
    }
    state.iteration += 1;
    Ok(())
}
```

Reveal 评分函数对应论文公式 (1):

```
V^m = max_v s_v
     − β1 · N^m                              // 探测成本惩罚
     + β2 · (N^m / max(1, k^m))              // 并行度奖励
```

`β` 是**按 benchmark 离线 sweep 选定的**,由 policy-development agent 在批间根据 replay 反馈自适应(论文 §B.2 "Beta: fixed per run, adaptive across cycles")。

---

## 7. 每日进展 / 边界掌握的度量

### 7.1 边界文件 `boundary.yaml`

用户一开始声明系统认知边界(目标 SaaS 的子域),例如:

```yaml
boundary:
  - id: finance-core
    modules: [gl, ap, ar, fa, banking, tax]
    surfaces: [结账, 过账, 对账, 票据, 凭证, 报表]
  - id: biz-core
    modules: [inventory, sales, purchase, crm]
    surfaces: [订单, 出入库, 客户, 报价]
  - id: integration
    modules: [gl↔biz, gl↔bank, biz↔tax]
    surfaces: [过账接口, 银行对账单, 发票开具]
```

### 7.2 三个核心指标(每天写入 history)

```sql
-- 每天一条 summary row
CREATE TABLE daily_progress (
    day                INTEGER PRIMARY KEY,
    new_nodes          INTEGER,
    closed_nodes       INTEGER,
    reopened_nodes     INTEGER,
    goal_progress      REAL,    -- 0..1, frontier 上 understanding 的加权均值
    boundary_mastery   REAL,    -- 0..1, frontier∩boundary / frontier
    pareto_auc         REAL,    -- 当前策略的 pareto.auc
    parallel_penalty   REAL,    -- parallel_penalty 的最近一次值
    current_beta       REAL,    -- 当前 β
    selected_policy    TEXT     -- 选中的策略 ID(便于回滚 / A/B)
);
```

`cogni-tui` 启动时读这张表,画出 N 天趋势线;也可以在对话框里 `[ day NNN ]` 头部显示 `Δ vs day NNN-1`。

---

## 8. 安全 / 部署

* **只读为默认**:所有 `SourceAdapter` 默认走 dry-run。破坏性操作(写文件/调用 POST)需通过环境变量 `COGNI_ALLOW_WRITE=1` 显式开启。
* **密钥**:通过 `~/.config/cogni-rsi/.env`(不进 git),`dotenvy` 加载,**绝不**硬编码到源码里。
* **速率限制**:`HttpReadOnlyAdapter` 用 `governor` crate 做 token bucket;每个 endpoint 的 QPS 由用户配置。
* **离线优先**:整个循环可在完全离线状态运行(只要历史已沉淀),LLM 调用也可以走本地模型后端(Ollama 等)。
* **可审计**:所有 `CognitionNode::observation` 持久化到 SQLite,任意时刻都能回放某一天的完整思考链。

---

## 9. 实施路线(对齐 common/development-workflow.md)

1. **Week 1 — Research & Reuse**
   * 用 `gh search code` 找类似的 Rust LLM agent 项目(参考 `openhelper`、`rust-llm-eval` 等)。
   * 用 Context7 拉 `rig-core`、`tree-sitter`、`axum`、`ratatui` 当前版本的 API。
2. **Week 2 — Plan First**
   * planner agent 输出本仓库的 task_list 与每个 crate 的接口签名。
3. **Week 3 — TDD Scaffold**
   * 先写 `cogni-replay` 的 prefix-only 不变式测试(任何人写出"凭空生成 observation"的代码都会失败)。
   * 写 `cogni-policy` 的 trait 测试,默认实现满足 §3.4 的 hard constraints。
4. **Week 4 — 实现 dreaming 循环**
   * `cogni-dreamer` 主循环 + paper-to-code 公式(1)。
5. **Week 5 — TUI & Webhook**
   * `cogni-tui` 显示 §3.1 的思考块;可选 axum 服务暴露 JSON 进度给外部。
6. **Week 6 — Code Review & 文档**
   * code-reviewer / security-reviewer;补 CLAUDE.md 与 README。

---

## 10. 成功标准

* `cargo test -p cogni-replay` 通过 prefix-only 不变式。
* `cargo test -p cogni-dreamer` 通过论文公式 (1) 的单元测试。
* 在本机对一个真实 SaaS(如某开源 ERP)的 50 个节点做一次 outer iteration:dreaming 比 baseline 策略在 pareto.auc 上有可见提升。
* TUI 能连续显示 `[ day N ] [ k ] [ thinking ] [ stop ]` 块,且每日 summary 写入 `daily_progress`。