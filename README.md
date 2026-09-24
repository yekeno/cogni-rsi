# cogni-rsi: Recursive Self-Improvement Exploration over Complex SaaS

<p align="center">
  <a href="#english"><b>English</b></a> | <a href="#chinese"><b>简体中文</b></a>
</p>

<p align="center">
  <b>A production-grade Rust implementation of the Dream-RSI framework applied to autonomous cognition and boundary discovery across complex enterprise SaaS domains.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.80%2B-orange.svg" alt="Rust Version" />
  <img src="https://img.shields.io/badge/Architecture-10--Crate%20Workspace-blue.svg" alt="Architecture" />
  <img src="https://img.shields.io/badge/Tests-76%20Passed-brightgreen.svg" alt="Tests" />
  <img src="https://img.shields.io/badge/License-Apache%202.0%20%2F%20MIT-lightgrey.svg" alt="License" />
</p>

---

<a name="english"></a>
## English

### 1. Overview & Theoretical Background: Dream-RSI

Recursive Self-Improvement (RSI) is a foundational objective for autonomous AI systems. A primary challenge in long-horizon exploration across massive search spaces is the evaluation bottleneck:
* **The Exploration Dilemma**: Fixed exploration heuristics fail to adapt as environments scale, while online policy optimization requires navigating vast search spaces with delayed, computationally expensive feedback over real environments.
* **The Dream-RSI Insight** ([Zheng et al., 2026](#citation)): Realized exploration histories can be structured into an offline **Replay Simulator**. Within this simulator, the agent conducts **"dreaming"**—evaluating and refining candidate exploration policies off-policy using counterfactual prefix-trees without invoking repetitive, expensive online evaluations.
* Once refined through dreaming, the superior policy is redeployed online to explore new frontiers, feeding new trajectories back into the replay store in a continuous, self-improving flywheel.

```text
               ┌────────────────────────────────────────────────────────┐
               │              Online Environment (SaaS Target)          │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          │ Real Action Rollout (Read-Only)
                                          ▼
                               ┌─────────────────────┐
                               │  Replay Store       │
                               │  (Prefix-Only Tree) │
                               └──────────┬──────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        │ Off-Policy Dreaming Optimization  │
                        │                                   │
                        │   Policy Candidates {π^0, π^m}    │
                        │                │                  │
                        │                ▼                  │
                        │   Simulation: V(π, D_explore)     │
                        │                │                  │
                        │                ▼                  │
                        │      Argmax Selection (m*)        │
                        └─────────────────┬─────────────────┘
                                          │
                                          │ Winner Policy Deployed
                                          ▼
                               ┌─────────────────────┐
                               │ Next Outer Iteration│
                               └─────────────────────┘
```

### 2. Cogni-RSI: Architecture & Implementation

`cogni-rsi` brings Dream-RSI to life in Rust, targeting the automated cognition and frontier mastery of enterprise SaaS systems (e.g., Financial and Business ERP integrations).

#### The Dream-RSI Objective Function in `cogni-dreamer`

Each outer iteration optimizes the exploration policy $\pi$ over historical tree $\mathcal{D}_{\text{explore}}$ according to Eq. (1) of the Dream-RSI specification:

$$V(\pi; \mathcal{D}_{\text{explore}}) = \text{AUC}_{\text{Pareto}}(\pi; \mathcal{D}_{\text{explore}}) - \beta \cdot \frac{K(\pi; \mathcal{D}_{\text{explore}})}{P(\pi; \mathcal{D}_{\text{explore}})}$$

Where:
* $\text{AUC}_{\text{Pareto}}$ measures the quality of discovered states on the frontier.
* $K$ represents sequential exploration rounds, and $P$ represents total parallel probes.
* $\beta \cdot \frac{K}{P}$ serves as a strict penalty against excessive serialized reasoning, encouraging effective parallel search.
* The incumbent policy $\pi^0$ competes against $M$ perturbed candidates (narrow, wide, deep, and LLM-synthesized policies via MiniMax-M3). A candidate replaces the incumbent if and only if $V(\pi^m) \ge V(\pi^0)$.

#### Workspace Crates Structure

The repository is modularized into 10 decoupled crates:

| Crate | Responsibility |
|---|---|
| **`cogni-core`** | Fundamental domain models: `CognitionNode`, `CognitionTree`, `Score`, `DirectionPlan`, `DailyProgressRow`. |
| **`cogni-cognition-tree`** | Tree structural analysis, `boundary.yaml` SaaS specification, `TreeDelta` tracking (`+N/-M/rK`), frontier leaves, and `goal_progress` & `boundary_mastery` metrics. |
| **`cogni-replay`** | **Prefix-only** simulator (`LocalReplayStore`). Strictly prohibits hallucinating unobserved branches or unauthorized node mutations. |
| **`cogni-policy`** | `CognitionPolicy` trait, baseline grid search, and candidate policy parameter models. |
| **`cogni-crawler`** | Source adapters for SaaS targets (docs, codebases, APIs) with a strict, default-deny `WriteGate`. |
| **`cogni-dreamer`** | The outer dreaming loop, equation (1) scoring, candidate generation, and SQLite synchronization. |
| **`cogni-llm`** | Dual-mode LLM transport (live OpenAI-compatible client targeting MiniMax-M3, plus zero-cost deterministic dry-run fallback). |
| **`cogni-store`** | SQLite storage with `sqlite-vec` integration; persists daily progress, policy XMLs, and historical trajectories. |
| **`cogni-tui`** | Terminal User Interface built on Ratatui with 4 block renderers: Day Header, Policy Step, Thinking Stream, and Termination Status. |
| **`cogni-cli`** | User-facing CLI entrypoint (`dream`, `tui`, `show-day`, `smoke`, `prompts`, `recent-rows`). |

---

### 3. Getting Started & Usage Guide

#### Prerequisites
* **Rust**: `1.80.0` or later (`rustup update stable`)
* **SQLite**: `3.38+`

#### Configuration (`.env`)
Copy the template and configure your parameters:
```bash
cp .env.example .env
```

Key environment variables:
```ini
# MiniMax LLM Settings (OpenAI-compatible protocol)
MINIMAX_API_KEY=your_minimax_api_key_here
COGNI_LLM_BASE_URL=https://api.minimax.chat/v1
COGNI_LLM_MODEL=minimax-m3

# Safety Guardrail: 0 = Read-Only, 1 = Allow Write actions
COGNI_ALLOW_WRITE=0

# SQLite Persistence
COGNI_DB_URL=sqlite://cogni.db
```
> **Security Notice**: `.env` and `cogni.db` are strictly excluded in `.gitignore`. Secrets are never committed or logged.

#### Running Subcommands

1. **Autonomous Dreaming Loop (`dream`)**
   Executes $N$ outer iterations of policy discovery, frontier expansion, and SQLite recording:
   ```bash
   # Run 5 dreaming iterations with default SaaS boundary spec
   cargo run -p cogni-cli -- dream --db cogni.db --iterations 5

   # Run with a custom boundary domain specification
   cargo run -p cogni-cli -- dream --db cogni.db --iterations 10 --boundary path/to/boundary.yaml
   ```

2. **Interactive Terminal UI Dashboard (`tui`)**
   Visualizes cognition trajectory history, frontier metrics, and real-time parameters:
   ```bash
   cargo run -p cogni-cli -- tui --db cogni.db --limit 10
   ```

3. **Smoke Test Execution (`smoke`)**
   Runs a single dry-run iteration verifying state progression:
   ```bash
   cargo run -p cogni-cli -- smoke
   ```

4. **Inspect Day Progress (`show-day`)**
   Inspects a specific historical day header:
   ```bash
   cargo run -p cogni-cli -- show-day --day 1
   ```

5. **List Prompt Contracts (`prompts`)**
   Dumps verbatim system prompt contracts used across the three-stage loop:
   ```bash
   cargo run -p cogni-cli -- prompts
   ```

6. **Recent Progress Rows (`recent-rows`)**
   Prints formatted SQLite records from `daily_progress`:
   ```bash
   cargo run -p cogni-cli -- recent-rows --db cogni.db --limit 5
   ```

---

### 4. Milestone Implementation History

* **PR #1: Foundation & Scaffold**
  * Initialized 10-crate workspace layout, workspace `Cargo.toml`, and dependency graph.
  * Formulated core domain types: `CognitionNode`, `Score`, `CognitionTree`.
  * Established hard invariants: read-only crawl protection and prefix-only replay.
* **PR #2: Cognition Tree Core & Storage Schema**
  * Implemented `CognitionTree` index and parent-child navigation.
  * Built SQLite schema in `cogni-store` with `daily_progress` and `policies` tables.
* **PR #3: Dream-RSI Eq.(1) Objective & Perturbation Engine**
  * Implemented paper objective formula (1) with Pareto AUC and parallel efficiency penalty $\beta \cdot \frac{K}{P}$.
  * Built policy variant generator (`narrow`, `wide`, `deep`, `balanced`).
* **PR #4: LLM Transport & Prompt Contracts**
  * Integrated `rig-core` OpenAI-compatible client targeting MiniMax-M3.
  * Added robust dry-run fallback when API keys are absent or endpoints hit rate limits.
  * Formalized XML prompt parser for policy improvements.
* **PR #5: Policy XML Persistence & Multi-Day Rollout**
  * Persisted candidate and winner policy XML source code into SQLite.
  * Built audit trail enabling exact replay of historical policy evaluations.
* **PR #6: Ratatui TUI 4-Block Renderer**
  * Implemented TUI widget architecture with 4 distinct blocks: Day Header, Policy Step, Thinking Stream, and Termination Status.
* **PR #7: End-to-End TUI & CLI Integration**
  * Connected live SQLite reads into Ratatui TUI dashboard.
  * Added subcommands: `tui`, `show-day`, `smoke`, `prompts`, `recent-rows`.
* **PR #8: Frontier Analysis & Boundary Mastery**
  * Implemented `BoundarySpec` and `BoundaryDomain` with `boundary.yaml` parser and embedded SaaS presets.
  * Implemented `TreeDelta` tracking structural transitions (`new_nodes`, `closed_nodes`, `reopened_nodes`).
  * Built real frontier metrics: weighted `goal_progress` and $|frontier \cap boundary| / |frontier|$ `boundary_mastery`.
  * Connected outer dreaming loop to frontier child node expansion and real SQLite metric persistence.

---

<a name="chinese"></a>
## 简体中文

### 1. 概述与理论背景：Dream-RSI

递归自我改进（Recursive Self-Improvement, RSI）是构建高阶自主智能系统的核心方向。然而，在海量搜索空间与长链条任务中，传统方法面临着严峻的评估瓶颈：
* **探索的固有困境**：固定的探索策略在复杂空间扩展时容易停滞；而直接在线优化策略不仅耗时漫长，且每次真实交互与验证的成本高昂、反馈存在显著延迟。
* **Dream-RSI 核心思想**（[Zheng et al., 2026](#citation)）：智能体历史探索所积累的路径，本身构成了对已知搜索空间的真实采样。Dream-RSI 将这些历史沉淀为**可回放的模拟器（Replay Simulator）**，并在其中展开**“做梦（Dreaming）”**——利用反事实前缀树在离线模拟环境中低成本、高并发地评估与迭代探索策略，无需频繁执行昂贵的真实在线验证。
* 经由 Dreaming 胜出的更优策略，随即重新部署至在线环境中拓宽未探知边界，并将新的探索轨迹反哺回模拟器，形成**“在线探索 ➔ 离线做梦 ➔ 策略进化 ➔ 深度探索”**的正向自驱循环。

```text
               ┌────────────────────────────────────────────────────────┐
               │              真实在线环境（SaaS 目标系统）                │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          │ 真实探索行为（严格只读）
                                          ▼
                               ┌─────────────────────┐
                               │  回放模拟器 (Replay)  │
                               │  (纯前缀不可伪造树)    │
                               └──────────┬──────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        │       离线做梦策略优化 (Dreaming)    │
                        │                                   │
                        │   策略候选集 {π^0, π^m, π^LLM}     │
                        │                │                  │
                        │                ▼                  │
                        │   模拟评估: V(π, D_explore)        │
                        │                │                  │
                        │                ▼                  │
                        │      Argmax 择优选择 (m*)          │
                        └─────────────────┬─────────────────┘
                                          │
                                          │ 优胜策略在线重新部署
                                          ▼
                               ┌─────────────────────┐
                               │     下一轮外部迭代   │
                               └─────────────────────┘
```

### 2. Cogni-RSI：架构设计与工程实现

`cogni-rsi` 是 Dream-RSI 理论在 Rust 工业级工程中的完整落地，专为**复杂企业级 SaaS 财务与业务一体化系统的认知学习与边界掌控**而设计。

#### 核心优化目标函数（`cogni-dreamer`）

每一轮外部迭代均遵循 Dream-RSI 论文中的公式 (1) 评估策略：

$$V(\pi; \mathcal{D}_{\text{explore}}) = \text{AUC}_{\text{Pareto}}(\pi; \mathcal{D}_{\text{explore}}) - \beta \cdot \frac{K(\pi; \mathcal{D}_{\text{explore}})}{P(\pi; \mathcal{D}_{\text{explore}})}$$

其中：
* $\text{AUC}_{\text{Pareto}}$：前沿叶子节点的认知质量与 Pareto 收益。
* $K$：策略执行所需的串行探测轮数（Rounds）。
* $P$：策略并发探测的总节点数（Probes）。
* $\beta \cdot \frac{K}{P}$：串行化惩罚项，抑制盲目的串行深搜，惩罚缺乏并发效率的策略。
* 现有策略 $\pi^0$ 与 $M$ 个变异候选策略（狭窄型、广度型、深度型、以及基于 MiniMax-M3 LLM 生成的策略）进行离线推演竞争。仅当 $V(\pi^m) \ge V(\pi^0)$ 时触发策略升级。

#### 工作区 10-Crate 架构

整个系统解耦为 10 个独立的 Rust Crate，保证核心不变式与高可测性：

| Crate | 功能职责 |
|---|---|
| **`cogni-core`** | 基础领域模型：`CognitionNode`（认知节点）、`Score`（评分）、`DirectionPlan`（探索方向规划）、`DailyProgressRow`。 |
| **`cogni-cognition-tree`** | 认知树拓扑结构、`boundary.yaml` 业务边界规范解析、`TreeDelta` 状态增量变更（`+N/-M/rK`）、前沿叶子提取与目标推进度/边界掌控度计算。 |
| **`cogni-replay`** | **纯前缀（Prefix-Only）**回放模拟器（`LocalReplayStore`），严禁凭空捏造未曾观测的事实或篡改历史。 |
| **`cogni-policy`** | `CognitionPolicy` 抽象接口、网格计划基线算法与策略超参模型。 |
| **`cogni-crawler`** | 面向 SaaS 目标源（文档、代码库、API 端点）的适配器，内置强制默认拒绝的写保护安全门（`WriteGate`）。 |
| **`cogni-dreamer`** | 外部做梦循环引擎、论文公式 (1) 模拟评估、多候选变异生成与 SQLite 状态同步。 |
| **`cogni-llm`** | 双模 LLM 传输层（适配 MiniMax-M3 的 OpenAI 兼容协议客户端，且具备免外部凭证的安全确定性 Dry-Run 回退）。 |
| **`cogni-store`** | 基于 SQLite（并预留 `sqlite-vec` 向量检索）的持久化存储，记录每日进度、策略 XML 全量源码与拓扑日志。 |
| **`cogni-tui`** | 基于 Ratatui 构建的终端可视化监控面板，呈现 Day Header、Policy Step、Thinking Stream 及 Stop Status 四大模块。 |
| **`cogni-cli`** | 统一命令行交互入口（支持 `dream`, `tui`, `show-day`, `smoke`, `prompts`, `recent-rows`）。 |

---

### 3. 安装与使用指南

#### 环境依赖
* **Rust**: `1.80.0` 或更高版本（`rustup update stable`）
* **SQLite**: `3.38+`

#### 配置文件初始化（`.env`）
复制环境变量模板并填入配置：
```bash
cp .env.example .env
```

核心配置项说明：
```ini
# MiniMax 模型接口配置 (标准 OpenAI 兼容协议)
MINIMAX_API_KEY=your_minimax_api_key_here
COGNI_LLM_BASE_URL=https://api.minimax.chat/v1
COGNI_LLM_MODEL=minimax-m3

# 写入安全门禁：0 = 严格只读（默认），1 = 允许写操作（极度危险，仅受控场景使用）
COGNI_ALLOW_WRITE=0

# SQLite 数据库持久化路径
COGNI_DB_URL=sqlite://cogni.db
```
> **安全说明**：`.env` 与本地数据库文件 `cogni.db` 均已被 `.gitignore` 严格排除，杜绝密钥外泄风险。

#### CLI 命令使用示例

1. **执行外部做梦循环（`dream`）**
   运行 $N$ 轮策略做梦、前沿节点扩展及 SQLite 指标记录：
   ```bash
   # 执行 5 轮 dreaming，默认使用内置 SaaS 财务/业务领域边界规范
   cargo run -p cogni-cli -- dream --db cogni.db --iterations 5

   # 指定自定义 boundary.yaml 业务边界规范运行
   cargo run -p cogni-cli -- dream --db cogni.db --iterations 10 --boundary boundary.yaml
   ```

2. **启动交互式终端面板（`tui`）**
   启动基于 Ratatui 的交互式监控看板，查看历史认知进展与最新指标：
   ```bash
   cargo run -p cogni-cli -- tui --db cogni.db --limit 10
   ```

3. **冒烟测试（`smoke`）**
   无需外部依赖，运行单步 Dry-Run 校验循环完整性：
   ```bash
   cargo run -p cogni-cli -- smoke
   ```

4. **单日进度报告（`show-day`）**
   查看指定某一天的决策头部信息与状态：
   ```bash
   cargo run -p cogni-cli -- show-day --day 1
   ```

5. **查看三阶段 Prompt 协议（`prompts`）**
   打印系统三阶段循环（探索、回放策略改进、网格规划）的原始 Prompt 模板：
   ```bash
   cargo run -p cogni-cli -- prompts
   ```

6. **查询数据库记录（`recent-rows`）**
   从 SQLite 的 `daily_progress` 表直接拉取最近记录：
   ```bash
   cargo run -p cogni-cli -- recent-rows --db cogni.db --limit 5
   ```

---

### 4. 提交里程碑演进历史

* **PR #1: 骨架工程与核心不变式设立**
  * 构建 10-crate 工作区脚手架，定义 `CognitionNode` 与 `CognitionTree`。
  * 确立两大核心不变式：爬虫写操作默认拒绝门禁（`WriteGate`）与回放器纯前缀只读不可伪造（`Prefix-Only`）。
* **PR #2: 认知树核心与 SQLite 存储底座**
  * 实现基于 Ulid 的认知树索引与父子分支双向导航。
  * 完成 `cogni-store` SQLite 模式设计，支持 `daily_progress` 及 `policies` 事务性入库。
* **PR #3: Dream-RSI 公式(1)与策略变异引擎**
  * 完整实现论文公式 (1) 目标函数：Pareto AUC 奖励结合并发效率惩罚 $\beta \cdot \frac{K}{P}$。
  * 构建参数变异生成器（狭窄型、宽度型、深度型与平衡型）。
* **PR #4: LLM 传输层与 Prompt 契约化**
  * 基于 `rig-core` 接入 MiniMax-M3 OpenAI 兼容协议。
  * 内置无需密钥的确定性 Dry-Run 回退机制与 XML 策略解析提取器。
* **PR #5: 策略 XML 审计持久化与回放验证**
  * 实现获胜策略与候选策略 XML 源码的持久化存储，构建完整可审计思考链。
* **PR #6: Ratatui 终端 TUI 四块渲染引擎**
  * 实现 Day Header、Policy Step、Thinking Stream 及 Stop Status 独立渲染组件。
* **PR #7: 终端 UI 与 CLI 全链路集成**
  * 打通 SQLite 至 TUI 的实时数据管道，丰富 CLI 子命令体系。
* **PR #8: 认知树前沿分析与边界掌控度 (Boundary Mastery)**
  * 引入 `BoundarySpec` / `BoundaryDomain` 规范与 `boundary.yaml` 解析器，预置 SaaS 典型域。
  * 实现 `TreeDelta` 结构增量变更追踪（新发现 `+N`、失效 `-M`、修复 `rK`）。
  * 接入加权 `goal_progress` 与边界覆盖率 `boundary_mastery` 真实计算，彻底消除硬编码与代理指标。

---

<a name="citation"></a>
### 5. Citation & Reference

If you find this research implementation helpful, please cite the original Dream-RSI foundation paper:

```bibtex
@article{zheng2026dreamrsi,
  title   = {Dream-RSI: Recursive Self-Improvement through Evolving Worlds},
  author  = {Tong Zheng and Xidong Wu and Zheng Zhang and Zhankui He and 
             Chaoyi Zhang and Benjamin Coleman and Ruoqiao Wei and Di Bai and 
             Haolin Liu and Rui Liu and Xue Wang and Yue Zhuan and 
             Wang-Cheng Kang and Renkai Xiang and Heng Huang and 
             Xinwu Cheng and Yunsong Guo},
  journal = {arXiv preprint arXiv:2609.14858v1},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.14858}
}
```

* **Paper**: [arXiv:2609.14858v1 [cs.CL]](https://arxiv.org/abs/2609.14858)
* **Official Project**: [dream-rsi.com](https://dream-rsi.com) | [github.com/zhengkid/Dream-RSI](https://github.com/zhengkid/Dream-RSI)
