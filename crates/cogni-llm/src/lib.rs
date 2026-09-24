//! LLM client for minimax-m3 (OpenAI-compatible) and prompt orchestration.
//!
//! Three prompt templates map to Dream-RSI's Listings 1–3:
//! * `online_explore` — drives the discovery agent during online rollout.
//! * `replay_policy_improve` — drives the policy-development agent during
//!   dreaming.
//! * `grid_plan` — drives the deterministic grid planner.
//!
//! PR #3 wires `LlmClient::complete_text` over `rig-core 0.5`'s
//! OpenAI-compatible provider.  When `dry_run` is set (or the configured
//! env-var is missing), the client returns a deterministic stub response
//! so the dreaming loop never depends on a network call.

use serde::{Deserialize, Serialize};
// rig-core 0.5 publishes the crate name `rig-core` but its lib name is `rig`,
// so we import via `rig::...` and alias the `CompletionModel` trait to
// avoid colliding with the OpenAI provider struct of the same name.
use rig::completion::CompletionModel as CompletionModelTrait;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub base_url: String,
    pub model: String,
    /// Name of the env-var that holds the API key. The actual key value is
    /// NEVER serialized or logged.
    pub api_key_env: String,
    /// Number of retries on transient errors before surfacing a hard error.
    pub max_retries: u32,
    /// Per-request timeout, seconds. The rig-core client does not expose a
    /// per-request budget, so we wrap the future with `tokio::time::timeout`.
    pub timeout_secs: u64,
    /// Sampling temperature passed to the provider.
    pub temperature: f64,
    /// When true, never call the network: return a stubbed completion. The
    /// dry-run flag is honored by `LlmClient::from_config` so test binaries
    /// never reach out.
    pub dry_run: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "openai_compat".into(),
            base_url: "https://api.minimax.chat/v1".into(),
            model: "minimax-m3".into(),
            api_key_env: "MINIMAX_API_KEY".into(),
            max_retries: 2,
            timeout_secs: 30,
            temperature: 0.2,
            // default: dry-run ON so the binary is safe to run without a key.
            // Callers (the dreaming orchestrator) flip it off once a real key
            // is detected in the environment.
            dry_run: true,
        }
    }
}

impl LlmConfig {
    /// Build a config that's actually allowed to hit the network. Callers
    /// should only use this when `MINIMAX_API_KEY` is present.
    pub fn live() -> Self {
        Self {
            dry_run: false,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PromptTemplate {
    OnlineExplore,
    ReplayPolicyImprove,
    GridPlan,
}

impl PromptTemplate {
    pub fn file_name(&self) -> &'static str {
        match self {
            PromptTemplate::OnlineExplore => "online_explore.md",
            PromptTemplate::ReplayPolicyImprove => "replay_policy_improve.md",
            PromptTemplate::GridPlan => "grid_plan.md",
        }
    }
}

/// Trait that abstracts over the LLM transport so the dreaming orchestrator
/// (and tests) can substitute a stub client without touching the rig-core
/// dep tree.
#[async_trait::async_trait]
pub trait LlmTransport: Send + Sync {
    async fn complete(&self, prompt: &str) -> anyhow::Result<String>;
}

/// Real client backed by `rig-core`'s OpenAI-compatible provider.
///
/// Construct via [`LlmClient::from_config`] — the constructor refuses to
/// hit the network unless `cfg.dry_run == false` AND the configured env
/// variable resolves to a non-empty string. The key itself is read once,
/// held in memory, and never logged.
pub struct LlmClient {
    cfg: LlmConfig,
    /// Resolved API key. Empty in dry-run. Held for diagnostics but never
    /// logged — use `tracing::debug!` only with explicit field-level gating.
    #[allow(dead_code)]
    api_key: String,
    /// Cached rig-core completion model. `None` when dry-run.
    inner: Option<rig::providers::openai::CompletionModel>,
}

impl LlmClient {
    /// Build a client. Reads `cfg.api_key_env` from `std::env::var` exactly
    /// once. If the env var is missing OR `cfg.dry_run` is set, the client
    /// is constructed in dry-run mode and `complete()` returns a stub.
    pub fn from_config(cfg: LlmConfig) -> Self {
        let api_key = std::env::var(&cfg.api_key_env).unwrap_or_default();
        let dry_run = cfg.dry_run || api_key.is_empty();
        let inner = if !dry_run {
            // Note: rig-core's openai client internally appends "/v1/chat/completions"
            // with `format!("{}/{}", base_url, path).replace("//", "/")`.
            // Strip trailing `/v1` or `/` so we don't end up with `/v1/v1/chat/completions`.
            let normalized_base = cfg
                .base_url
                .trim_end_matches('/')
                .trim_end_matches("/v1")
                .trim_end_matches('/');
            let client =
                rig::providers::openai::Client::from_url(&api_key, normalized_base);
            Some(rig::providers::openai::CompletionModel::new(
                client,
                &cfg.model,
            ))
        } else {
            None
        };
        Self {
            cfg,
            api_key,
            inner,
        }
    }

    pub fn is_dry_run(&self) -> bool {
        self.inner.is_none()
    }

    /// Stub response used in dry-run. The text shape is chosen so the
    /// replay_policy_improve parser in `cogni-dreamer` can still extract
    /// reasonable defaults (width=4, depth=4, close_on_failure=false).
    fn stub(&self, prompt: &str) -> String {
        let tail: String = prompt.chars().rev().take(8).collect::<String>().chars().rev().collect();
        format!(
            "<policy>\n  max_width: 4\n  max_depth: 4\n  close_on_failure: false\n  beta: 0.6\n  rationale: stub for {tail}\n</policy>"
        )
    }
}

#[async_trait::async_trait]
impl LlmTransport for LlmClient {
    async fn complete(&self, prompt: &str) -> anyhow::Result<String> {
        let Some(model) = self.inner.as_ref() else {
            return Ok(self.stub(prompt));
        };
        let timeout = std::time::Duration::from_secs(self.cfg.timeout_secs);
        // PR #3: a single-shot retry on timeout. More sophisticated backoff
        // lives in PR #4 once we have real failure telemetry.
        let mut last_err: Option<anyhow::Error> = None;
        for _ in 0..self.cfg.max_retries.max(1) {
            let attempt = async {
                let req = CompletionModelTrait::completion_request(model, prompt)
                    .temperature(self.cfg.temperature)
                    .build();
                let resp = CompletionModelTrait::completion(model, req)
                    .await
                    .map_err(|e| {
                        anyhow::anyhow!("rig-core completion error: {e}")
                    })?;
                let text = match resp.choice {
                    rig::completion::ModelChoice::Message(s) => s,
                    rig::completion::ModelChoice::ToolCall(name, _) => {
                        anyhow::bail!("provider returned a tool-call ({name}) for plain-text prompt");
                    }
                };
                Ok::<String, anyhow::Error>(text)
            };
            match tokio::time::timeout(timeout, attempt).await {
                Ok(Ok(text)) => return Ok(text),
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    last_err = Some(anyhow::anyhow!(
                        "timeout after {}s (provider={}, model={})",
                        timeout.as_secs(),
                        self.cfg.provider,
                        self.cfg.model,
                    ));
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("unknown failure")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_targets_minimax_m3() {
        let c = LlmConfig::default();
        assert_eq!(c.model, "minimax-m3");
        assert_eq!(c.provider, "openai_compat");
        assert!(c.dry_run, "default must be dry-run for safety");
    }

    #[test]
    fn live_config_drops_dry_run() {
        let c = LlmConfig::live();
        assert!(!c.dry_run);
        assert_eq!(c.model, "minimax-m3");
    }

    #[test]
    fn prompt_files_match_three_stage_loop() {
        assert_eq!(PromptTemplate::OnlineExplore.file_name(), "online_explore.md");
        assert_eq!(
            PromptTemplate::ReplayPolicyImprove.file_name(),
            "replay_policy_improve.md"
        );
        assert_eq!(PromptTemplate::GridPlan.file_name(), "grid_plan.md");
    }

    #[tokio::test]
    async fn client_without_api_key_stays_dry_run() {
        // Use an env name that cannot exist.
        std::env::remove_var("MINIMAX_API_KEY");
        let c = LlmClient::from_config(LlmConfig {
            api_key_env: "COGNI_TEST_NONEXISTENT_KEY_XYZ".into(),
            ..LlmConfig::default()
        });
        assert!(c.is_dry_run(), "missing key forces dry-run");
        let text = c.complete("hello").await.expect("stub returns");
        assert!(text.contains("<policy>"));
    }

    #[tokio::test]
    async fn client_with_dry_run_flag_stays_dry_run() {
        // Even with a fake key in env, dry_run=true wins.
        std::env::set_var("COGNI_TEST_FAKE_KEY", "fake");
        let c = LlmClient::from_config(LlmConfig {
            api_key_env: "COGNI_TEST_FAKE_KEY".into(),
            dry_run: true,
            ..LlmConfig::default()
        });
        assert!(c.is_dry_run());
        std::env::remove_var("COGNI_TEST_FAKE_KEY");
    }
}