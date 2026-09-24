//! Source adapters for the real SaaS system (read-only by default).
//!
//! Concrete impls land in PR #3 — see `docs/PROPOSAL.md` §4.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocLocator {
    pub path: String,
    pub section: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    pub locator: DocLocator,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLocator {
    pub repo: String,
    pub path: String,
    pub line_start: u32,
    pub line_end: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    pub locator: CodeLocator,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiLocator {
    pub method: String,
    pub url: String,
    pub query: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    pub locator: ApiLocator,
    pub status: u16,
    pub body: serde_json::Value,
}

#[async_trait]
pub trait SourceAdapter: Send + Sync {
    async fn read_doc(&self, locator: DocLocator) -> anyhow::Result<Vec<TextChunk>>;
    async fn read_code(&self, locator: CodeLocator) -> anyhow::Result<Vec<CodeChunk>>;
    async fn call_api(&self, locator: ApiLocator) -> anyhow::Result<ApiResponse>;
}

/// Whether the crawler is allowed to make state-changing calls.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_gate_defaults_to_deny() {
        // SAFETY: tests must not mutate process-wide env; rely on default
        // which is deny.
        assert!(!WriteGate::default().allow);
    }
}