//! Placeholder module: CognitionQuestion will be expanded alongside the
//! dialogue protocol in PR #3. Keeping it here so `cogni-policy` can compile.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CognitionQuestion {
    pub daily_question: String,
    pub boundary_hint: Vec<String>,
    pub session_id: String,
}