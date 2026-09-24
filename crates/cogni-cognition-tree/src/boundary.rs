//! Boundary specification and domain matching.
//!
//! Mirrors `docs/PROPOSAL.md` §7.1 `boundary.yaml`:
//! Users declare SaaS domain boundaries (modules and key business surfaces).
//! The cognition system matches tree nodes against these boundaries to compute
//! boundary mastery (`|frontier ∩ boundary| / |frontier|`).

use std::path::Path;

use anyhow::Context;
use cogni_core::{CognitionNode, NodeKind};
use serde::{Deserialize, Serialize};

/// A single business or technical domain boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BoundaryDomain {
    pub id: String,
    #[serde(default)]
    pub modules: Vec<String>,
    #[serde(default)]
    pub surfaces: Vec<String>,
}

impl BoundaryDomain {
    pub fn new(
        id: impl Into<String>,
        modules: Vec<String>,
        surfaces: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            modules,
            surfaces,
        }
    }

    /// Check if a cognition node falls inside this domain boundary.
    pub fn matches(&self, node: &CognitionNode) -> bool {
        // 1. Tag match (case-insensitive or exact)
        for tag in &node.tags {
            let t = tag.0.to_lowercase();
            if self
                .modules
                .iter()
                .any(|m| t == m.to_lowercase() || t.contains(&m.to_lowercase()))
            {
                return true;
            }
            if self
                .surfaces
                .iter()
                .any(|s| t == s.to_lowercase() || t.contains(&s.to_lowercase()))
            {
                return true;
            }
        }

        // 2. Title match
        let title_lower = node.title.to_lowercase();
        if self
            .modules
            .iter()
            .any(|m| title_lower.contains(&m.to_lowercase()))
        {
            return true;
        }
        if self
            .surfaces
            .iter()
            .any(|s| node.title.contains(s) || title_lower.contains(&s.to_lowercase()))
        {
            return true;
        }

        // 3. Evidence source match
        for ev in &node.evidence {
            let src_lower = ev.source.to_lowercase();
            let loc_lower = ev.locator.to_lowercase();
            if self.modules.iter().any(|m| {
                let m_lower = m.to_lowercase();
                src_lower.contains(&m_lower) || loc_lower.contains(&m_lower)
            }) {
                return true;
            }
            if self.surfaces.iter().any(|s| {
                ev.source.contains(s) || ev.locator.contains(s)
            }) {
                return true;
            }
        }

        false
    }
}

/// Boundary specification, typically loaded from `boundary.yaml` or embedded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BoundarySpec {
    pub boundary: Vec<BoundaryDomain>,
}

impl BoundarySpec {
    /// Built-in SaaS financial and business integration boundary from `docs/PROPOSAL.md` §7.1.
    pub fn default_saas() -> Self {
        Self {
            boundary: vec![
                BoundaryDomain {
                    id: "finance-core".into(),
                    modules: vec![
                        "gl".into(),
                        "ap".into(),
                        "ar".into(),
                        "fa".into(),
                        "banking".into(),
                        "tax".into(),
                    ],
                    surfaces: vec![
                        "结账".into(),
                        "过账".into(),
                        "对账".into(),
                        "票据".into(),
                        "凭证".into(),
                        "报表".into(),
                    ],
                },
                BoundaryDomain {
                    id: "biz-core".into(),
                    modules: vec![
                        "inventory".into(),
                        "sales".into(),
                        "purchase".into(),
                        "crm".into(),
                    ],
                    surfaces: vec![
                        "订单".into(),
                        "出入库".into(),
                        "客户".into(),
                        "报价".into(),
                    ],
                },
                BoundaryDomain {
                    id: "integration".into(),
                    modules: vec![
                        "gl↔biz".into(),
                        "gl↔bank".into(),
                        "biz↔tax".into(),
                        "integration".into(),
                    ],
                    surfaces: vec![
                        "过账接口".into(),
                        "银行对账单".into(),
                        "发票开具".into(),
                        "post".into(),
                        "bill".into(),
                    ],
                },
            ],
        }
    }

    /// Check if a node belongs to any declared boundary domain, has explicit
    /// boundary fit (>= 0.5), or is a dedicated Boundary kind.
    pub fn contains_node(&self, node: &CognitionNode) -> bool {
        if node.kind == NodeKind::Boundary {
            return true;
        }
        if node.score.boundary_fit >= 0.5 {
            return true;
        }
        self.boundary.iter().any(|d| d.matches(node))
    }

    /// Return all domain IDs that match the given node.
    pub fn matching_domains<'a>(&'a self, node: &CognitionNode) -> Vec<&'a str> {
        self.boundary
            .iter()
            .filter(|d| d.matches(node))
            .map(|d| d.id.as_str())
            .collect()
    }

    /// Parse from YAML string.
    pub fn from_yaml_str(s: &str) -> anyhow::Result<Self> {
        serde_yaml::from_str(s).context("failed to parse boundary yaml")
    }

    /// Parse from JSON string.
    pub fn from_json_str(s: &str) -> anyhow::Result<Self> {
        serde_json::from_str(s).context("failed to parse boundary json")
    }

    /// Load from file, determining format by extension (.yaml/.yml vs .json).
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let p = path.as_ref();
        let content = std::fs::read_to_string(p)
            .with_context(|| format!("failed to read boundary file: {}", p.display()))?;
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ext == "json" {
            Self::from_json_str(&content)
        } else {
            Self::from_yaml_str(&content)
        }
    }
}

impl Default for BoundarySpec {
    fn default() -> Self {
        Self::default_saas()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use cogni_core::{EvidenceRef, FailureClass, NodeKind, Score, Tag};
    use serde_json::json;
    use ulid::Ulid;

    fn make_node(
        title: &str,
        kind: NodeKind,
        boundary_fit: f32,
        tags: Vec<&str>,
        evidence: Vec<(&str, &str)>,
    ) -> CognitionNode {
        CognitionNode {
            id: Ulid::new(),
            parent: None,
            iteration: 0,
            kind,
            title: title.into(),
            workspace: json!({}),
            observation: "".into(),
            score: Score {
                understanding: 0.8,
                boundary_fit,
                reuse_potential: 0.5,
                confidence: 0.9,
            },
            tags: tags.into_iter().map(|t| Tag(t.into())).collect(),
            evidence: evidence
                .into_iter()
                .map(|(s, l)| EvidenceRef {
                    source: s.into(),
                    locator: l.into(),
                })
                .collect(),
            failure_class: FailureClass::Ok,
            valid: true,
            no_total: false,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn parse_proposal_boundary_yaml() {
        let yaml = r#"
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
"#;
        let spec = BoundarySpec::from_yaml_str(yaml).expect("parse yaml");
        assert_eq!(spec.boundary.len(), 3);
        assert_eq!(spec.boundary[0].id, "finance-core");
        assert!(spec.boundary[0].modules.contains(&"gl".to_string()));
        assert!(spec.boundary[0].surfaces.contains(&"结账".to_string()));
        assert_eq!(spec.boundary[1].id, "biz-core");
        assert_eq!(spec.boundary[2].id, "integration");
    }

    #[test]
    fn domain_matches_node_by_tag() {
        let spec = BoundarySpec::default_saas();
        let n = make_node("read gl config", NodeKind::Code, 0.0, vec!["gl"], vec![]);
        assert!(spec.contains_node(&n));
        assert_eq!(spec.matching_domains(&n), vec!["finance-core"]);
    }

    #[test]
    fn domain_matches_node_by_surface_in_title() {
        let spec = BoundarySpec::default_saas();
        let n = make_node("测试银行对账单自动化", NodeKind::Document, 0.0, vec![], vec![]);
        assert!(spec.contains_node(&n));
        let domains = spec.matching_domains(&n);
        assert!(domains.contains(&"integration"));
        assert!(domains.contains(&"finance-core")); // Matches "对账" in finance-core surfaces
    }

    #[test]
    fn domain_matches_node_by_evidence_source() {
        let spec = BoundarySpec::default_saas();
        let n = make_node(
            "internal handler",
            NodeKind::Code,
            0.0,
            vec![],
            vec![("git:src/modules/inventory/stock.rs", "L10-L40")],
        );
        assert!(spec.contains_node(&n));
        assert_eq!(spec.matching_domains(&n), vec!["biz-core"]);
    }

    #[test]
    fn boundary_fit_and_boundary_kind_bypass_domain_text_match() {
        let spec = BoundarySpec { boundary: vec![] };
        let n1 = make_node("arbitrary", NodeKind::Concept, 0.8, vec![], vec![]);
        assert!(spec.contains_node(&n1));

        let n2 = make_node("external limit", NodeKind::Boundary, 0.0, vec![], vec![]);
        assert!(spec.contains_node(&n2));

        let n3 = make_node("unrelated", NodeKind::Concept, 0.2, vec![], vec![]);
        assert!(!spec.contains_node(&n3));
    }
}
