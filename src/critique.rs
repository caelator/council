use serde::{Deserialize, Serialize};

/// The current schema version for all council artifacts.
/// Bump this when making breaking changes to serialized formats.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CritiquePoint {
    pub issue: String,
    pub severity: Severity,
    pub why_it_matters: String,
    pub suggested_delta: String,
    #[serde(default)]
    pub evidence: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    High,
    Medium,
    Low,
}

fn default_schema_version() -> u32 {
    CURRENT_SCHEMA_VERSION
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredCritique {
    pub reviewer_role: String,
    pub critiques: Vec<CritiquePoint>,
    #[serde(default)]
    pub things_to_keep: Vec<String>,
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Disposition {
    Accept,
    Partial,
    Reject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionEntry {
    pub issue: String,
    pub severity: Severity,
    pub disposition: Disposition,
    pub rationale: String,
    /// Which critic originated this issue (e.g. "claude", "gemini").
    /// Extracted from [model] tags in the decision log.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionLog {
    pub round: u32,
    pub decisions: Vec<DecisionEntry>,
    pub material_change: bool,
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
}

/// The context payload written as run-summary.json — the primary artifact
/// that downstream consumers (and resume mode) read to understand a council run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPayload {
    pub run_id: String,
    pub task: String,
    pub rounds: u32,
    pub converged: bool,
    pub council_dir: String,
    pub final_plan: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
}

/// Read and validate a ContextPayload from a JSON file (run-summary.json).
/// Handles schema version negotiation:
/// - Missing version field: assumes v1
/// - Version > CURRENT_SCHEMA_VERSION: warns but attempts best-effort parse
/// - Version == CURRENT_SCHEMA_VERSION: passes through
pub fn read_context_payload(path: &std::path::Path) -> Result<ContextPayload, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

    // First, try to extract the schema_version from raw JSON to warn early
    if let Ok(raw) = serde_json::from_str::<serde_json::Value>(&content) {
        match raw.get("schema_version") {
            None => {
                eprintln!(
                    "  [info] read_context_payload: no schema_version in {}, assuming v1",
                    path.display()
                );
            }
            Some(v) => {
                if let Some(ver) = v.as_u64() {
                    if ver > CURRENT_SCHEMA_VERSION as u64 {
                        eprintln!(
                            "  [warn] read_context_payload: schema_version {} in {} is newer than supported ({}), attempting best-effort parse",
                            ver,
                            path.display(),
                            CURRENT_SCHEMA_VERSION
                        );
                    }
                }
            }
        }
    }

    serde_json::from_str::<ContextPayload>(&content)
        .map_err(|e| format!("failed to parse {}: {}", path.display(), e))
}

/// Warn on unknown schema versions but don't fail — forward-compatible.
fn check_schema_version(sc: &StructuredCritique, reviewer_role: &str) {
    if sc.schema_version > CURRENT_SCHEMA_VERSION {
        eprintln!(
            "  [warn] parse_critique: schema_version {} for role '{}' is newer than supported ({}), proceeding anyway",
            sc.schema_version, reviewer_role, CURRENT_SCHEMA_VERSION
        );
    }
}

/// Warn on unknown schema versions in decision logs.
pub fn check_decision_log_version(log: &DecisionLog) {
    if log.schema_version > CURRENT_SCHEMA_VERSION {
        eprintln!(
            "  [warn] decision_log: schema_version {} (round {}) is newer than supported ({}), proceeding anyway",
            log.schema_version, log.round, CURRENT_SCHEMA_VERSION
        );
    }
}

/// Try to parse a structured critique from model output.
/// Uses a multi-strategy approach to handle markdown-fenced JSON, raw JSON, and
/// brace-matched JSON. Falls back to wrapping raw text only as a last resort.
pub fn parse_critique(raw: &str, reviewer_role: &str) -> StructuredCritique {
    // Strategy 1: Extract from ```json ... ``` code fence
    if let Some(json_str) = extract_fenced_json(raw) {
        if let Ok(mut sc) = serde_json::from_str::<StructuredCritique>(json_str) {
            if sc.reviewer_role.is_empty() {
                sc.reviewer_role = reviewer_role.to_string();
            }
            check_schema_version(&sc, reviewer_role);
            return sc;
        }
    }

    // Strategy 2: Try deserializing the entire raw text directly
    if let Ok(mut sc) = serde_json::from_str::<StructuredCritique>(raw.trim()) {
        if sc.reviewer_role.is_empty() {
            sc.reviewer_role = reviewer_role.to_string();
        }
        check_schema_version(&sc, reviewer_role);
        return sc;
    }

    // Strategy 3: Brace-matching to find outermost { ... } block
    if let Some(json_str) = extract_brace_json(raw) {
        if let Ok(mut sc) = serde_json::from_str::<StructuredCritique>(json_str) {
            if sc.reviewer_role.is_empty() {
                sc.reviewer_role = reviewer_role.to_string();
            }
            check_schema_version(&sc, reviewer_role);
            return sc;
        }
    }

    // All strategies failed — fall back to unstructured wrapper
    eprintln!(
        "  [warn] parse_critique: all JSON strategies failed for role '{}', falling back to unstructured wrapper",
        reviewer_role
    );
    StructuredCritique {
        reviewer_role: reviewer_role.to_string(),
        critiques: vec![CritiquePoint {
            issue: "Unstructured critique (model did not return valid JSON)".into(),
            severity: Severity::Medium,
            why_it_matters: "Review manually".into(),
            suggested_delta: raw.to_string(),
            evidence: String::new(),
        }],
        things_to_keep: vec![],
        schema_version: CURRENT_SCHEMA_VERSION,
    }
}

/// Check whether remaining critiques are all low severity (convergence signal).
/// Includes defense-in-depth: if the critique looks like a fallback wrapper but the
/// suggested_delta contains valid JSON with all-low critiques, use those severities.
pub fn is_converged(critique: &StructuredCritique) -> bool {
    // Fast path: normal parsed critique
    let all_low = critique
        .critiques
        .iter()
        .all(|c| matches!(c.severity, Severity::Low));

    if all_low {
        return true;
    }

    // Defense-in-depth: detect fallback wrapper and try to recover severities
    if critique.critiques.len() == 1
        && critique.critiques[0]
            .issue
            .starts_with("Unstructured critique")
    {
        let raw = &critique.critiques[0].suggested_delta;
        // Try all extraction strategies on the raw text embedded in the fallback
        let candidate = extract_fenced_json(raw)
            .and_then(|s| serde_json::from_str::<StructuredCritique>(s).ok())
            .or_else(|| serde_json::from_str::<StructuredCritique>(raw.trim()).ok())
            .or_else(|| {
                extract_brace_json(raw)
                    .and_then(|s| serde_json::from_str::<StructuredCritique>(s).ok())
            });

        if let Some(sc) = candidate {
            return sc
                .critiques
                .iter()
                .all(|c| matches!(c.severity, Severity::Low));
        }
    }

    false
}

/// Parse decision entries from the Codex deliberation/revision output.
/// Looks for lines matching: `- ACCEPT|PARTIAL|REJECT: [model] <issue> -- <rationale>`
/// The `[model]` tag is optional for backward compatibility.
///
/// Handles format variations: bold markers (**ACCEPT**:), numbered lists (1.),
/// asterisk bullets (*), and various whitespace/punctuation around the disposition keyword.
pub fn parse_decision_log(revision_text: &str) -> Vec<DecisionEntry> {
    let mut entries = Vec::new();
    // Track current section source: lines under "### Claude" or "### Gemini" headers
    let mut section_source: Option<String> = None;

    for line in revision_text.lines() {
        // Detect section headers like "### Claude (conceptual-risk)" or "### Gemini"
        let trimmed_line = line.trim();
        if trimmed_line.starts_with("###") || trimmed_line.starts_with("**##") {
            let header = trimmed_line
                .trim_start_matches('#')
                .trim_start_matches('*')
                .trim()
                .to_lowercase();
            if header.starts_with("claude") || header.contains("(conceptual") {
                section_source = Some("claude".into());
            } else if header.starts_with("gemini") || header.contains("(implementation") {
                section_source = Some("gemini".into());
            } else if header.starts_with("codex") {
                section_source = Some("codex".into());
            }
            continue;
        }

        // Normalize the line: strip list markers, bold wrappers, etc.
        let normalized = normalize_decision_line(trimmed_line);
        let trimmed = normalized.as_str();

        // Match ACCEPT: / PARTIAL: / REJECT: at the start
        let (disposition, rest) = if let Some(rest) = strip_prefix_ci(trimmed, "ACCEPT:") {
            (Disposition::Accept, rest)
        } else if let Some(rest) = strip_prefix_ci(trimmed, "PARTIAL:") {
            (Disposition::Partial, rest)
        } else if let Some(rest) = strip_prefix_ci(trimmed, "REJECT:") {
            (Disposition::Reject, rest)
        } else if let Some(rest) = strip_prefix_ci(trimmed, "ACCEPT -") {
            (Disposition::Accept, rest)
        } else if let Some(rest) = strip_prefix_ci(trimmed, "PARTIAL -") {
            (Disposition::Partial, rest)
        } else if let Some(rest) = strip_prefix_ci(trimmed, "REJECT -") {
            (Disposition::Reject, rest)
        } else {
            continue;
        };

        let rest = rest.trim();

        // Extract optional [model] tag at the start of the issue
        let (source_model, rest) = extract_source_tag(rest, &section_source);

        // Split on " -- " or " — " (em-dash) or " - " for issue/rationale separation
        let (issue, rationale) = split_issue_rationale(rest);

        if issue.is_empty() {
            continue;
        }

        // Infer severity from disposition (REJECT = high, PARTIAL = medium, ACCEPT = low)
        let severity = match disposition {
            Disposition::Reject => Severity::High,
            Disposition::Partial => Severity::Medium,
            Disposition::Accept => Severity::Low,
        };

        entries.push(DecisionEntry {
            issue,
            severity,
            disposition,
            rationale,
            source_model,
        });
    }
    entries
}

/// Normalize a decision log line by stripping list markers and bold wrappers.
/// Produces an owned String so we can freely manipulate the text.
///
/// Examples:
/// - `"- **ACCEPT**: issue"` -> `"ACCEPT: issue"`
/// - `"1. ACCEPT: issue"` -> `"ACCEPT: issue"`
/// - `"* **PARTIAL:** issue"` -> `"PARTIAL: issue"`
fn normalize_decision_line(line: &str) -> String {
    let mut s = line.trim().to_string();

    // Strip leading list markers: "- ", "* ", "1. ", "1) "
    if s.starts_with("- ") || s.starts_with("* ") {
        s = s[2..].trim_start().to_string();
    } else {
        // Check for numbered list: "1. " or "1) "
        let digit_end = s.bytes().take_while(|b| b.is_ascii_digit()).count();
        if digit_end > 0 && digit_end < s.len() {
            let next = s.as_bytes()[digit_end];
            if (next == b'.' || next == b')') && s.len() > digit_end + 1 {
                s = s[digit_end + 1..].trim_start().to_string();
            }
        }
    }

    // Strip bold wrappers: "**ACCEPT**:" -> "ACCEPT:", "**ACCEPT:**" -> "ACCEPT:"
    if s.starts_with("**") {
        if let Some(end) = s[2..].find("**") {
            let inner = &s[2..2 + end];
            let kw = inner.trim_end_matches(':');
            if kw.eq_ignore_ascii_case("ACCEPT")
                || kw.eq_ignore_ascii_case("PARTIAL")
                || kw.eq_ignore_ascii_case("REJECT")
            {
                let after = s[2 + end + 2..].to_string();
                if inner.ends_with(':') {
                    // "**ACCEPT:**rest" -> "ACCEPT:rest"
                    s = format!("{}{}", inner, after);
                } else if after.starts_with(':') {
                    // "**ACCEPT**:rest" -> "ACCEPT:rest"
                    s = format!("{}{}", kw, after);
                }
            }
        }
    }

    s
}

/// Split "issue -- rationale" or "issue — rationale" or "issue - rationale".
fn split_issue_rationale(s: &str) -> (String, String) {
    // Try " -- " first (most common/expected)
    if let Some(idx) = s.find(" -- ") {
        return (s[..idx].trim().to_string(), s[idx + 4..].trim().to_string());
    }
    // Try em-dash " — "
    if let Some(idx) = s.find(" \u{2014} ") {
        let em_len = " \u{2014} ".len();
        return (s[..idx].trim().to_string(), s[idx + em_len..].trim().to_string());
    }
    // Try single dash " - " (last resort, may be ambiguous)
    if let Some(idx) = s.find(" - ") {
        return (s[..idx].trim().to_string(), s[idx + 3..].trim().to_string());
    }
    (s.to_string(), String::new())
}

/// Extract a `[model]` tag from the start of text, falling back to section source.
fn extract_source_tag<'a>(
    text: &'a str,
    section_source: &Option<String>,
) -> (Option<String>, &'a str) {
    if text.starts_with('[') {
        if let Some(end) = text.find(']') {
            let tag = text[1..end].trim().to_lowercase();
            if tag == "claude" || tag == "gemini" || tag == "codex" {
                return (Some(tag), text[end + 1..].trim());
            }
        }
    }
    (section_source.clone(), text)
}

/// Case-insensitive prefix strip: returns the remainder if `text` starts with `prefix` (ignoring case).
fn strip_prefix_ci<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    if text.len() >= prefix.len() && text[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&text[prefix.len()..])
    } else {
        None
    }
}

/// Extract JSON from a ```json ... ``` fenced code block.
fn extract_fenced_json(text: &str) -> Option<&str> {
    let start = text.find("```json")?;
    let content_start = start + 7;
    // Skip optional newline after ```json
    let content_start = if text[content_start..].starts_with('\n') {
        content_start + 1
    } else {
        content_start
    };
    let end = text[content_start..].find("```")?;
    Some(text[content_start..content_start + end].trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> String {
        r#"{"reviewer_role":"test","critiques":[{"issue":"test issue","severity":"low","why_it_matters":"reason","suggested_delta":"fix it","evidence":""}],"things_to_keep":["good stuff"]}"#.to_string()
    }

    #[test]
    fn parse_critique_from_fenced_json() {
        let raw = format!("Here is my review:\n```json\n{}\n```\nDone.", sample_json());
        let result = parse_critique(&raw, "test-role");
        assert_eq!(result.critiques.len(), 1);
        assert_eq!(result.critiques[0].issue, "test issue");
        assert!(matches!(result.critiques[0].severity, Severity::Low));
    }

    #[test]
    fn parse_critique_from_raw_json() {
        let raw = sample_json();
        let result = parse_critique(&raw, "test-role");
        assert_eq!(result.critiques.len(), 1);
        assert_eq!(result.critiques[0].issue, "test issue");
    }

    #[test]
    fn parse_critique_from_brace_match() {
        let raw = format!("Some preamble text\n{}\nSome trailing text", sample_json());
        let result = parse_critique(&raw, "test-role");
        assert_eq!(result.critiques.len(), 1);
        assert_eq!(result.critiques[0].issue, "test issue");
    }

    #[test]
    fn parse_critique_fallback_on_garbage() {
        let result = parse_critique("This is not JSON at all", "test-role");
        assert_eq!(result.critiques.len(), 1);
        assert!(
            result.critiques[0]
                .issue
                .starts_with("Unstructured critique")
        );
    }

    #[test]
    fn is_converged_all_low() {
        let sc = StructuredCritique {
            reviewer_role: "test".into(),
            critiques: vec![CritiquePoint {
                issue: "minor".into(),
                severity: Severity::Low,
                why_it_matters: "".into(),
                suggested_delta: "".into(),
                evidence: "".into(),
            }],
            things_to_keep: vec![],
            schema_version: 1,
        };
        assert!(is_converged(&sc));
    }

    #[test]
    fn is_converged_with_high() {
        let sc = StructuredCritique {
            reviewer_role: "test".into(),
            critiques: vec![CritiquePoint {
                issue: "big".into(),
                severity: Severity::High,
                why_it_matters: "".into(),
                suggested_delta: "".into(),
                evidence: "".into(),
            }],
            things_to_keep: vec![],
            schema_version: 1,
        };
        assert!(!is_converged(&sc));
    }

    #[test]
    fn is_converged_defense_in_depth_recovers_from_fallback() {
        // Simulate a fallback wrapper where the raw text actually contains valid all-low JSON
        let inner_json = r#"{"reviewer_role":"test","critiques":[{"issue":"minor","severity":"low","why_it_matters":"","suggested_delta":"","evidence":""}],"things_to_keep":[]}"#;
        let sc = StructuredCritique {
            reviewer_role: "test".into(),
            critiques: vec![CritiquePoint {
                issue: "Unstructured critique (model did not return valid JSON)".into(),
                severity: Severity::Medium,
                why_it_matters: "Review manually".into(),
                suggested_delta: inner_json.to_string(),
                evidence: String::new(),
            }],
            things_to_keep: vec![],
            schema_version: 1,
        };
        assert!(is_converged(&sc));
    }

    #[test]
    fn parse_decision_log_extracts_entries() {
        let text = r#"## Decision Log
- ACCEPT: Missing error handling -- Already addressed in revised plan
- PARTIAL: Complexity in auth flow -- Simplified but kept core structure
- REJECT: Remove caching entirely -- Caching is essential for performance

## Revised Plan
..."#;
        let entries = parse_decision_log(text);
        assert_eq!(entries.len(), 3);
        assert!(matches!(entries[0].disposition, Disposition::Accept));
        assert_eq!(entries[0].issue, "Missing error handling");
        assert_eq!(entries[0].rationale, "Already addressed in revised plan");
        assert!(matches!(entries[1].disposition, Disposition::Partial));
        assert!(matches!(entries[2].disposition, Disposition::Reject));
        assert_eq!(entries[2].issue, "Remove caching entirely");
    }

    #[test]
    fn parse_decision_log_extracts_source_tags() {
        let text = r#"## Decision Log
### Claude (conceptual-risk)
- ACCEPT: Missing error handling -- Already addressed
- REJECT: Coupling issue -- Not a real problem

### Gemini (implementation-risk)
- PARTIAL: Scaling concern -- Added connection pool

## Revised Plan
..."#;
        let entries = parse_decision_log(text);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].source_model.as_deref(), Some("claude"));
        assert_eq!(entries[1].source_model.as_deref(), Some("claude"));
        assert_eq!(entries[2].source_model.as_deref(), Some("gemini"));
    }

    #[test]
    fn parse_decision_log_extracts_inline_tags() {
        let text = r#"## Decision Log
- ACCEPT: [claude] Missing error handling -- Fixed
- PARTIAL: [gemini] Scaling issue -- Added pool
"#;
        let entries = parse_decision_log(text);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].source_model.as_deref(), Some("claude"));
        assert_eq!(entries[0].issue, "Missing error handling");
        assert_eq!(entries[1].source_model.as_deref(), Some("gemini"));
    }

    #[test]
    fn parse_decision_log_empty_on_no_entries() {
        let entries = parse_decision_log("No decisions here, just text.");
        assert!(entries.is_empty());
    }

    #[test]
    fn parse_decision_log_bold_markers() {
        let text = r#"## Decision Log
### Claude (conceptual-risk)
- **ACCEPT**: Missing error handling -- Fixed in revised plan
- **PARTIAL:** Coupling issue -- Scoped to module boundary
- **REJECT**: Remove caching -- Essential for performance

## Revised Plan
..."#;
        let entries = parse_decision_log(text);
        assert_eq!(entries.len(), 3);
        assert!(matches!(entries[0].disposition, Disposition::Accept));
        assert!(matches!(entries[1].disposition, Disposition::Partial));
        assert!(matches!(entries[2].disposition, Disposition::Reject));
        assert_eq!(entries[0].source_model.as_deref(), Some("claude"));
    }

    #[test]
    fn parse_decision_log_numbered_list() {
        let text = r#"## Decision Log
### Gemini (implementation-risk)
1. ACCEPT: No rate limiting -- Added basic rate limiter
2. PARTIAL: Missing retry logic -- Added timeout only

## Revised Plan
..."#;
        let entries = parse_decision_log(text);
        assert_eq!(entries.len(), 2);
        assert!(matches!(entries[0].disposition, Disposition::Accept));
        assert_eq!(entries[0].issue, "No rate limiting");
        assert_eq!(entries[0].source_model.as_deref(), Some("gemini"));
    }

    #[test]
    fn parse_decision_log_asterisk_bullets() {
        let text = "* ACCEPT: Issue one -- Rationale one\n* REJECT: Issue two -- Rationale two";
        let entries = parse_decision_log(text);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn parse_decision_log_em_dash_separator() {
        let text = "- ACCEPT: Missing validation — Added input checks";
        let entries = parse_decision_log(text);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].issue, "Missing validation");
        assert_eq!(entries[0].rationale, "Added input checks");
    }

    #[test]
    fn parse_decision_log_accept_dash_format() {
        let text = "- ACCEPT - Missing validation -- Added input checks";
        let entries = parse_decision_log(text);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].issue, "Missing validation");
    }

    #[test]
    fn extract_fenced_json_handles_newline_after_fence() {
        let text = "```json\n{\"key\": \"value\"}\n```";
        let result = extract_fenced_json(text);
        assert_eq!(result, Some("{\"key\": \"value\"}"));
    }

    #[test]
    fn parse_critique_without_version_defaults_to_1() {
        let raw = sample_json(); // sample_json has no schema_version field
        let result = parse_critique(&raw, "test-role");
        assert_eq!(result.schema_version, 1);
    }

    #[test]
    fn parse_critique_with_version_1_works() {
        let raw = r#"{"reviewer_role":"test","schema_version":1,"critiques":[{"issue":"test issue","severity":"low","why_it_matters":"reason","suggested_delta":"fix it","evidence":""}],"things_to_keep":["good stuff"]}"#;
        let result = parse_critique(raw, "test-role");
        assert_eq!(result.schema_version, 1);
        assert_eq!(result.critiques.len(), 1);
        assert_eq!(result.critiques[0].issue, "test issue");
    }

    #[test]
    fn parse_critique_with_unknown_version_warns_but_succeeds() {
        let raw = r#"{"reviewer_role":"test","schema_version":99,"critiques":[{"issue":"future issue","severity":"low","why_it_matters":"reason","suggested_delta":"fix it","evidence":""}],"things_to_keep":[]}"#;
        let result = parse_critique(raw, "test-role");
        assert_eq!(result.schema_version, 99);
        assert_eq!(result.critiques.len(), 1);
        assert_eq!(result.critiques[0].issue, "future issue");
    }

    #[test]
    fn decision_log_without_version_defaults_to_1() {
        let json = r#"{"round":1,"decisions":[],"material_change":false}"#;
        let log: DecisionLog = serde_json::from_str(json).unwrap();
        assert_eq!(log.schema_version, 1);
    }

    #[test]
    fn decision_log_with_version_works() {
        let json = r#"{"round":1,"decisions":[],"material_change":false,"schema_version":1}"#;
        let log: DecisionLog = serde_json::from_str(json).unwrap();
        assert_eq!(log.schema_version, 1);
    }

    #[test]
    fn extract_brace_json_handles_strings_with_braces() {
        let text = r#"prefix {"key": "value with { brace }"} suffix"#;
        let result = extract_brace_json(text);
        assert_eq!(result, Some(r#"{"key": "value with { brace }"}"#));
    }

    // -- ContextPayload schema negotiation tests --

    #[test]
    fn context_payload_without_version_defaults_to_current() {
        let json = r#"{"run_id":"r1","task":"t","rounds":1,"converged":true,"council_dir":"/tmp","final_plan":"/tmp/p.md"}"#;
        let payload: ContextPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.schema_version, CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn context_payload_with_current_version_parses() {
        let json = format!(
            r#"{{"run_id":"r1","task":"t","rounds":1,"converged":true,"council_dir":"/tmp","final_plan":"/tmp/p.md","schema_version":{}}}"#,
            CURRENT_SCHEMA_VERSION
        );
        let payload: ContextPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(payload.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(payload.run_id, "r1");
    }

    #[test]
    fn context_payload_with_future_version_parses_best_effort() {
        let json = r#"{"run_id":"r1","task":"t","rounds":2,"converged":false,"council_dir":"/tmp","final_plan":"/tmp/p.md","schema_version":99,"unknown_field":"ignored"}"#;
        // serde ignores unknown fields by default, so future payloads parse fine
        let payload: ContextPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.schema_version, 99);
        assert_eq!(payload.rounds, 2);
    }

    #[test]
    fn read_context_payload_from_file() {
        let dir = std::env::temp_dir().join(format!("council-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("run-summary.json");
        std::fs::write(&path, r#"{"run_id":"test","task":"t","rounds":1,"converged":true,"council_dir":"/tmp","final_plan":"/tmp/p.md"}"#).unwrap();

        let payload = read_context_payload(&path).unwrap();
        assert_eq!(payload.run_id, "test");
        assert_eq!(payload.schema_version, CURRENT_SCHEMA_VERSION);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_context_payload_missing_file_errors() {
        let path = std::path::PathBuf::from("/tmp/nonexistent-council-test/summary.json");
        let result = read_context_payload(&path);
        assert!(result.is_err());
    }

    #[test]
    fn current_schema_version_constant_is_1() {
        assert_eq!(CURRENT_SCHEMA_VERSION, 1);
    }

    #[test]
    fn decision_log_with_future_version_parses() {
        let json = r#"{"round":1,"decisions":[],"material_change":false,"schema_version":42}"#;
        let log: DecisionLog = serde_json::from_str(json).unwrap();
        assert_eq!(log.schema_version, 42);
    }
}

/// Extract JSON by brace-matching the outermost { ... } block.
fn extract_brace_json(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;
    for (i, ch) in text[start..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}
