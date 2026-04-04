use crate::critique::{Disposition, Severity, StructuredCritique};
use crate::model::Model;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ── Per-run, per-model, per-role metrics ─────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    pub run_id: String,
    pub timestamp: String,
    pub model: String,
    pub role: String,
    pub phase: String,

    // Critique metrics (populated when model acts as a critic)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub critiques_submitted: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub critiques_accepted: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub critiques_partial: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub critiques_rejected: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high_severity_submitted: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high_severity_accepted: Option<u32>,

    // Output quality
    pub output_bytes: u32,
    pub parse_success: bool,
}

impl PhaseMetrics {
    /// Acceptance rate: fraction of critiques that were ACCEPT or PARTIAL (not REJECT).
    pub fn acceptance_rate(&self) -> Option<f64> {
        let submitted = self.critiques_submitted? as f64;
        if submitted == 0.0 {
            return Some(1.0);
        }
        let rejected = self.critiques_rejected.unwrap_or(0) as f64;
        Some((submitted - rejected) / submitted)
    }

    /// Severity accuracy: fraction of HIGH critiques that were accepted (not rejected).
    /// Measures whether the model's HIGH flags are actually warranted.
    pub fn severity_accuracy(&self) -> Option<f64> {
        let high = self.high_severity_submitted? as f64;
        if high == 0.0 {
            return Some(1.0); // no high critiques = no overinflation
        }
        let accepted = self.high_severity_accepted.unwrap_or(0) as f64;
        Some(accepted / high)
    }
}

// ── Aggregated model profile ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ModelProfile {
    pub model: String,
    pub runs_analyzed: u32,
    pub role_scores: Vec<RoleScore>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoleScore {
    pub role: String,
    pub phase: String,
    pub observations: u32,
    pub acceptance_rate: f64,
    pub severity_accuracy: f64,
    pub parse_reliability: f64,
    /// Composite score (0.0-1.0) combining all signals.
    pub composite: f64,
}

/// Build aggregated profiles from historical metrics.
pub fn aggregate_profiles(metrics: &[PhaseMetrics]) -> Vec<ModelProfile> {
    // Group by model
    let mut by_model: HashMap<String, Vec<&PhaseMetrics>> = HashMap::new();
    for m in metrics {
        by_model.entry(m.model.clone()).or_default().push(m);
    }

    let mut profiles = Vec::new();
    for (model, entries) in &by_model {
        // Group by (role, phase)
        let mut by_role: HashMap<(String, String), Vec<&PhaseMetrics>> = HashMap::new();
        for e in entries {
            by_role
                .entry((e.role.clone(), e.phase.clone()))
                .or_default()
                .push(e);
        }

        let mut role_scores = Vec::new();
        for ((role, phase), role_entries) in &by_role {
            let n = role_entries.len() as u32;

            let acceptance_rates: Vec<f64> = role_entries
                .iter()
                .filter_map(|e| e.acceptance_rate())
                .collect();
            let avg_acceptance = if acceptance_rates.is_empty() {
                0.5
            } else {
                acceptance_rates.iter().sum::<f64>() / acceptance_rates.len() as f64
            };

            let severity_accs: Vec<f64> = role_entries
                .iter()
                .filter_map(|e| e.severity_accuracy())
                .collect();
            let avg_severity_acc = if severity_accs.is_empty() {
                0.5
            } else {
                severity_accs.iter().sum::<f64>() / severity_accs.len() as f64
            };

            let parse_success_count = role_entries.iter().filter(|e| e.parse_success).count();
            let parse_reliability = parse_success_count as f64 / n as f64;

            // Composite: weighted blend
            // - Critics: acceptance_rate (40%) + severity_accuracy (40%) + parse (20%)
            // - Non-critics: parse reliability dominates
            let composite = if role.contains("adversar") || role.contains("critic") {
                avg_acceptance * 0.4 + avg_severity_acc * 0.4 + parse_reliability * 0.2
            } else {
                avg_acceptance * 0.2 + parse_reliability * 0.8
            };

            role_scores.push(RoleScore {
                role: role.clone(),
                phase: phase.clone(),
                observations: n,
                acceptance_rate: round2(avg_acceptance),
                severity_accuracy: round2(avg_severity_acc),
                parse_reliability: round2(parse_reliability),
                composite: round2(composite),
            });
        }

        role_scores.sort_by(|a, b| b.composite.partial_cmp(&a.composite).unwrap_or(std::cmp::Ordering::Equal));

        profiles.push(ModelProfile {
            model: model.clone(),
            runs_analyzed: entries.len() as u32,
            role_scores,
        });
    }

    profiles.sort_by(|a, b| a.model.cmp(&b.model));
    profiles
}

// ── Role recommendations ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct RoleRecommendation {
    pub phase: String,
    pub current_model: String,
    pub current_role: String,
    pub recommended_model: String,
    pub confidence: f64,
    pub reason: String,
}

/// Compare model profiles and recommend role swaps where the data supports it.
/// Only recommends swaps with >= `min_observations` data points and >= `min_confidence`.
pub fn recommend_roles(
    profiles: &[ModelProfile],
    current_assignments: &[(Model, &str, &str)], // (model, role, phase)
    min_observations: u32,
    min_confidence: f64,
) -> Vec<RoleRecommendation> {
    let mut recommendations = Vec::new();

    for &(current_model, role, phase) in current_assignments {
        let current_name = current_model.name();

        // Find current model's score for this role
        let current_score = find_role_score(profiles, current_name, role);

        // Check if any other model scores higher on this role
        for profile in profiles {
            if profile.model == current_name {
                continue;
            }
            if let Some(candidate_score) = profile
                .role_scores
                .iter()
                .find(|s| s.role == role && s.phase == phase)
            {
                if candidate_score.observations < min_observations {
                    continue;
                }

                let current_composite = current_score.map(|s| s.composite).unwrap_or(0.5);
                let delta = candidate_score.composite - current_composite;

                // Only recommend if meaningful improvement (>10% delta)
                if delta > 0.10 {
                    let confidence = (delta * 2.0).min(1.0); // scale delta to confidence
                    if confidence >= min_confidence {
                        recommendations.push(RoleRecommendation {
                            phase: phase.into(),
                            current_model: current_name.into(),
                            current_role: role.into(),
                            recommended_model: profile.model.clone(),
                            confidence: round2(confidence),
                            reason: format!(
                                "{} scores {:.0}% vs {}'s {:.0}% on {} (acceptance: {:.0}% vs {:.0}%, severity accuracy: {:.0}% vs {:.0}%)",
                                profile.model,
                                candidate_score.composite * 100.0,
                                current_name,
                                current_composite * 100.0,
                                role,
                                candidate_score.acceptance_rate * 100.0,
                                current_score.map(|s| s.acceptance_rate).unwrap_or(0.5) * 100.0,
                                candidate_score.severity_accuracy * 100.0,
                                current_score.map(|s| s.severity_accuracy).unwrap_or(0.5) * 100.0,
                            ),
                        });
                    }
                }
            }
        }
    }

    recommendations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    recommendations
}

fn find_role_score<'a>(
    profiles: &'a [ModelProfile],
    model: &str,
    role: &str,
) -> Option<&'a RoleScore> {
    profiles
        .iter()
        .find(|p| p.model == model)?
        .role_scores
        .iter()
        .find(|s| s.role == role)
}

// ── Metrics collection helpers ───────────────────────────────────────

/// Build critique metrics for a model that acted as critic this round.
/// Matches the critic's issues to decisions by the synthesis owner.
pub fn collect_critique_metrics(
    run_id: &str,
    timestamp: &str,
    model: Model,
    role: &str,
    phase: &str,
    critique: &StructuredCritique,
    decisions: &[(String, Disposition, Severity)], // (issue_text, disposition, original_severity)
    output_bytes: u32,
    parse_success: bool,
) -> PhaseMetrics {
    let submitted = critique.critiques.len() as u32;
    let high_submitted = critique
        .critiques
        .iter()
        .filter(|c| matches!(c.severity, Severity::High))
        .count() as u32;

    // Match decisions to this critic's issues via fuzzy substring matching
    let mut accepted = 0u32;
    let mut partial = 0u32;
    let mut rejected = 0u32;
    let mut high_accepted = 0u32;

    for cp in &critique.critiques {
        if let Some((_, disposition, _)) = decisions.iter().find(|(issue, _, _)| {
            fuzzy_match(&cp.issue, issue)
        }) {
            match disposition {
                Disposition::Accept => {
                    accepted += 1;
                    if matches!(cp.severity, Severity::High) {
                        high_accepted += 1;
                    }
                }
                Disposition::Partial => {
                    partial += 1;
                    if matches!(cp.severity, Severity::High) {
                        high_accepted += 1; // partial counts as validated
                    }
                }
                Disposition::Reject => {
                    rejected += 1;
                }
            }
        }
        // Unmatched critiques are not counted as rejected — data gap, not failure
    }

    PhaseMetrics {
        run_id: run_id.into(),
        timestamp: timestamp.into(),
        model: model.name().into(),
        role: role.into(),
        phase: phase.into(),
        critiques_submitted: Some(submitted),
        critiques_accepted: Some(accepted),
        critiques_partial: Some(partial),
        critiques_rejected: Some(rejected),
        high_severity_submitted: Some(high_submitted),
        high_severity_accepted: Some(high_accepted),
        output_bytes,
        parse_success,
    }
}

/// Build basic output metrics for non-critic roles (framing, brainstorm, synthesis, handoff).
pub fn collect_output_metrics(
    run_id: &str,
    timestamp: &str,
    model: Model,
    role: &str,
    phase: &str,
    output_bytes: u32,
    parse_success: bool,
) -> PhaseMetrics {
    PhaseMetrics {
        run_id: run_id.into(),
        timestamp: timestamp.into(),
        model: model.name().into(),
        role: role.into(),
        phase: phase.into(),
        critiques_submitted: None,
        critiques_accepted: None,
        critiques_partial: None,
        critiques_rejected: None,
        high_severity_submitted: None,
        high_severity_accepted: None,
        output_bytes,
        parse_success,
    }
}

// ── Persistence ──────────────────────────────────────────────────────

/// Load all historical PhaseMetrics from a JSONL file.
pub fn load_metrics(path: &Path) -> Vec<PhaseMetrics> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    content
        .lines()
        .filter_map(|line| serde_json::from_str::<PhaseMetrics>(line).ok())
        .collect()
}

/// Format a human-readable scorecard from model profiles.
pub fn format_scorecard(profiles: &[ModelProfile]) -> String {
    let mut out = String::new();
    out.push_str("╔══════════════════════════════════════════════════════╗\n");
    out.push_str("║           Model Capability Scorecard                ║\n");
    out.push_str("╚══════════════════════════════════════════════════════╝\n\n");

    for profile in profiles {
        out.push_str(&format!(
            "  {} ({} observations)\n",
            profile.model, profile.runs_analyzed
        ));
        out.push_str("  ┌─────────────────────────┬────────┬──────────┬───────┬───────┐\n");
        out.push_str("  │ Role                    │ Accept │ Sev.Acc. │ Parse │ Score │\n");
        out.push_str("  ├─────────────────────────┼────────┼──────────┼───────┼───────┤\n");

        for rs in &profile.role_scores {
            out.push_str(&format!(
                "  │ {:23} │ {:5.0}% │ {:7.0}% │ {:4.0}% │ {:4.0}% │\n",
                truncate(&rs.role, 23),
                rs.acceptance_rate * 100.0,
                rs.severity_accuracy * 100.0,
                rs.parse_reliability * 100.0,
                rs.composite * 100.0,
            ));
        }
        out.push_str("  └─────────────────────────┴────────┴──────────┴───────┴───────┘\n\n");
    }

    out
}

/// Format role recommendations as human-readable text.
pub fn format_recommendations(recs: &[RoleRecommendation]) -> String {
    if recs.is_empty() {
        return "  No role swaps recommended (current assignments are optimal or insufficient data).\n"
            .into();
    }

    let mut out = String::new();
    out.push_str("  Recommended role swaps:\n");
    for rec in recs {
        out.push_str(&format!(
            "  * [{:.0}% confidence] {} {}: {} -> {}\n    {}\n",
            rec.confidence * 100.0,
            rec.phase,
            rec.current_role,
            rec.current_model,
            rec.recommended_model,
            rec.reason,
        ));
    }
    out
}

// ── Model-role fit scoring ───────────────────────────────────────────

/// A single model-role fit score entry, used for telemetry-driven analysis.
#[derive(Debug, Clone, Serialize)]
pub struct ModelRoleFit {
    pub model: String,
    pub role: String,
    pub phase: String,
    pub observations: u32,
    pub composite: f64,
    /// How many runs had 0% acceptance rate (all critiques rejected).
    pub zero_acceptance_runs: u32,
    /// Fraction of runs with zero acceptance.
    pub zero_acceptance_rate: f64,
    pub avg_acceptance: f64,
    pub avg_severity_accuracy: f64,
    pub parse_reliability: f64,
    /// Qualitative grade: "excellent", "good", "fair", "poor".
    pub grade: String,
}

/// Diagnose model-role fit from historical metrics, including zero-acceptance detection.
/// Returns fit scores sorted by composite (best first) for each (model, role) pair.
pub fn score_model_role_fit(metrics: &[PhaseMetrics]) -> Vec<ModelRoleFit> {
    let mut by_key: HashMap<(String, String, String), Vec<&PhaseMetrics>> = HashMap::new();
    for m in metrics {
        by_key
            .entry((m.model.clone(), m.role.clone(), m.phase.clone()))
            .or_default()
            .push(m);
    }

    let mut fits = Vec::new();
    for ((model, role, phase), entries) in &by_key {
        let n = entries.len() as u32;

        let acceptance_rates: Vec<f64> = entries
            .iter()
            .filter_map(|e| e.acceptance_rate())
            .collect();
        let avg_acceptance = if acceptance_rates.is_empty() {
            0.5
        } else {
            acceptance_rates.iter().sum::<f64>() / acceptance_rates.len() as f64
        };

        let zero_acceptance_runs = acceptance_rates
            .iter()
            .filter(|&&r| r == 0.0)
            .count() as u32;
        let zero_acceptance_rate = if acceptance_rates.is_empty() {
            0.0
        } else {
            zero_acceptance_runs as f64 / acceptance_rates.len() as f64
        };

        let severity_accs: Vec<f64> = entries
            .iter()
            .filter_map(|e| e.severity_accuracy())
            .collect();
        let avg_severity_acc = if severity_accs.is_empty() {
            0.5
        } else {
            severity_accs.iter().sum::<f64>() / severity_accs.len() as f64
        };

        let parse_success_count = entries.iter().filter(|e| e.parse_success).count();
        let parse_reliability = parse_success_count as f64 / n as f64;

        let is_critic = role.contains("adversar") || role.contains("critic");
        let composite = if is_critic {
            avg_acceptance * 0.4 + avg_severity_acc * 0.4 + parse_reliability * 0.2
        } else {
            avg_acceptance * 0.2 + parse_reliability * 0.8
        };

        let grade = if composite >= 0.85 {
            "excellent"
        } else if composite >= 0.65 {
            "good"
        } else if composite >= 0.45 {
            "fair"
        } else {
            "poor"
        };

        fits.push(ModelRoleFit {
            model: model.clone(),
            role: role.clone(),
            phase: phase.clone(),
            observations: n,
            composite: round2(composite),
            zero_acceptance_runs,
            zero_acceptance_rate: round2(zero_acceptance_rate),
            avg_acceptance: round2(avg_acceptance),
            avg_severity_accuracy: round2(avg_severity_acc),
            parse_reliability: round2(parse_reliability),
            grade: grade.into(),
        });
    }

    fits.sort_by(|a, b| {
        b.composite
            .partial_cmp(&a.composite)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fits
}

/// Format a human-readable model-role fit report with zero-acceptance diagnostics.
pub fn format_fit_report(fits: &[ModelRoleFit]) -> String {
    let mut out = String::new();
    out.push_str("Model-Role Fit Analysis\n");
    out.push_str("══════════════════════════════════════════════════════════════\n\n");

    for fit in fits {
        out.push_str(&format!(
            "  {} / {} ({})\n",
            fit.model, fit.role, fit.phase
        ));
        out.push_str(&format!(
            "    Observations: {}  Grade: {}  Composite: {:.0}%\n",
            fit.observations,
            fit.grade,
            fit.composite * 100.0
        ));
        out.push_str(&format!(
            "    Acceptance: {:.0}%  Severity Acc: {:.0}%  Parse: {:.0}%\n",
            fit.avg_acceptance * 100.0,
            fit.avg_severity_accuracy * 100.0,
            fit.parse_reliability * 100.0,
        ));
        if fit.zero_acceptance_runs > 0 {
            out.push_str(&format!(
                "    ⚠ Zero-acceptance runs: {} ({:.0}% of observations)\n",
                fit.zero_acceptance_runs,
                fit.zero_acceptance_rate * 100.0,
            ));
        }
        out.push('\n');
    }

    // Summary: flag any models with high zero-acceptance rates
    let problem_fits: Vec<&ModelRoleFit> = fits
        .iter()
        .filter(|f| f.zero_acceptance_rate > 0.2 && f.observations >= 2)
        .collect();
    if !problem_fits.is_empty() {
        out.push_str("Flagged: high zero-acceptance rate (>20% of runs)\n");
        for f in &problem_fits {
            out.push_str(&format!(
                "  → {} as {} in {}: {:.0}% zero-acceptance\n",
                f.model,
                f.role,
                f.phase,
                f.zero_acceptance_rate * 100.0,
            ));
        }
    }

    out
}

// ── Utilities ────────────────────────────────────────────────────────

/// Normalize a string for fuzzy matching: lowercase, strip punctuation.
fn normalize_for_match(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Fuzzy match: true if significant words from `needle` appear in `haystack`.
fn fuzzy_match(needle: &str, haystack: &str) -> bool {
    let needle_norm = normalize_for_match(needle);
    let haystack_norm = normalize_for_match(haystack);

    // Direct substring
    if haystack_norm.contains(&needle_norm) || needle_norm.contains(&haystack_norm) {
        return true;
    }

    // Word overlap: if >= 35% of significant words match
    let needle_words: Vec<&str> = needle_norm
        .split_whitespace()
        .filter(|w| w.len() > 3) // skip short words
        .collect();

    if needle_words.is_empty() {
        return false;
    }

    let word_matches = needle_words
        .iter()
        .filter(|w| haystack_norm.contains(**w))
        .count();

    if word_matches as f64 / needle_words.len() as f64 >= 0.35 {
        return true;
    }

    // Bigram matching: if any 2-word sequence from needle appears in haystack
    if needle_words.len() >= 2 {
        let haystack_words: Vec<&str> = haystack_norm.split_whitespace().collect();
        let haystack_bigrams: Vec<String> = haystack_words
            .windows(2)
            .map(|w| format!("{} {}", w[0], w[1]))
            .collect();

        for pair in needle_words.windows(2) {
            let bigram = format!("{} {}", pair[0], pair[1]);
            if haystack_bigrams.contains(&bigram) {
                return true;
            }
        }
    }

    false
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        format!("{:<width$}", s, width = max)
    } else {
        format!("{}...", &s[..max - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_phase_metrics(
        model: &str,
        role: &str,
        phase: &str,
        submitted: u32,
        accepted: u32,
        rejected: u32,
        high_sub: u32,
        high_acc: u32,
        parse_ok: bool,
    ) -> PhaseMetrics {
        PhaseMetrics {
            run_id: "test-run".into(),
            timestamp: "2026-04-03T00:00:00Z".into(),
            model: model.into(),
            role: role.into(),
            phase: phase.into(),
            critiques_submitted: Some(submitted),
            critiques_accepted: Some(accepted),
            critiques_partial: Some(0),
            critiques_rejected: Some(rejected),
            high_severity_submitted: Some(high_sub),
            high_severity_accepted: Some(high_acc),
            output_bytes: 1000,
            parse_success: parse_ok,
        }
    }

    #[test]
    fn acceptance_rate_basic() {
        let m = make_phase_metrics("claude", "adversary", "deliberation", 10, 7, 3, 2, 1, true);
        assert!((m.acceptance_rate().unwrap() - 0.7).abs() < 0.01);
    }

    #[test]
    fn acceptance_rate_zero_submitted() {
        let m = make_phase_metrics("claude", "adversary", "deliberation", 0, 0, 0, 0, 0, true);
        assert!((m.acceptance_rate().unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn severity_accuracy_basic() {
        let m = make_phase_metrics("claude", "adversary", "deliberation", 10, 7, 3, 4, 3, true);
        assert!((m.severity_accuracy().unwrap() - 0.75).abs() < 0.01);
    }

    #[test]
    fn severity_accuracy_no_highs() {
        let m = make_phase_metrics("claude", "adversary", "deliberation", 10, 7, 3, 0, 0, true);
        assert!((m.severity_accuracy().unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn aggregate_profiles_groups_by_model_and_role() {
        let metrics = vec![
            make_phase_metrics("claude", "adversary", "deliberation", 5, 4, 1, 1, 1, true),
            make_phase_metrics("claude", "adversary", "deliberation", 6, 5, 1, 2, 2, true),
            make_phase_metrics("gemini", "adversary", "deliberation", 4, 2, 2, 2, 0, true),
        ];
        let profiles = aggregate_profiles(&metrics);
        assert_eq!(profiles.len(), 2);

        let claude = profiles.iter().find(|p| p.model == "claude").unwrap();
        assert_eq!(claude.runs_analyzed, 2);
        assert_eq!(claude.role_scores.len(), 1);
        assert_eq!(claude.role_scores[0].observations, 2);
        // Claude: avg acceptance = (4/5 + 5/6) / 2 = (0.8 + 0.833) / 2 = 0.817
        assert!(claude.role_scores[0].acceptance_rate > 0.80);

        let gemini = profiles.iter().find(|p| p.model == "gemini").unwrap();
        // Gemini: acceptance = 2/4 = 0.5
        assert!((gemini.role_scores[0].acceptance_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn recommend_roles_suggests_swap_on_big_delta() {
        let metrics = vec![
            // Claude is great at adversary
            make_phase_metrics("claude", "adversary", "deliberation", 10, 9, 1, 2, 2, true),
            make_phase_metrics("claude", "adversary", "deliberation", 10, 8, 2, 3, 3, true),
            // Gemini is weak at adversary
            make_phase_metrics("gemini", "adversary", "deliberation", 10, 4, 6, 4, 1, true),
            make_phase_metrics("gemini", "adversary", "deliberation", 10, 3, 7, 3, 0, true),
        ];
        let profiles = aggregate_profiles(&metrics);

        // Currently gemini is adversary
        let assignments = vec![(Model::Gemini, "adversary", "deliberation")];
        let recs = recommend_roles(&profiles, &assignments, 2, 0.2);

        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].recommended_model, "claude");
        assert_eq!(recs[0].current_model, "gemini");
    }

    #[test]
    fn recommend_roles_no_swap_when_close() {
        let metrics = vec![
            make_phase_metrics("claude", "adversary", "deliberation", 10, 8, 2, 2, 2, true),
            make_phase_metrics("gemini", "adversary", "deliberation", 10, 7, 3, 2, 2, true),
        ];
        let profiles = aggregate_profiles(&metrics);
        let assignments = vec![(Model::Gemini, "adversary", "deliberation")];
        let recs = recommend_roles(&profiles, &assignments, 1, 0.2);

        // Delta is small, no recommendation
        assert!(recs.is_empty());
    }

    #[test]
    fn fuzzy_match_substring() {
        assert!(fuzzy_match("timeout risk", "Synchronous Tool Timeout Risk"));
        assert!(fuzzy_match(
            "Synchronous Tool Timeout Risk",
            "timeout risk in sync tools"
        ));
    }

    #[test]
    fn fuzzy_match_word_overlap() {
        assert!(fuzzy_match(
            "Missing arXiv category filtering",
            "No arXiv category pre-filtering at the API level"
        ));
    }

    #[test]
    fn fuzzy_match_no_match() {
        assert!(!fuzzy_match("completely different", "nothing in common here"));
    }

    #[test]
    fn format_scorecard_produces_output() {
        let profiles = vec![ModelProfile {
            model: "claude".into(),
            runs_analyzed: 5,
            role_scores: vec![RoleScore {
                role: "adversary".into(),
                phase: "deliberation".into(),
                observations: 5,
                acceptance_rate: 0.85,
                severity_accuracy: 0.90,
                parse_reliability: 1.0,
                composite: 0.90,
            }],
        }];
        let card = format_scorecard(&profiles);
        assert!(card.contains("claude"));
        assert!(card.contains("adversary"));
        assert!(card.contains("85%"));
    }

    // -- Model-role fit scoring tests --

    #[test]
    fn score_model_role_fit_detects_zero_acceptance() {
        let metrics = vec![
            // Run 1: all rejected (0 accepted, 5 rejected)
            make_phase_metrics("codex", "synthesis-owner", "deliberation", 5, 0, 5, 0, 0, true),
            // Run 2: all rejected again
            make_phase_metrics("codex", "synthesis-owner", "deliberation", 4, 0, 4, 0, 0, true),
            // Run 3: some accepted
            make_phase_metrics("codex", "synthesis-owner", "deliberation", 6, 4, 2, 0, 0, true),
        ];
        let fits = score_model_role_fit(&metrics);
        assert_eq!(fits.len(), 1);
        let fit = &fits[0];
        assert_eq!(fit.model, "codex");
        assert_eq!(fit.zero_acceptance_runs, 2);
        assert!((fit.zero_acceptance_rate - 0.67).abs() < 0.01);
    }

    #[test]
    fn score_model_role_fit_grades_correctly() {
        let metrics = vec![
            // Excellent: high acceptance, good severity accuracy, good parse
            make_phase_metrics("claude", "adversarial-reviewer", "deliberation", 10, 9, 1, 2, 2, true),
            // Poor: low acceptance, bad severity accuracy, bad parse
            make_phase_metrics("gemini", "adversarial-reviewer", "deliberation", 10, 2, 8, 4, 0, false),
        ];
        let fits = score_model_role_fit(&metrics);
        let claude_fit = fits.iter().find(|f| f.model == "claude").unwrap();
        let gemini_fit = fits.iter().find(|f| f.model == "gemini").unwrap();
        assert_eq!(claude_fit.grade, "excellent");
        assert!(gemini_fit.grade == "poor" || gemini_fit.grade == "fair");
    }

    #[test]
    fn score_model_role_fit_no_zero_acceptance_when_all_accepted() {
        let metrics = vec![
            make_phase_metrics("claude", "adversarial-reviewer", "deliberation", 5, 5, 0, 1, 1, true),
            make_phase_metrics("claude", "adversarial-reviewer", "deliberation", 3, 3, 0, 0, 0, true),
        ];
        let fits = score_model_role_fit(&metrics);
        assert_eq!(fits[0].zero_acceptance_runs, 0);
        assert!((fits[0].zero_acceptance_rate - 0.0).abs() < 0.01);
    }

    #[test]
    fn format_fit_report_includes_warning_for_zero_acceptance() {
        let fits = vec![ModelRoleFit {
            model: "codex".into(),
            role: "synthesis-owner".into(),
            phase: "deliberation".into(),
            observations: 4,
            composite: 0.30,
            zero_acceptance_runs: 3,
            zero_acceptance_rate: 0.75,
            avg_acceptance: 0.10,
            avg_severity_accuracy: 0.50,
            parse_reliability: 1.0,
            grade: "poor".into(),
        }];
        let report = format_fit_report(&fits);
        assert!(report.contains("Zero-acceptance runs: 3"));
        assert!(report.contains("zero-acceptance"));
        assert!(report.contains("codex"));
    }
}
