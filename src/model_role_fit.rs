use crate::metrics::{PhaseMetrics, load_metrics};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelRoleFitScore {
    pub model: String,
    pub role: String,
    pub score: f64,
    pub confidence: f64,
    pub sample_count: usize,
}

pub struct RoleFitness;

impl RoleFitness {
    /// Load historical telemetry from the default council metrics path and score
    /// model-role fitness for the given task type.
    #[allow(dead_code)]
    pub fn new(task_type: &str) -> Vec<ModelRoleFitScore> {
        let metrics = load_metrics(&default_metrics_path());
        Self::from_metrics(task_type, &metrics)
    }

    /// Score model-role fitness from the provided telemetry, falling back to
    /// capability heuristics when telemetry is sparse.
    pub fn from_metrics(task_type: &str, metrics: &[PhaseMetrics]) -> Vec<ModelRoleFitScore> {
        let mut by_pair: HashMap<(String, String), Vec<&PhaseMetrics>> = HashMap::new();
        for metric in metrics {
            let canonical_role = canonical_role(&metric.role, &metric.phase);
            by_pair
                .entry((metric.model.clone(), canonical_role.to_string()))
                .or_default()
                .push(metric);
        }

        let mut scores = Vec::new();
        for model in ["claude", "gemini", "codex", "qwen-36-plus"] {
            for role in ["orchestrator", "critic", "expert", "synthesizer"] {
                let telemetry = by_pair
                    .get(&(model.to_string(), role.to_string()))
                    .map(|entries| summarize_telemetry(entries.as_slice()))
                    .unwrap_or_default();

                let capability = capability_fit(model, role, task_type);
                let telemetry_weight = telemetry_weight(telemetry.sample_count);
                let blended =
                    capability * (1.0 - telemetry_weight) + telemetry.score * telemetry_weight;
                let confidence =
                    ((telemetry.confidence * 0.7) + (telemetry_weight * 0.3)).clamp(0.2, 0.99);

                scores.push(ModelRoleFitScore {
                    model: model.to_string(),
                    role: role.to_string(),
                    score: round3(blended),
                    confidence: round3(confidence),
                    sample_count: telemetry.sample_count,
                });
            }
        }

        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| b.sample_count.cmp(&a.sample_count))
                .then_with(|| a.model.cmp(&b.model))
        });
        scores
    }
}

#[derive(Debug, Default)]
struct TelemetrySummary {
    score: f64,
    confidence: f64,
    sample_count: usize,
}

fn summarize_telemetry(entries: &[&PhaseMetrics]) -> TelemetrySummary {
    if entries.is_empty() {
        return TelemetrySummary {
            score: 0.5,
            confidence: 0.2,
            sample_count: 0,
        };
    }

    let sample_count = entries.len();
    let parse_reliability =
        entries.iter().filter(|m| m.parse_success).count() as f64 / sample_count as f64;
    let avg_acceptance = avg(entries.iter().filter_map(|m| m.acceptance_rate())).unwrap_or(0.55);
    let avg_severity_accuracy =
        avg(entries.iter().filter_map(|m| m.severity_accuracy())).unwrap_or(0.65);
    let avg_output_size = avg(entries.iter().map(|m| m.output_bytes as f64)).unwrap_or(2_000.0);
    let output_size_score = output_size_fitness(avg_output_size);
    let avg_unmatched = avg(entries.iter().filter_map(|m| m.unmatched_rate())).unwrap_or(0.0);
    let match_quality = 1.0 - avg_unmatched;

    let score = round3(
        avg_acceptance * 0.25
            + avg_severity_accuracy * 0.20
            + parse_reliability * 0.20
            + output_size_score * 0.15
            + match_quality * 0.20,
    );

    // Confidence drops when unmatched rate is high (data is unreliable)
    let confidence = round3(
        ((sample_count as f64 / 6.0).min(1.0) * 0.6 + parse_reliability * 0.2 + match_quality * 0.2).clamp(0.2, 0.99),
    );

    TelemetrySummary {
        score,
        confidence,
        sample_count,
    }
}

fn canonical_role(role: &str, phase: &str) -> &'static str {
    let role = role.to_ascii_lowercase();
    let phase = phase.to_ascii_lowercase();

    if role.contains("framing") || role.contains("orchestr") {
        "orchestrator"
    } else if role.contains("adversarial") || role.contains("critic") {
        "critic"
    } else if role.contains("synthesis") || role.contains("build-lead") || role.contains("handoff")
    {
        "synthesizer"
    } else if role.contains("scout") || phase.contains("brainstorm") || role.contains("expert") {
        "expert"
    } else {
        "expert"
    }
}

fn capability_fit(model: &str, role: &str, task_type: &str) -> f64 {
    let caps = capability_profile(model);
    let task = task_type.to_ascii_lowercase();

    let base = match role {
        "orchestrator" => {
            caps.intelligence * 0.40
                + caps.planning * 0.35
                + caps.context * 0.15
                + caps.speed * 0.10
        }
        "critic" => {
            caps.intelligence * 0.35
                + caps.critique * 0.35
                + caps.context * 0.15
                + caps.cost_efficiency * 0.15
        }
        "expert" => {
            caps.domain * 0.30
                + caps.implementation * 0.25
                + caps.context * 0.20
                + caps.intelligence * 0.15
                + caps.speed * 0.10
        }
        "synthesizer" => {
            caps.synthesis * 0.35
                + caps.intelligence * 0.25
                + caps.context * 0.20
                + caps.implementation * 0.10
                + caps.cost_efficiency * 0.10
        }
        _ => 0.5,
    };

    let task_bonus = match role {
        "expert"
            if task.contains("rust")
                || task.contains("code")
                || task.contains("implement")
                || task.contains("engineering") =>
        {
            if model == "codex" {
                0.08
            } else if model == "claude" {
                0.04
            } else {
                0.02
            }
        }
        "critic" if task.contains("safety") || task.contains("review") || task.contains("risk") => {
            if model == "claude" {
                0.07
            } else if model == "qwen-36-plus" {
                0.06 // reasoning model, strong at risk analysis
            } else if model == "gemini" {
                0.04
            } else {
                0.02
            }
        }
        "orchestrator"
            if task.contains("plan")
                || task.contains("architecture")
                || task.contains("design") =>
        {
            if model == "claude" {
                0.08
            } else if model == "gemini" {
                0.04
            } else {
                0.03
            }
        }
        "synthesizer"
            if task.contains("plan") || task.contains("handoff") || task.contains("synth") =>
        {
            if model == "codex" {
                0.07
            } else if model == "claude" {
                0.05
            } else {
                0.03
            }
        }
        _ => 0.0,
    };

    (base + task_bonus).clamp(0.0, 1.0)
}

struct CapabilityProfile {
    intelligence: f64,
    context: f64,
    speed: f64,
    cost_efficiency: f64,
    planning: f64,
    critique: f64,
    synthesis: f64,
    domain: f64,
    implementation: f64,
}

fn capability_profile(model: &str) -> CapabilityProfile {
    match model {
        // Heuristic priors from current council operating assumptions.
        "claude" => CapabilityProfile {
            intelligence: 0.96,
            context: 0.92,
            speed: 0.72,
            cost_efficiency: 0.58,
            planning: 0.97,
            critique: 0.95,
            synthesis: 0.88,
            domain: 0.86,
            implementation: 0.82,
        },
        "gemini" => CapabilityProfile {
            intelligence: 0.82,
            context: 0.90,
            speed: 0.88,
            cost_efficiency: 0.82,
            planning: 0.80,
            critique: 0.78,
            synthesis: 0.74,
            domain: 0.83,
            implementation: 0.75,
        },
        "codex" => CapabilityProfile {
            intelligence: 0.86,
            context: 0.78,
            speed: 0.84,
            cost_efficiency: 0.76,
            planning: 0.83,
            critique: 0.70,
            synthesis: 0.91,
            domain: 0.88,
            implementation: 0.94,
        },
        "qwen-36-plus" => CapabilityProfile {
            intelligence: 0.88,
            context: 0.95,  // 1M token context window
            speed: 0.65,    // remote API, free tier
            cost_efficiency: 1.0, // free tier
            planning: 0.85,
            critique: 0.88,  // reasoning model
            synthesis: 0.80,
            domain: 0.84,
            implementation: 0.78,
        },
        _ => openrouter_free_default_profile(model),
    }
}

/// Derive a default capability profile for unknown models.
///
/// If the model name maps to a known OpenRouter free-tier model (by checking
/// the provider registry), we generate a reasonable heuristic profile based
/// on whether the model supports reasoning.  This lets new free models be
/// added to the registry without requiring a handcrafted profile.
fn openrouter_free_default_profile(model_name: &str) -> CapabilityProfile {
    // Try to match council model name back to a registry entry.
    // Convention: council names like "qwen-36-plus" map to IDs like "qwen/qwen3.6-plus".
    // For models we don't have an explicit profile for, check if their OpenRouter ID
    // is in the free registry.
    if let Some(entry) = crate::provider::lookup_free_model(model_name) {
        return free_tier_profile(entry.reasoning);
    }

    // Also check by iterating the registry for partial name matches.
    // This handles the case where the council model name is derived from the
    // OpenRouter ID (e.g. "deepseek-r1" matching "deepseek/deepseek-r1-0528:free").
    for entry in crate::provider::free_model_registry() {
        let id_lower = entry.id.to_ascii_lowercase();
        let name_lower = model_name.to_ascii_lowercase();
        // Match if the model name appears as a substring of the provider ID
        // (minus the org prefix and :free suffix).
        let slug = id_lower
            .split('/')
            .last()
            .unwrap_or(&id_lower)
            .trim_end_matches(":free");
        if slug.contains(&name_lower) || name_lower.contains(slug) {
            return free_tier_profile(entry.reasoning);
        }
    }

    // Completely unknown model — conservative defaults.
    CapabilityProfile {
        intelligence: 0.5,
        context: 0.5,
        speed: 0.5,
        cost_efficiency: 0.5,
        planning: 0.5,
        critique: 0.5,
        synthesis: 0.5,
        domain: 0.5,
        implementation: 0.5,
    }
}

/// Heuristic capability profile for a free-tier OpenRouter model.
///
/// Reasoning models (chain-of-thought) get higher critique/planning scores.
/// All free-tier models get cost_efficiency = 1.0 and lower speed (remote API).
fn free_tier_profile(reasoning: bool) -> CapabilityProfile {
    if reasoning {
        CapabilityProfile {
            intelligence: 0.82,
            context: 0.85,
            speed: 0.60,      // remote API, free tier
            cost_efficiency: 1.0, // free tier
            planning: 0.80,
            critique: 0.84,   // reasoning models are good critics
            synthesis: 0.75,
            domain: 0.78,
            implementation: 0.72,
        }
    } else {
        CapabilityProfile {
            intelligence: 0.72,
            context: 0.80,
            speed: 0.60,
            cost_efficiency: 1.0,
            planning: 0.68,
            critique: 0.70,
            synthesis: 0.68,
            domain: 0.72,
            implementation: 0.68,
        }
    }
}

fn telemetry_weight(sample_count: usize) -> f64 {
    match sample_count {
        0 => 0.0,
        1 => 0.25,
        2..=3 => 0.45,
        4..=6 => 0.65,
        _ => 0.8,
    }
}

fn output_size_fitness(avg_output_size: f64) -> f64 {
    match avg_output_size as u32 {
        0..=149 => 0.30,
        150..=799 => 0.70,
        800..=12_000 => 1.0,
        12_001..=18_000 => 0.75,
        _ => 0.55,
    }
}

#[allow(dead_code)]
fn default_metrics_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".council")
        .join("council-metrics.jsonl")
}

fn avg(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut count = 0usize;
    let mut total = 0.0;
    for value in values {
        count += 1;
        total += value;
    }
    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::PhaseMetrics;

    fn metric(
        model: &str,
        role: &str,
        phase: &str,
        accepted: u32,
        rejected: u32,
        parse_success: bool,
    ) -> PhaseMetrics {
        PhaseMetrics {
            run_id: "run".into(),
            timestamp: "2026-04-04T00:00:00Z".into(),
            model: model.into(),
            role: role.into(),
            phase: phase.into(),
            critiques_submitted: Some(accepted + rejected),
            critiques_accepted: Some(accepted),
            critiques_partial: Some(0),
            critiques_rejected: Some(rejected),
            high_severity_submitted: Some(1),
            high_severity_accepted: Some(1.min(accepted)),
            critiques_unmatched: Some(0),
            output_bytes: 1_200,
            parse_success,
        }
    }

    #[test]
    fn canonical_role_maps_current_roles() {
        assert_eq!(
            canonical_role("framing-controller", "framing"),
            "orchestrator"
        );
        assert_eq!(
            canonical_role("adversarial-reviewer/conceptual-risk", "deliberation"),
            "critic"
        );
        assert_eq!(canonical_role("solution-scout", "brainstorming"), "expert");
        assert_eq!(
            canonical_role("synthesis-owner", "deliberation"),
            "synthesizer"
        );
    }

    #[test]
    fn role_fitness_prefers_claude_for_critic_with_good_telemetry() {
        let metrics = vec![
            metric(
                "claude",
                "adversarial-reviewer/conceptual-risk",
                "deliberation",
                5,
                0,
                true,
            ),
            metric(
                "claude",
                "adversarial-reviewer/conceptual-risk",
                "deliberation",
                4,
                1,
                true,
            ),
            metric(
                "gemini",
                "adversarial-reviewer/implementation-risk",
                "deliberation",
                2,
                3,
                true,
            ),
        ];

        let scores = RoleFitness::from_metrics("planning", &metrics);
        let claude_critic = scores
            .iter()
            .find(|s| s.model == "claude" && s.role == "critic")
            .unwrap();
        let gemini_critic = scores
            .iter()
            .find(|s| s.model == "gemini" && s.role == "critic")
            .unwrap();
        assert!(claude_critic.score > gemini_critic.score);
        assert_eq!(claude_critic.sample_count, 2);
    }

    #[test]
    fn heuristics_fill_in_when_no_telemetry() {
        let scores = RoleFitness::from_metrics("architecture planning", &[]);
        let orchestrator = scores
            .iter()
            .find(|s| s.model == "claude" && s.role == "orchestrator")
            .unwrap();
        let gemini_orch = scores
            .iter()
            .find(|s| s.model == "gemini" && s.role == "orchestrator")
            .unwrap();
        assert!(orchestrator.score > gemini_orch.score);
        assert_eq!(orchestrator.sample_count, 0);
    }

    #[test]
    fn free_tier_profile_reasoning_model_scores_higher_critique() {
        let reasoning = super::free_tier_profile(true);
        let standard = super::free_tier_profile(false);
        assert!(reasoning.critique > standard.critique);
        assert!(reasoning.planning > standard.planning);
        assert_eq!(reasoning.cost_efficiency, 1.0);
        assert_eq!(standard.cost_efficiency, 1.0);
    }

    #[test]
    fn openrouter_free_default_matches_registry_entry() {
        // A model ID directly in the registry should get a free-tier profile
        let profile = super::openrouter_free_default_profile("deepseek/deepseek-r1-0528:free");
        assert_eq!(profile.cost_efficiency, 1.0);
        assert!(profile.critique > 0.8); // reasoning model
    }

    #[test]
    fn openrouter_free_default_unknown_gets_conservative() {
        let profile = super::openrouter_free_default_profile("totally-unknown-model");
        assert_eq!(profile.cost_efficiency, 0.5);
        assert_eq!(profile.intelligence, 0.5);
    }

    #[test]
    fn qwen_included_in_role_fitness_scoring() {
        let scores = RoleFitness::from_metrics("planning", &[]);
        let qwen_scores: Vec<_> = scores.iter().filter(|s| s.model == "qwen-36-plus").collect();
        // Qwen should have entries for all 4 canonical roles
        assert_eq!(qwen_scores.len(), 4, "Qwen should have 4 role scores");
        for s in &qwen_scores {
            assert!(s.score > 0.0 && s.score <= 1.0, "score out of range: {}", s.score);
        }
        // Qwen should score well as critic (reasoning model)
        let qwen_critic = qwen_scores.iter().find(|s| s.role == "critic").unwrap();
        assert!(qwen_critic.score > 0.7, "Qwen critic score too low: {}", qwen_critic.score);
    }

    // ── Proof D: role-fit scoring effect-on vs effect-off ────────────

    #[test]
    fn proof_d_telemetry_shifts_scores_vs_heuristic_only() {
        // Effect-OFF: no telemetry → pure heuristic scoring.
        let heuristic_only = RoleFitness::from_metrics("planning", &[]);

        // Effect-ON: inject telemetry showing Gemini is an excellent critic
        // and Claude is a poor critic (opposite of heuristics).
        let telemetry = vec![
            metric("gemini", "adversarial-reviewer/conceptual-risk", "deliberation", 8, 0, true),
            metric("gemini", "adversarial-reviewer/conceptual-risk", "deliberation", 7, 1, true),
            metric("gemini", "adversarial-reviewer/conceptual-risk", "deliberation", 6, 1, true),
            metric("gemini", "adversarial-reviewer/conceptual-risk", "deliberation", 9, 0, true),
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 1, 7, true),
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 0, 6, true),
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 1, 5, true),
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 0, 8, true),
        ];
        let with_telemetry = RoleFitness::from_metrics("planning", &telemetry);

        let find = |scores: &[ModelRoleFitScore], model: &str, role: &str| -> f64 {
            scores.iter().find(|s| s.model == model && s.role == role).unwrap().score
        };

        // Heuristic-only: Claude should score higher than Gemini as critic.
        let h_claude_critic = find(&heuristic_only, "claude", "critic");
        let h_gemini_critic = find(&heuristic_only, "gemini", "critic");
        assert!(
            h_claude_critic > h_gemini_critic,
            "heuristic: Claude ({h_claude_critic}) should beat Gemini ({h_gemini_critic}) as critic"
        );

        // With telemetry: Gemini's excellent critique data should boost its score,
        // and Claude's poor critique data should drop its score.
        let t_claude_critic = find(&with_telemetry, "claude", "critic");
        let t_gemini_critic = find(&with_telemetry, "gemini", "critic");

        // The telemetry should measurably shift scores.
        assert!(
            t_gemini_critic > h_gemini_critic,
            "telemetry should boost Gemini critic: {t_gemini_critic} > {h_gemini_critic}"
        );
        assert!(
            t_claude_critic < h_claude_critic,
            "telemetry should reduce Claude critic: {t_claude_critic} < {h_claude_critic}"
        );
    }

    #[test]
    fn proof_d_more_telemetry_increases_confidence() {
        let one_sample = vec![
            metric("codex", "synthesis-owner", "deliberation", 5, 1, true),
        ];
        let many_samples = vec![
            metric("codex", "synthesis-owner", "deliberation", 5, 1, true),
            metric("codex", "synthesis-owner", "deliberation", 6, 0, true),
            metric("codex", "synthesis-owner", "deliberation", 4, 2, true),
            metric("codex", "synthesis-owner", "deliberation", 5, 1, true),
            metric("codex", "synthesis-owner", "deliberation", 7, 0, true),
            metric("codex", "synthesis-owner", "deliberation", 6, 1, true),
            metric("codex", "synthesis-owner", "deliberation", 5, 0, true),
        ];

        let scores_1 = RoleFitness::from_metrics("planning", &one_sample);
        let scores_7 = RoleFitness::from_metrics("planning", &many_samples);

        let find_confidence = |scores: &[ModelRoleFitScore]| -> f64 {
            scores.iter()
                .find(|s| s.model == "codex" && s.role == "synthesizer")
                .unwrap()
                .confidence
        };

        let conf_1 = find_confidence(&scores_1);
        let conf_7 = find_confidence(&scores_7);
        assert!(
            conf_7 > conf_1,
            "more samples should increase confidence: {conf_7} > {conf_1}"
        );
    }

    #[test]
    fn proof_d_parse_failures_reduce_score() {
        let good_parse = vec![
            metric("gemini", "solution-scout", "brainstorming", 4, 1, true),
            metric("gemini", "solution-scout", "brainstorming", 5, 0, true),
        ];
        let bad_parse = vec![
            metric("gemini", "solution-scout", "brainstorming", 4, 1, false),
            metric("gemini", "solution-scout", "brainstorming", 5, 0, false),
        ];

        let scores_good = RoleFitness::from_metrics("planning", &good_parse);
        let scores_bad = RoleFitness::from_metrics("planning", &bad_parse);

        let find = |scores: &[ModelRoleFitScore]| -> f64 {
            scores.iter().find(|s| s.model == "gemini" && s.role == "expert").unwrap().score
        };

        assert!(
            find(&scores_good) > find(&scores_bad),
            "parse failures should reduce score"
        );
    }

    // ── Proof F: unmatched_rate degrades role-fit scoring ───────────────
    //
    // The summarize_telemetry function uses match_quality (1 - unmatched_rate)
    // as 20% of the telemetry score. This proof shows that high unmatched
    // rates in PhaseMetrics degrade the blended role-fit score.

    fn metric_with_unmatched(
        model: &str,
        role: &str,
        phase: &str,
        accepted: u32,
        rejected: u32,
        unmatched: u32,
        parse_success: bool,
    ) -> PhaseMetrics {
        let submitted = accepted + rejected + unmatched;
        PhaseMetrics {
            run_id: "run".into(),
            timestamp: "2026-04-08T00:00:00Z".into(),
            model: model.into(),
            role: role.into(),
            phase: phase.into(),
            critiques_submitted: Some(submitted),
            critiques_accepted: Some(accepted),
            critiques_partial: Some(0),
            critiques_rejected: Some(rejected),
            high_severity_submitted: Some(1),
            high_severity_accepted: Some(1.min(accepted)),
            critiques_unmatched: Some(unmatched),
            output_bytes: 1_200,
            parse_success,
        }
    }

    #[test]
    fn proof_f_high_unmatched_rate_degrades_role_fit_score() {
        // Effect-OFF: low unmatched rate (clean decision-log matching)
        let clean = vec![
            metric_with_unmatched("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 5, 1, 0, true),
            metric_with_unmatched("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 6, 1, 0, true),
            metric_with_unmatched("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 5, 2, 0, true),
        ];

        // Effect-ON: high unmatched rate (format mismatch in decision log)
        let messy = vec![
            metric_with_unmatched("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 2, 0, 5, true),
            metric_with_unmatched("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 1, 0, 6, true),
            metric_with_unmatched("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 2, 1, 4, true),
        ];

        let scores_clean = RoleFitness::from_metrics("planning", &clean);
        let scores_messy = RoleFitness::from_metrics("planning", &messy);

        let find = |scores: &[ModelRoleFitScore]| -> f64 {
            scores.iter().find(|s| s.model == "claude" && s.role == "critic").unwrap().score
        };

        let clean_score = find(&scores_clean);
        let messy_score = find(&scores_messy);

        assert!(
            clean_score > messy_score,
            "high unmatched rate should degrade role-fit score: clean={clean_score} > messy={messy_score}"
        );
    }

    #[test]
    fn proof_f_unmatched_rate_reduces_confidence() {
        let clean = vec![
            metric_with_unmatched("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 5, 1, 0, true),
            metric_with_unmatched("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 4, 2, 0, true),
        ];
        let messy = vec![
            metric_with_unmatched("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 1, 0, 5, true),
            metric_with_unmatched("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 0, 0, 6, true),
        ];

        let scores_clean = RoleFitness::from_metrics("planning", &clean);
        let scores_messy = RoleFitness::from_metrics("planning", &messy);

        let find_conf = |scores: &[ModelRoleFitScore]| -> f64 {
            scores.iter().find(|s| s.model == "gemini" && s.role == "critic").unwrap().confidence
        };

        assert!(
            find_conf(&scores_clean) > find_conf(&scores_messy),
            "high unmatched rate should reduce confidence"
        );
    }

    // ── Proof G: task-type bonuses shift ranking order ──────────────────
    //
    // Task-type bonuses are small (+0.02–0.08) but for closely-matched models
    // they can change who ranks first. This proves the effect is real.

    #[test]
    fn proof_g_rust_task_boosts_codex_expert_over_gemini() {
        // Heuristic-only (no telemetry) — compare expert scores.
        let generic = RoleFitness::from_metrics("planning", &[]);
        let rust_task = RoleFitness::from_metrics("rust code implementation", &[]);

        let find = |scores: &[ModelRoleFitScore], model: &str| -> f64 {
            scores.iter().find(|s| s.model == model && s.role == "expert").unwrap().score
        };

        let codex_generic = find(&generic, "codex");
        let codex_rust = find(&rust_task, "codex");

        // Codex should get a bonus on Rust tasks
        assert!(
            codex_rust > codex_generic,
            "Codex expert should score higher on Rust task: {codex_rust} > {codex_generic}"
        );

        // The Rust bonus for Codex (+0.08) should be larger than for Gemini (+0.02)
        let gemini_generic = find(&generic, "gemini");
        let gemini_rust = find(&rust_task, "gemini");
        let codex_boost = codex_rust - codex_generic;
        let gemini_boost = gemini_rust - gemini_generic;
        assert!(
            codex_boost > gemini_boost,
            "Codex Rust boost ({codex_boost:.3}) should exceed Gemini's ({gemini_boost:.3})"
        );
    }

    #[test]
    fn proof_g_safety_task_boosts_claude_critic() {
        let generic = RoleFitness::from_metrics("planning", &[]);
        let safety = RoleFitness::from_metrics("safety review", &[]);

        let find = |scores: &[ModelRoleFitScore], model: &str| -> f64 {
            scores.iter().find(|s| s.model == model && s.role == "critic").unwrap().score
        };

        let claude_generic = find(&generic, "claude");
        let claude_safety = find(&safety, "claude");

        assert!(
            claude_safety > claude_generic,
            "Claude critic should score higher on safety task: {claude_safety} > {claude_generic}"
        );

        // Claude's safety bonus (+0.07) should be the largest among all models
        let qwen_boost = find(&safety, "qwen-36-plus") - find(&generic, "qwen-36-plus");
        let claude_boost = claude_safety - claude_generic;
        assert!(
            claude_boost > qwen_boost,
            "Claude's safety critic boost ({claude_boost:.3}) should exceed Qwen's ({qwen_boost:.3})"
        );
    }

    #[test]
    fn proof_g_design_task_boosts_claude_orchestrator() {
        let generic = RoleFitness::from_metrics("generic task", &[]);
        let design = RoleFitness::from_metrics("architecture design planning", &[]);

        let find = |scores: &[ModelRoleFitScore], model: &str| -> f64 {
            scores.iter().find(|s| s.model == model && s.role == "orchestrator").unwrap().score
        };

        let claude_generic = find(&generic, "claude");
        let claude_design = find(&design, "claude");

        assert!(
            claude_design > claude_generic,
            "Claude orchestrator should score higher on design task"
        );
    }

    // ── Proof H: metrics → role-fit → recommendation coherence ─────────
    //
    // End-to-end: when telemetry shows model A outperforming model B for
    // a role, both the role-fit scoring AND the recommendation engine agree.

    #[test]
    fn proof_h_role_fit_and_recommendations_agree_on_best_critic() {
        // Telemetry: Claude is excellent, Gemini is poor — SAME role string
        // so aggregate_profiles can compare them head-to-head.
        let role = "adversarial-reviewer/conceptual-risk";
        let telemetry = vec![
            metric("claude", role, "deliberation", 8, 0, true),
            metric("claude", role, "deliberation", 7, 1, true),
            metric("claude", role, "deliberation", 9, 0, true),
            metric("gemini", role, "deliberation", 2, 6, true),
            metric("gemini", role, "deliberation", 1, 7, true),
            metric("gemini", role, "deliberation", 3, 5, true),
        ];

        // Role-fit scoring path: Claude should rank higher as critic.
        let fit_scores = RoleFitness::from_metrics("planning", &telemetry);
        let claude_critic = fit_scores.iter().find(|s| s.model == "claude" && s.role == "critic").unwrap();
        let gemini_critic = fit_scores.iter().find(|s| s.model == "gemini" && s.role == "critic").unwrap();
        assert!(
            claude_critic.score > gemini_critic.score,
            "role-fit: Claude critic ({}) should beat Gemini critic ({})",
            claude_critic.score, gemini_critic.score
        );

        // Recommendation path: should recommend swapping Gemini → Claude for adversary.
        let profiles = crate::metrics::aggregate_profiles(&telemetry);
        let assignments = vec![
            (crate::model::Model::Gemini, role, "deliberation"),
        ];
        let recs = crate::metrics::recommend_roles(&profiles, &assignments, 2, 0.1);

        // Both systems agree: Claude is the better critic.
        assert!(
            !recs.is_empty(),
            "recommendation engine should suggest swapping weak Gemini adversary"
        );
        assert_eq!(recs[0].recommended_model, "claude");
    }

    #[test]
    fn proof_h_role_fit_ranking_stable_across_task_types() {
        // When telemetry strongly favors one model, the ranking should hold
        // regardless of task type (telemetry dominates heuristic bonuses).
        let strong_telemetry = vec![
            metric("gemini", "synthesis-owner", "deliberation", 9, 0, true),
            metric("gemini", "synthesis-owner", "deliberation", 8, 1, true),
            metric("gemini", "synthesis-owner", "deliberation", 9, 0, true),
            metric("gemini", "synthesis-owner", "deliberation", 7, 1, true),
            metric("gemini", "synthesis-owner", "deliberation", 8, 0, true),
            metric("gemini", "synthesis-owner", "deliberation", 9, 1, true),
            metric("gemini", "synthesis-owner", "deliberation", 8, 0, true),
            metric("codex", "synthesis-owner", "deliberation", 1, 7, true),
            metric("codex", "synthesis-owner", "deliberation", 0, 8, true),
            metric("codex", "synthesis-owner", "deliberation", 2, 6, true),
            metric("codex", "synthesis-owner", "deliberation", 1, 7, true),
            metric("codex", "synthesis-owner", "deliberation", 0, 9, true),
            metric("codex", "synthesis-owner", "deliberation", 1, 6, true),
            metric("codex", "synthesis-owner", "deliberation", 0, 8, true),
        ];

        // Test across multiple task types — telemetry weight is 0.8 at 7+ samples.
        for task in ["planning", "rust code implementation", "safety review"] {
            let scores = RoleFitness::from_metrics(task, &strong_telemetry);
            let gemini = scores.iter().find(|s| s.model == "gemini" && s.role == "synthesizer").unwrap();
            let codex = scores.iter().find(|s| s.model == "codex" && s.role == "synthesizer").unwrap();
            assert!(
                gemini.score > codex.score,
                "task={task}: Gemini synthesizer ({}) should beat Codex ({}) with strong telemetry",
                gemini.score, codex.score
            );
        }
    }

    // ── Proof I: mixed-quality telemetry across runs ───────────────────
    //
    // Real-world telemetry has variance: some runs are great, some are poor.
    // This proves the scoring system handles mixed quality gracefully.

    #[test]
    fn proof_i_mixed_telemetry_produces_intermediate_score() {
        // Pure good telemetry
        let all_good = vec![
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 8, 1, true),
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 7, 1, true),
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 9, 0, true),
        ];
        // Pure bad telemetry
        let all_bad = vec![
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 1, 7, true),
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 0, 8, true),
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 1, 6, true),
        ];
        // Mixed: 2 good + 1 bad
        let mixed = vec![
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 8, 1, true),
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 7, 1, true),
            metric("claude", "adversarial-reviewer/conceptual-risk", "deliberation", 1, 7, true),
        ];

        let find = |scores: &[ModelRoleFitScore]| -> f64 {
            scores.iter().find(|s| s.model == "claude" && s.role == "critic").unwrap().score
        };

        let good_score = find(&RoleFitness::from_metrics("planning", &all_good));
        let bad_score = find(&RoleFitness::from_metrics("planning", &all_bad));
        let mixed_score = find(&RoleFitness::from_metrics("planning", &mixed));

        // Mixed should fall between good and bad
        assert!(
            mixed_score < good_score,
            "mixed ({mixed_score}) should be less than all-good ({good_score})"
        );
        assert!(
            mixed_score > bad_score,
            "mixed ({mixed_score}) should be more than all-bad ({bad_score})"
        );
    }

    #[test]
    fn proof_i_single_bad_run_doesnt_destroy_score() {
        // 4 good runs + 1 terrible run
        let mostly_good = vec![
            metric("codex", "synthesis-owner", "deliberation", 6, 1, true),
            metric("codex", "synthesis-owner", "deliberation", 5, 1, true),
            metric("codex", "synthesis-owner", "deliberation", 7, 0, true),
            metric("codex", "synthesis-owner", "deliberation", 6, 0, true),
            metric("codex", "synthesis-owner", "deliberation", 0, 8, false), // terrible run
        ];

        let find = |scores: &[ModelRoleFitScore]| -> f64 {
            scores.iter().find(|s| s.model == "codex" && s.role == "synthesizer").unwrap().score
        };

        let mixed_score = find(&RoleFitness::from_metrics("planning", &mostly_good));
        // The heuristic-only score for Codex synthesizer
        let heuristic_score = find(&RoleFitness::from_metrics("planning", &[]));

        // Despite one terrible run, the 4 good runs should keep the score reasonable.
        // Score should still be within 0.15 of the heuristic baseline.
        assert!(
            (mixed_score - heuristic_score).abs() < 0.15,
            "one bad run shouldn't destroy score: mixed={mixed_score}, heuristic={heuristic_score}"
        );
    }

    #[test]
    fn proof_i_increasing_sample_count_converges_toward_telemetry() {
        // With 1 sample, telemetry weight is 0.25 — mostly heuristic.
        // With 7+ samples, telemetry weight is 0.80 — mostly telemetry.
        // All samples show perfect performance → score should increase with more samples.
        let find = |scores: &[ModelRoleFitScore]| -> f64 {
            scores.iter().find(|s| s.model == "gemini" && s.role == "critic").unwrap().score
        };

        let one = vec![
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 10, 0, true),
        ];
        let three = vec![
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 10, 0, true),
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 9, 0, true),
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 10, 0, true),
        ];
        let eight = vec![
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 10, 0, true),
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 9, 0, true),
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 10, 0, true),
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 9, 1, true),
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 10, 0, true),
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 8, 1, true),
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 10, 0, true),
            metric("gemini", "adversarial-reviewer/implementation-risk", "deliberation", 9, 0, true),
        ];

        let s1 = find(&RoleFitness::from_metrics("planning", &one));
        let s3 = find(&RoleFitness::from_metrics("planning", &three));
        let s8 = find(&RoleFitness::from_metrics("planning", &eight));

        // Perfect telemetry should exceed the heuristic baseline (Gemini critic is 0.78 heuristic).
        // More samples → more telemetry weight → higher score for perfect data.
        assert!(
            s3 > s1 || (s3 - s1).abs() < 0.01,
            "3 samples should score >= 1 sample: {s3} vs {s1}"
        );
        assert!(
            s8 > s1,
            "8 perfect samples should clearly beat 1: {s8} > {s1}"
        );
    }
}
