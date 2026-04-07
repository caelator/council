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
        for model in ["claude", "gemini", "codex"] {
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

    let score = round3(
        avg_acceptance * 0.30
            + avg_severity_accuracy * 0.25
            + parse_reliability * 0.25
            + output_size_score * 0.20,
    );

    let confidence = round3(
        ((sample_count as f64 / 6.0).min(1.0) * 0.7 + parse_reliability * 0.3).clamp(0.2, 0.99),
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
        _ => CapabilityProfile {
            intelligence: 0.5,
            context: 0.5,
            speed: 0.5,
            cost_efficiency: 0.5,
            planning: 0.5,
            critique: 0.5,
            synthesis: 0.5,
            domain: 0.5,
            implementation: 0.5,
        },
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
}
