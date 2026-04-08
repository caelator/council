mod critique;
mod metrics;
mod model;
mod model_role_fit;
mod phase;
mod provider;
mod trace;

use critique::{
    CURRENT_SCHEMA_VERSION, ContextPayload, DecisionLog, Disposition, Severity, StructuredCritique,
    check_decision_log_version, is_converged, parse_critique, parse_decision_log,
    read_context_payload,
};
use metrics::{
    PhaseMetrics, aggregate_profiles, collect_critique_metrics, collect_output_metrics,
    format_fit_report, format_recommendations, format_scorecard, load_metrics, recommend_roles,
    score_model_role_fit,
};
use model::{Model, run_model};
use model_role_fit::RoleFitness;
use phase::fmt_prompt;
use std::fs;
use std::path::{Path, PathBuf};
use trace::{
    LearningRecord, PlanRecord, RunTrace, append_jsonl, generate_run_id, iso_now,
    persona_assignment,
};

struct Config {
    workdir: PathBuf,
    task: String,
    max_rounds: u32,
    council_dir: PathBuf,
    run_id: String,
    resume: bool,
    trace_file: PathBuf,
    plans_file: PathBuf,
    learnings_file: PathBuf,
    metrics_file: PathBuf,
    fit_scores_file: PathBuf,
}

/// Resolve a roster slot model, allowing env-var overrides.
///
/// For a role like "framing-controller", checks `COUNCIL_MODEL_FRAMING_CONTROLLER`
/// for a model name or OpenRouter free model ID.  Falls back to `default`.
///
/// This lets users swap any roster slot to a free-tier model without code changes:
///   COUNCIL_MODEL_FRAMING_CONTROLLER=deepseek-r1-0528 council ...
fn resolve_roster_model(role: &str, default: Model) -> Model {
    let env_key = format!(
        "COUNCIL_MODEL_{}",
        role.to_ascii_uppercase().replace('-', "_").replace('/', "_")
    );
    if let Ok(val) = std::env::var(&env_key) {
        if let Some(m) = Model::from_name(&val) {
            if m.is_remote() && !model::check_available(m) {
                eprintln!(
                    "  [warn] {env_key}={val} resolved but OpenRouter is not available, using default {}",
                    default.name()
                );
                return default;
            }
            eprintln!("  [override] {env_key}={val} → {}", m.name());
            return m;
        }
        eprintln!(
            "  [warn] {env_key}={val} is not a known model, using default {}",
            default.name()
        );
    }
    default
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Support --resume or --context-file as args
    let mut resume = std::env::var("COUNCIL_RESUME")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);
    let mut context_file: Option<String> = None;
    let mut list_models = false;
    let mut real_args = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--resume" {
            resume = true;
        } else if args[i] == "--list-models" {
            list_models = true;
        } else if args[i] == "--context-file" && i + 1 < args.len() {
            context_file = Some(args[i + 1].clone());
            i += 1;
        } else {
            real_args.push(&args[i]);
        }
        i += 1;
    }

    if list_models {
        print_model_catalog();
        return Ok(());
    }

    if real_args.len() < 2 {
        eprintln!("Usage: council [--resume] [--list-models] [--context-file <path>] <workdir> <task prompt>");
        eprintln!(
            "Optional env: COUNCIL_ROUNDS (default 2), COUNCIL_DIR, COUNCIL_NAME, COUNCIL_RESUME=1"
        );
        eprintln!("  Roster overrides: COUNCIL_MODEL_<ROLE>=<model-name>");
        std::process::exit(1);
    }

    let workdir = PathBuf::from(real_args[0]);
    if !workdir.is_dir() {
        eprintln!("Workdir does not exist: {}", workdir.display());
        std::process::exit(1);
    }

    let task: String = real_args[1..]
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    // Read external context if provided
    let external_context = if let Some(path) = context_file {
        fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("Warning: failed to read context file {}: {}", path, e);
            String::new()
        })
    } else {
        String::new()
    };
    let max_rounds: u32 = std::env::var("COUNCIL_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let run_id = generate_run_id();
    let slug = slugify(&task);
    let name = std::env::var("COUNCIL_NAME").unwrap_or_else(|_| slug);

    let council_dir = std::env::var("COUNCIL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workdir.join(".council").join(&name));

    let memoryport = workdir.join("memoryport");

    let config = Config {
        workdir: workdir.clone(),
        task: task.clone(),
        max_rounds,
        council_dir: council_dir.clone(),
        run_id: run_id.clone(),
        resume,
        trace_file: memoryport.join("council-traces.jsonl"),
        plans_file: memoryport.join("council-plans.jsonl"),
        learnings_file: memoryport.join("council-learnings.jsonl"),
        metrics_file: memoryport.join("council-metrics.jsonl"),
        fit_scores_file: memoryport.join("council-fit-scores.jsonl"),
    };

    if resume {
        eprintln!(
            "[resume mode] Checking for existing artifacts in {}",
            council_dir.display()
        );
        // Try to load and validate the existing run-summary.json for schema negotiation
        let summary_path = council_dir.join("run-summary.json");
        if summary_path.exists() {
            match read_context_payload(&summary_path) {
                Ok(payload) => {
                    eprintln!(
                        "  [resume] Loaded run-summary.json (schema v{}, run_id={})",
                        payload.schema_version, payload.run_id
                    );
                }
                Err(e) => {
                    eprintln!("  [resume] Could not parse run-summary.json: {}", e);
                    eprintln!("  [resume] Will overwrite with fresh run data");
                }
            }
        }
    }

    fs::create_dir_all(&config.council_dir).expect("Failed to create council directory");
    fs::create_dir_all(&memoryport).expect("Failed to create memoryport directory");

    // Save task
    write_artifact(&config.council_dir, "task.txt", &config.task);

    // Check model availability — enumerate all known models (local + free catalog).
    let all_models = Model::all_models();
    for m in &all_models {
        let available = model::check_available(*m);
        if available {
            eprintln!("  [ok] {} available{}", m.name(),
                if m.is_remote() { " (openrouter)" } else { "" });
        } else {
            eprintln!("  [!!] {} not found", m.name());
        }
    }

    // ── Load historical metrics and print scorecard ────────────────────
    let historical_metrics = load_metrics(&config.metrics_file);
    if !historical_metrics.is_empty() {
        let profiles = aggregate_profiles(&historical_metrics);
        eprintln!("\n{}", format_scorecard(&profiles));

        // Current roster (defaults, before env overrides apply).
        let current_assignments = vec![
            (
                Model::Claude,
                "adversarial-reviewer/conceptual-risk",
                "deliberation",
            ),
            (
                Model::Gemini,
                "adversarial-reviewer/implementation-risk",
                "deliberation",
            ),
            (Model::Codex, "synthesis-owner", "deliberation"),
            (Model::Gemini, "framing-controller", "framing"),
            (Model::Gemini, "solution-scout", "brainstorming"),
            (Model::Claude, "elegance-scout", "brainstorming"),
            (Model::Codex, "feasibility-scout", "brainstorming"),
            (Model::Codex, "build-lead", "handoff"),
        ];
        let recommendations = recommend_roles(&profiles, &current_assignments, 2, 0.2);
        eprintln!("{}", format_recommendations(&recommendations));

        if recommendations.iter().any(|r| r.confidence >= 0.5) {
            eprintln!("  [!] High-confidence role swaps detected. Review before proceeding.\n");
        }
    } else {
        eprintln!("\n  [info] No historical metrics found. Metrics will be collected this run.\n");
    }

    let mut run_metrics: Vec<PhaseMetrics> = Vec::new();
    let mut personas = Vec::new();
    let mut phases_completed = Vec::new();

    // ── Resolve roster models (env overrides applied here) ────────────
    let framing_model = resolve_roster_model("framing-controller", Model::Gemini);
    let conceptual_critic = resolve_roster_model("adversarial-reviewer/conceptual-risk", Model::Claude);
    let impl_critic = resolve_roster_model("adversarial-reviewer/implementation-risk", Model::Gemini);
    let synthesis_model = resolve_roster_model("synthesis-owner", Model::Codex);
    let handoff_model = resolve_roster_model("build-lead", Model::Codex);

    // ── Stage 0: Framing ──────────────────────────────────────────────
    eprintln!("\n═══ Stage 0: Framing ═══");
    let framing = if config.resume {
        if let Some(existing) = try_resume_artifact(&config.council_dir, "stage0-framing.md") {
            eprintln!(
                "  [resume] Loaded existing stage0-framing.md ({} bytes)",
                existing.len()
            );
            existing
        } else {
            let framing_prompt = fmt_prompt(
                phase::FRAMING_PROMPT,
                &[
                    ("task", &config.task),
                    ("external_context", &external_context),
                ],
            );
            let f = run_phase_model(
                framing_model,
                &framing_prompt,
                &config.workdir,
                &config.council_dir,
                "stage0-framing",
            );
            write_artifact(&config.council_dir, "problem-brief.md", &f);
            f
        }
    } else {
        let framing_prompt = fmt_prompt(
            phase::FRAMING_PROMPT,
            &[
                ("task", &config.task),
                ("external_context", &external_context),
            ],
        );
        let f = run_phase_model(
            framing_model,
            &framing_prompt,
            &config.workdir,
            &config.council_dir,
            "stage0-framing",
        );
        write_artifact(&config.council_dir, "problem-brief.md", &f);
        f
    };
    run_metrics.push(collect_output_metrics(
        &config.run_id,
        &iso_now(),
        framing_model,
        "framing-controller",
        "framing",
        framing.len() as u32,
        framing.len() > 20 && framing.contains("##"),
    ));
    phases_completed.push("framing".into());
    personas.push(persona_assignment(
        framing_model,
        "framing-controller",
        "framing",
    ));

    // ── Stage 1: Brainstorming ────────────────────────────────────────
    eprintln!("\n═══ Stage 1: Brainstorming ═══");
    let stage1_dir = config.council_dir.join("stage1-brainstorm");
    fs::create_dir_all(&stage1_dir).ok();

    let brainstorm_plan = if config.resume {
        if let Some(existing) = try_resume_artifact(&config.council_dir, "brainstorm-plan-v1.md") {
            eprintln!(
                "  [resume] Loaded existing brainstorm-plan-v1.md ({} bytes)",
                existing.len()
            );
            existing
        } else {
            run_brainstorm_stage(
                &config,
                &stage1_dir,
                &framing,
                &mut personas,
                &mut run_metrics,
            )
        }
    } else {
        run_brainstorm_stage(
            &config,
            &stage1_dir,
            &framing,
            &mut personas,
            &mut run_metrics,
        )
    };
    phases_completed.push("brainstorming".into());

    // ── Stage 2: Adversarial Deliberation ─────────────────────────────
    eprintln!("\n═══ Stage 2: Adversarial Deliberation ═══");
    let mut current_plan = brainstorm_plan;
    let mut converged = false;
    let mut total_rounds = 0u32;

    // On resume, find the last completed round and load its plan
    let resume_from_round = if config.resume {
        let mut last_complete = 0u32;
        for r in 1..=config.max_rounds {
            let round_dir = config.council_dir.join(format!("stage2-round{r}"));
            if let Some(rev) = try_resume_artifact(&round_dir, "codex-revision.md") {
                eprintln!("  [resume] Found completed round {r} ({} bytes)", rev.len());
                current_plan = rev;
                last_complete = r;
            } else {
                break;
            }
        }
        // Also check for the round-level plan artifact
        for r in (1..=last_complete).rev() {
            if let Some(bp) =
                try_resume_artifact(&config.council_dir, &format!("build-plan-round{r}.md"))
            {
                current_plan = bp;
                break;
            }
        }
        last_complete
    } else {
        0
    };

    for round in 1..=config.max_rounds {
        // Skip already-completed rounds on resume
        if round <= resume_from_round {
            total_rounds = round;
            continue;
        }
        total_rounds = round;
        let round_dir = config.council_dir.join(format!("stage2-round{round}"));
        fs::create_dir_all(&round_dir).ok();
        eprintln!("  ── Round {round}/{} ──", config.max_rounds);

        // Conceptual adversary
        eprintln!("  → {}: conceptual-adversary", conceptual_critic.name());
        let claude_crit_prompt = fmt_prompt(
            phase::CRITIQUE_PROMPT,
            &[
                ("role", "adversarial-reviewer/conceptual-risk"),
                (
                    "role_description",
                    "Find conceptual flaws, ambiguity, hidden coupling, missing edge cases",
                ),
                ("plan", &current_plan),
                ("task", &config.task),
            ],
        );
        let claude_crit_raw = run_phase_model(
            conceptual_critic,
            &claude_crit_prompt,
            &config.workdir,
            &round_dir,
            "claude-critique",
        );
        let claude_critique =
            parse_critique(&claude_crit_raw, "adversarial-reviewer/conceptual-risk");
        write_artifact(
            &round_dir,
            "claude-critique.json",
            &serde_json::to_string_pretty(&claude_critique).unwrap_or_default(),
        );

        // Implementation-risk adversary
        eprintln!("  → {}: risk-adversary", impl_critic.name());
        let gemini_crit_prompt = fmt_prompt(
            phase::CRITIQUE_PROMPT,
            &[
                ("role", "adversarial-reviewer/implementation-risk"),
                (
                    "role_description",
                    "Find implementation risks, complexity traps, scaling issues, missing validation",
                ),
                ("plan", &current_plan),
                ("task", &config.task),
            ],
        );
        let gemini_crit_raw = run_phase_model(
            impl_critic,
            &gemini_crit_prompt,
            &config.workdir,
            &round_dir,
            "gemini-critique",
        );
        let gemini_critique =
            parse_critique(&gemini_crit_raw, "adversarial-reviewer/implementation-risk");
        write_artifact(
            &round_dir,
            "gemini-critique.json",
            &serde_json::to_string_pretty(&gemini_critique).unwrap_or_default(),
        );

        // Check convergence before synthesis
        let claude_converged = is_converged(&claude_critique);
        let gemini_converged = is_converged(&gemini_critique);
        let critique_count = claude_critique.critiques.len() + gemini_critique.critiques.len();
        let high_count =
            count_high_severity(&claude_critique) + count_high_severity(&gemini_critique);

        eprintln!("  → Critiques: {critique_count} total, {high_count} high-severity");

        if claude_converged && gemini_converged {
            eprintln!("  → Converged: all critiques are low severity");
            converged = true;

            append_jsonl(
                &config.learnings_file,
                &LearningRecord {
                    run_id: config.run_id.clone(),
                    timestamp: iso_now(),
                    kind: "convergence".into(),
                    summary: format!("Converged after {round} deliberation round(s)"),
                    task_type: "planning".into(),
                    model: "all".into(),
                    role: "deliberation".into(),
                },
            )?;
            break;
        }

        // Synthesis owner — process critiques and revise plan
        eprintln!("  → {}: synthesis-owner (revise plan)", synthesis_model.name());
        let combined_critiques = format!(
            "## {} (conceptual-risk)\n{}\n\n## {} (implementation-risk)\n{}",
            conceptual_critic.name(),
            serde_json::to_string_pretty(&claude_critique).unwrap_or_default(),
            impl_critic.name(),
            serde_json::to_string_pretty(&gemini_critique).unwrap_or_default()
        );
        let delib_prompt = fmt_prompt(
            phase::DELIBERATION_LEAD_PROMPT,
            &[
                ("plan", &current_plan),
                ("critiques", &combined_critiques),
                ("task", &config.task),
            ],
        );
        let revision = run_phase_model(
            synthesis_model,
            &delib_prompt,
            &config.workdir,
            &round_dir,
            "codex-revision",
        );

        // Detect material change
        let material_change =
            revision.len().abs_diff(current_plan.len()) > current_plan.len() / 20 || high_count > 0;

        let decisions = parse_decision_log(&revision);
        let decision_log = DecisionLog {
            round,
            decisions,
            material_change,
            schema_version: CURRENT_SCHEMA_VERSION,
        };
        check_decision_log_version(&decision_log);
        write_artifact(
            &round_dir,
            "decision-log.json",
            &serde_json::to_string_pretty(&decision_log).unwrap_or_default(),
        );

        // ── Collect per-model metrics for this round ────────────────
        let ts = iso_now();
        let decision_tuples: Vec<(String, Disposition, Severity)> = decision_log
            .decisions
            .iter()
            .map(|d| (d.issue.clone(), d.disposition, d.severity))
            .collect();

        // Partition decisions by source model for accurate per-critic metrics
        let claude_decisions: Vec<_> = decision_log
            .decisions
            .iter()
            .filter(|d| d.source_model.as_deref() == Some("claude"))
            .map(|d| (d.issue.clone(), d.disposition, d.severity))
            .collect();
        let gemini_decisions: Vec<_> = decision_log
            .decisions
            .iter()
            .filter(|d| d.source_model.as_deref() == Some("gemini"))
            .map(|d| (d.issue.clone(), d.disposition, d.severity))
            .collect();

        // Use source-partitioned decisions if available, fall back to fuzzy matching all
        let claude_decs = if claude_decisions.is_empty() {
            &decision_tuples
        } else {
            &claude_decisions
        };
        let gemini_decs = if gemini_decisions.is_empty() {
            &decision_tuples
        } else {
            &gemini_decisions
        };

        run_metrics.push(collect_critique_metrics(
            &config.run_id,
            &ts,
            conceptual_critic,
            "adversarial-reviewer/conceptual-risk",
            "deliberation",
            &claude_critique,
            claude_decs,
            claude_crit_raw.len() as u32,
            !claude_critique
                .critiques
                .first()
                .is_some_and(|c| c.issue.starts_with("Unstructured")),
        ));
        run_metrics.push(collect_critique_metrics(
            &config.run_id,
            &ts,
            impl_critic,
            "adversarial-reviewer/implementation-risk",
            "deliberation",
            &gemini_critique,
            gemini_decs,
            gemini_crit_raw.len() as u32,
            !gemini_critique
                .critiques
                .first()
                .is_some_and(|c| c.issue.starts_with("Unstructured")),
        ));
        run_metrics.push(collect_output_metrics(
            &config.run_id,
            &ts,
            synthesis_model,
            "synthesis-owner",
            "deliberation",
            revision.len() as u32,
            !decision_log.decisions.is_empty(),
        ));

        if !material_change && round > 1 {
            eprintln!("  → No material change detected, treating as converged");
            converged = true;
            current_plan = revision;
            break;
        }

        current_plan = revision;
        write_artifact(
            &config.council_dir,
            &format!("build-plan-round{round}.md"),
            &current_plan,
        );
    }

    phases_completed.push("deliberation".into());

    // ── Stage 3: Build Handoff ────────────────────────────────────────
    eprintln!("\n═══ Stage 3: Build Handoff ═══");
    let handoff = if config.resume {
        if let Some(existing) = try_resume_artifact(&config.council_dir, "stage3-handoff.md") {
            eprintln!(
                "  [resume] Loaded existing stage3-handoff.md ({} bytes)",
                existing.len()
            );
            existing
        } else {
            let handoff_prompt = fmt_prompt(
                phase::HANDOFF_PROMPT,
                &[("plan", &current_plan), ("task", &config.task)],
            );
            run_phase_model(
                handoff_model,
                &handoff_prompt,
                &config.workdir,
                &config.council_dir,
                "stage3-handoff",
            )
        }
    } else {
        let handoff_prompt = fmt_prompt(
            phase::HANDOFF_PROMPT,
            &[("plan", &current_plan), ("task", &config.task)],
        );
        run_phase_model(
            handoff_model,
            &handoff_prompt,
            &config.workdir,
            &config.council_dir,
            "stage3-handoff",
        )
    };
    run_metrics.push(collect_output_metrics(
        &config.run_id,
        &iso_now(),
        handoff_model,
        "build-lead",
        "handoff",
        handoff.len() as u32,
        handoff.contains("## Implementation Checklist"),
    ));
    personas.push(persona_assignment(handoff_model, "build-lead", "handoff"));

    write_artifact(&config.council_dir, "final-plan.md", &handoff);
    write_artifact(&config.council_dir, "build-plan-final.md", &current_plan);
    phases_completed.push("handoff".into());

    // ── Telemetry ─────────────────────────────────────────────────────
    let ts = iso_now();
    let models_used: Vec<String> = Model::all_models()
        .into_iter()
        .filter(|m| model::check_available(*m))
        .map(|m| m.name().into())
        .collect();

    let run_trace = RunTrace {
        run_id: config.run_id.clone(),
        timestamp: ts.clone(),
        task: config.task.clone(),
        task_type: "planning".into(),
        phases_completed,
        models_used,
        persona_assignments: personas,
        rounds: total_rounds,
        converged,
        artifacts_dir: config.council_dir.display().to_string(),
        final_plan_path: config
            .council_dir
            .join("final-plan.md")
            .display()
            .to_string(),
    };
    append_jsonl(&config.trace_file, &run_trace)?;

    let plan_content =
        fs::read_to_string(config.council_dir.join("final-plan.md")).unwrap_or_default();
    let plan_record = PlanRecord {
        run_id: config.run_id.clone(),
        timestamp: ts,
        task: config.task.clone(),
        task_type: "planning".into(),
        plan_markdown: plan_content,
        artifacts_dir: config.council_dir.display().to_string(),
    };
    append_jsonl(&config.plans_file, &plan_record)?;

    // ── Emit per-model metrics ─────────────────────────────────────────
    for m in &run_metrics {
        append_jsonl(&config.metrics_file, m)?;
    }
    eprintln!(
        "  Metrics: {} records written to {}",
        run_metrics.len(),
        config.metrics_file.display()
    );

    // Print end-of-run scorecard with this run's data included
    let mut all_metrics = historical_metrics;
    all_metrics.extend(run_metrics);
    let final_profiles = aggregate_profiles(&all_metrics);
    eprintln!("\n{}", format_scorecard(&final_profiles));

    // Print model-role fit analysis with zero-acceptance diagnostics
    let fits = score_model_role_fit(&all_metrics);
    eprintln!("{}", format_fit_report(&fits));

    let role_fitness = RoleFitness::from_metrics("planning", &all_metrics);
    eprintln!("Top model-role fits (planning):");
    for fit in role_fitness.iter().take(6) {
        eprintln!(
            "  - {:7} as {:12} score {:.0}% (confidence {:.0}%, n={})",
            fit.model,
            fit.role,
            fit.score * 100.0,
            fit.confidence * 100.0,
            fit.sample_count,
        );
    }

    // Persist fit scores to JSONL for cross-run querying
    for fit in &role_fitness {
        let record = serde_json::json!({
            "run_id": config.run_id,
            "timestamp": iso_now(),
            "model": fit.model,
            "role": fit.role,
            "score": fit.score,
            "confidence": fit.confidence,
            "sample_count": fit.sample_count,
        });
        if let Err(e) = append_jsonl(&config.fit_scores_file, &record) {
            eprintln!("  [warn] failed to write fit score: {e}");
        }
    }
    eprintln!(
        "  Fit scores: {} records written to {}",
        role_fitness.len(),
        config.fit_scores_file.display()
    );

    // ── Summary ───────────────────────────────────────────────────────
    let summary = ContextPayload {
        run_id: config.run_id.clone(),
        task: config.task.clone(),
        rounds: total_rounds,
        converged,
        council_dir: config.council_dir.display().to_string(),
        final_plan: config
            .council_dir
            .join("final-plan.md")
            .display()
            .to_string(),
        schema_version: CURRENT_SCHEMA_VERSION,
    };
    write_artifact(
        &config.council_dir,
        "run-summary.json",
        &serde_json::to_string_pretty(&summary).unwrap_or_default(),
    );

    eprintln!("\n═══ Council Complete ═══");
    eprintln!("  Rounds: {total_rounds}, Converged: {converged}");
    eprintln!(
        "  Final plan: {}",
        config.council_dir.join("final-plan.md").display()
    );

    // Print final plan path to stdout (for scripting)
    println!("{}", config.council_dir.join("final-plan.md").display());
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────

fn print_model_catalog() {
    use crate::provider::free_model_registry;

    println!("═══ Council Model Catalog ═══\n");
    println!("Local models:");
    for (model, cmd) in [
        (Model::Gemini, "gemini"),
        (Model::Claude, "claude"),
        (Model::Codex, "codex"),
        (Model::Gemma31B, "gemma-llama"),
    ] {
        let avail = model::check_available(model);
        let status = if avail { "ok" } else { "not found" };
        println!("  [{status}] {:<16} (CLI: {cmd})", model.name());
    }

    println!("\nOpenRouter free-tier models:");
    let or_available = provider::resolve_openrouter_api_key().is_some();
    for entry in free_model_registry() {
        let model = if entry.id == "qwen/qwen3.6-plus" {
            Model::Qwen36Plus
        } else {
            Model::OpenRouterFree(entry.id)
        };
        let status = if or_available { "ok" } else { "no key" };
        let reasoning = if entry.reasoning { " [reasoning]" } else { "" };
        println!(
            "  [{status}] {:<28} → {}{reasoning}",
            model.name(),
            entry.id,
        );
    }

    println!("\nRoster slot overrides (env vars):");
    let slots = [
        ("COUNCIL_MODEL_FRAMING_CONTROLLER", "framing-controller", "gemini"),
        ("COUNCIL_MODEL_SOLUTION_SCOUT", "solution-scout", "gemini"),
        ("COUNCIL_MODEL_ELEGANCE_SCOUT", "elegance-scout", "claude"),
        ("COUNCIL_MODEL_FEASIBILITY_SCOUT", "feasibility-scout", "codex"),
        ("COUNCIL_MODEL_RISK_SCOUT_1", "risk-scout-1", "gemma-llama"),
        ("COUNCIL_MODEL_RISK_SCOUT_2", "risk-scout-2", "qwen-36-plus"),
        ("COUNCIL_MODEL_ADVERSARIAL_REVIEWER_CONCEPTUAL_RISK", "adversarial-reviewer/conceptual-risk", "claude"),
        ("COUNCIL_MODEL_ADVERSARIAL_REVIEWER_IMPLEMENTATION_RISK", "adversarial-reviewer/implementation-risk", "gemini"),
        ("COUNCIL_MODEL_SYNTHESIS_OWNER", "synthesis-owner", "codex"),
        ("COUNCIL_MODEL_BUILD_LEAD", "build-lead", "codex"),
    ];
    for (env, role, default) in slots {
        println!("  {env}  (role: {role}, default: {default})");
    }
    println!("\nExample: COUNCIL_MODEL_FRAMING_CONTROLLER=deepseek-r1-0528 council ...");
}

fn run_brainstorm_stage(
    config: &Config,
    stage1_dir: &Path,
    framing: &str,
    personas: &mut Vec<trace::PersonaAssignment>,
    run_metrics: &mut Vec<PhaseMetrics>,
) -> String {
    // Resolve brainstorm roster (env overrides apply).
    let seed_model = resolve_roster_model("solution-scout", Model::Gemini);
    let elegance_model = resolve_roster_model("elegance-scout", Model::Claude);
    let feasibility_model = resolve_roster_model("feasibility-scout", Model::Codex);
    let risk_scout_1 = resolve_roster_model("risk-scout-1", Model::Gemma31B);
    let risk_scout_2 = resolve_roster_model("risk-scout-2", Model::Qwen36Plus);

    // Seed the plan
    eprintln!("  → {}: solution-scout (seed plan)", seed_model.name());
    let seed_prompt = fmt_prompt(
        phase::BRAINSTORM_SEED_PROMPT,
        &[("framing", framing), ("task", &config.task)],
    );
    let seed_plan = run_phase_model(
        seed_model,
        &seed_prompt,
        &config.workdir,
        stage1_dir,
        "gemini-seed",
    );
    personas.push(persona_assignment(
        seed_model,
        "solution-scout",
        "brainstorming",
    ));

    // Elegance/reframing contributions
    eprintln!("  → {}: elegance-scout (contribute)", elegance_model.name());
    let claude_contrib_prompt = fmt_prompt(
        phase::BRAINSTORM_CONTRIBUTE_PROMPT,
        &[
            ("role", "elegance-scout"),
            (
                "role_description",
                "Find hidden assumptions, simpler abstractions, and reframing opportunities",
            ),
            ("framing", framing),
            ("plan", &seed_plan),
            ("task", &config.task),
        ],
    );
    let claude_contributions = run_phase_model(
        elegance_model,
        &claude_contrib_prompt,
        &config.workdir,
        stage1_dir,
        "claude-contribute",
    );
    personas.push(persona_assignment(
        elegance_model,
        "elegance-scout",
        "brainstorming",
    ));

    // Feasibility contributions
    eprintln!("  → {}: feasibility-scout (contribute)", feasibility_model.name());
    let codex_contrib_prompt = fmt_prompt(
        phase::BRAINSTORM_CONTRIBUTE_PROMPT,
        &[
            ("role", "feasibility-scout"),
            (
                "role_description",
                "Check implementation realism, flag complexity risks, suggest simplifications",
            ),
            ("framing", framing),
            ("plan", &seed_plan),
            ("task", &config.task),
        ],
    );
    let codex_contributions = run_phase_model(
        feasibility_model,
        &codex_contrib_prompt,
        &config.workdir,
        stage1_dir,
        "codex-contribute",
    );
    personas.push(persona_assignment(
        feasibility_model,
        "feasibility-scout",
        "brainstorming",
    ));

    // Risk/failure-mode analysis (model 1)
    eprintln!("  → {}: risk-scout (contribute)", risk_scout_1.name());
    let gemma_contrib_prompt = fmt_prompt(
        phase::BRAINSTORM_CONTRIBUTE_PROMPT,
        &[
            ("role", "risk-scout"),
            (
                "role_description",
                "Identify failure modes, edge cases, and what could go wrong with this plan",
            ),
            ("framing", framing),
            ("plan", &seed_plan),
            ("task", &config.task),
        ],
    );
    let gemma_contributions = run_phase_model(
        risk_scout_1,
        &gemma_contrib_prompt,
        &config.workdir,
        stage1_dir,
        "gemma-contribute",
    );
    personas.push(persona_assignment(
        risk_scout_1,
        "risk-scout",
        "brainstorming",
    ));

    // Risk/reasoning analysis (model 2, default: Qwen via OpenRouter)
    eprintln!("  → {}: risk-scout (contribute)", risk_scout_2.name());
    let qwen_contrib_prompt = fmt_prompt(
        phase::BRAINSTORM_CONTRIBUTE_PROMPT,
        &[
            ("role", "risk-scout"),
            (
                "role_description",
                "Identify failure modes, edge cases, and what could go wrong with this plan. Use chain-of-thought reasoning to find subtle issues",
            ),
            ("framing", framing),
            ("plan", &seed_plan),
            ("task", &config.task),
        ],
    );
    let qwen_contributions = run_phase_model(
        risk_scout_2,
        &qwen_contrib_prompt,
        &config.workdir,
        stage1_dir,
        "qwen-contribute",
    );
    personas.push(persona_assignment(
        risk_scout_2,
        "risk-scout",
        "brainstorming",
    ));

    // Record brainstorm contributor metrics
    let ts = iso_now();
    run_metrics.push(collect_output_metrics(
        &config.run_id,
        &ts,
        seed_model,
        "solution-scout",
        "brainstorming",
        seed_plan.len() as u32,
        seed_plan.len() > 20 && seed_plan.contains("##"),
    ));
    run_metrics.push(collect_output_metrics(
        &config.run_id,
        &ts,
        elegance_model,
        "elegance-scout",
        "brainstorming",
        claude_contributions.len() as u32,
        claude_contributions.len() > 20,
    ));
    run_metrics.push(collect_output_metrics(
        &config.run_id,
        &ts,
        feasibility_model,
        "feasibility-scout",
        "brainstorming",
        codex_contributions.len() as u32,
        codex_contributions.len() > 20,
    ));
    run_metrics.push(collect_output_metrics(
        &config.run_id,
        &ts,
        risk_scout_1,
        "risk-scout",
        "brainstorming",
        gemma_contributions.len() as u32,
        gemma_contributions.len() > 20,
    ));
    run_metrics.push(collect_output_metrics(
        &config.run_id,
        &ts,
        risk_scout_2,
        "risk-scout",
        "brainstorming",
        qwen_contributions.len() as u32,
        qwen_contributions.len() > 20,
    ));

    // Synthesize
    eprintln!("  → {}: synthesizing contributions", seed_model.name());
    let synth_prompt = format!(
        "You are the brainstorming lead synthesizing contributions into a revised plan.\n\n\
         Original plan:\n{seed_plan}\n\n\
         Elegance contributions:\n{claude_contributions}\n\n\
         Feasibility contributions:\n{codex_contributions}\n\n\
         Risk/failure-mode contributions ({}):\n{gemma_contributions}\n\n\
         Risk/failure-mode contributions ({}):\n{qwen_contributions}\n\n\
         Integrate the valuable suggestions. Produce the revised complete plan.\n\
         Mark sections that changed with [CHANGED] tags.",
        risk_scout_1.name(),
        risk_scout_2.name(),
    );
    let brainstorm_plan = run_phase_model(
        seed_model,
        &synth_prompt,
        &config.workdir,
        stage1_dir,
        "gemini-synthesis",
    );
    write_artifact(
        &config.council_dir,
        "brainstorm-plan-v1.md",
        &brainstorm_plan,
    );
    brainstorm_plan
}

/// Try to load an existing artifact for resume. Returns Some(content) if the file
/// exists and has meaningful content (> 20 bytes, not a skip/error marker).
fn try_resume_artifact(dir: &Path, name: &str) -> Option<String> {
    let path = dir.join(name);
    match fs::read_to_string(&path) {
        Ok(content) if content.len() > 20 => {
            // Reject error markers from failed model runs
            if content.starts_with('[')
                && !content.trim_start().starts_with('{')
                && !content.trim_start().starts_with('#')
            {
                return None;
            }
            if content.contains("unavailable — skipped]") || content.contains("produced no output]")
            {
                return None;
            }
            Some(content)
        }
        _ => None,
    }
}

fn run_phase_model(
    model: Model,
    prompt: &str,
    workdir: &Path,
    output_dir: &Path,
    artifact_prefix: &str,
) -> String {
    fs::create_dir_all(output_dir).ok();

    // Save prompt
    write_artifact(output_dir, &format!("{artifact_prefix}-prompt.txt"), prompt);

    if !model::check_available(model) {
        let msg = format!("[{} unavailable — skipped]", model.name());
        eprintln!("    {msg}");
        write_artifact(output_dir, &format!("{artifact_prefix}.md"), &msg);
        return msg;
    }

    match run_model(model, prompt, workdir) {
        Ok(output) => {
            write_artifact(
                output_dir,
                &format!("{artifact_prefix}.stderr"),
                &output.stderr,
            );
            let text = if output.success && !output.stdout.trim().is_empty() {
                output.stdout
            } else {
                eprintln!(
                    "    [{}] failed or empty output, using stderr as fallback",
                    model.name()
                );
                if output.stderr.trim().is_empty() {
                    format!("[{} produced no output]", model.name())
                } else {
                    format!("[{} failed]\nstderr:\n{}", model.name(), output.stderr)
                }
            };
            write_artifact(output_dir, &format!("{artifact_prefix}.md"), &text);
            text
        }
        Err(e) => {
            let msg = format!("[{} error: {e}]", model.name());
            eprintln!("    {msg}");
            write_artifact(output_dir, &format!("{artifact_prefix}.md"), &msg);
            msg
        }
    }
}

fn write_artifact(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    if let Err(e) = fs::write(&path, content) {
        eprintln!("[write_artifact] failed to write {}: {}", path.display(), e);
    }
}

fn slugify(text: &str) -> String {
    let slug: String = text
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    let slug = slug.trim_matches('-').to_string();
    if slug.len() > 64 {
        slug[..64].trim_end_matches('-').to_string()
    } else if slug.is_empty() {
        "council-run".to_string()
    } else {
        slug
    }
}

fn count_high_severity(critique: &StructuredCritique) -> usize {
    critique
        .critiques
        .iter()
        .filter(|c| matches!(c.severity, critique::Severity::High))
        .count()
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Proof B: roster override via env var ─────────────────────────

    #[test]
    fn proof_b_roster_override_env_resolves_free_model() {
        // Set an env var that resolve_roster_model checks.
        // Use a unique role name to avoid collisions with other tests.
        let role = "proof-test-scout";
        let env_key = "COUNCIL_MODEL_PROOF_TEST_SCOUT";

        // SAFETY: test-only, single-threaded test runner for this test.
        unsafe { std::env::set_var(env_key, "deepseek-r1-0528") };
        let resolved = resolve_roster_model(role, Model::Gemini);

        // The resolved model should be the DeepSeek free model, not the default Gemini.
        assert_ne!(resolved, Model::Gemini, "override should change default");
        assert!(resolved.is_remote(), "override should resolve to remote model");
        assert_eq!(
            resolved.openrouter_model_id(),
            Some("deepseek/deepseek-r1-0528:free"),
            "override should resolve to correct OpenRouter ID"
        );

        // Clean up env.
        unsafe { std::env::remove_var(env_key) };
    }

    #[test]
    fn proof_b_roster_override_invalid_falls_back_to_default() {
        let role = "proof-fallback-scout";
        let env_key = "COUNCIL_MODEL_PROOF_FALLBACK_SCOUT";

        // SAFETY: test-only, single-threaded test runner for this test.
        unsafe { std::env::set_var(env_key, "nonexistent-model-xyz") };
        let resolved = resolve_roster_model(role, Model::Claude);
        assert_eq!(
            resolved,
            Model::Claude,
            "invalid override should fall back to default"
        );

        unsafe { std::env::remove_var(env_key) };
    }

    #[test]
    fn proof_b_roster_override_absent_uses_default() {
        // No env var set → default is used.
        let role = "proof-absent-scout";
        let resolved = resolve_roster_model(role, Model::Codex);
        assert_eq!(resolved, Model::Codex);
    }

    #[test]
    fn proof_b_roster_env_key_format() {
        // Verify the env key derivation: "risk-scout/alpha" → "COUNCIL_MODEL_RISK_SCOUT_ALPHA"
        let role = "risk-scout/alpha";
        let env_key = format!(
            "COUNCIL_MODEL_{}",
            role.to_ascii_uppercase().replace('-', "_").replace('/', "_")
        );
        assert_eq!(env_key, "COUNCIL_MODEL_RISK_SCOUT_ALPHA");
    }
}
