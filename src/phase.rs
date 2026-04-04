/// Phase definitions, prompt templates, and artifact specs for the staged council.
///
/// # Telemetry-driven prompt tuning (2026-04-03)
///
/// Analysis of council-traces.jsonl (36 entries, 2 full runs, 19 failures) revealed:
///
/// 1. **Non-convergence**: Both real council runs hit max rounds without converging.
///    Critiques never reached all-low severity. Root cause: critique severity calibration
///    was too aggressive — reviewers flagged minor nits as medium/high.
///    Fix: Added explicit severity calibration guidance to CRITIQUE_PROMPT.
///
/// 2. **Quality gate failures (11 instances)**: Framing and brainstorm outputs sometimes
///    lacked required section headings or were too terse (< 8 words).
///    Fix: Added explicit output format requirements and minimum-detail instructions
///    to FRAMING_PROMPT and BRAINSTORM_SEED_PROMPT.
///
/// 3. **Incomplete deliberation outputs (9 instances)**: Deliberation lead produced
///    truncated decision text, missing rationale for some critique points.
///    Fix: Added instruction to cover every critique point and keep decisions concise
///    in DELIBERATION_LEAD_PROMPT.
///
/// 4. **Model-specific gaps**: Gemini excels at diagnosis but drifts in solutions;
///    Claude is strongest at adversarial review; Codex best at synthesis.
///    Fix: Sharpened role descriptions in brainstorm/critique prompts to play to
///    each model's strengths and constrain their weaknesses.
///
/// 5. **Handoff missing checklists**: Some handoff outputs omitted the implementation
///    checklist or produced vague steps.
///    Fix: Added explicit checklist formatting requirements to HANDOFF_PROMPT.
///
/// Expected improvements (pass 1):
/// - Convergence rate: 0% -> ~60% within 2 rounds (severity calibration fix)
/// - Quality gate pass rate: ~70% -> ~95% (explicit format requirements)
/// - Deliberation completeness: ~60% -> ~90% (every-critique-point instruction)
/// - Handoff actionability: improved checklist specificity
///
/// # Second-pass tuning (2026-04-03, late)
///
/// Analysis of council-metrics.jsonl (2 real runs with metrics) revealed pass-1
/// fixes were insufficient:
///
/// 1. **Near-zero acceptance rate**: Run council-1775267346 had 0 critiques accepted
///    across 2 rounds. Run council-1775267158 improved to 4 accepted in round 2,
///    but still hit max rounds. Root cause: deliberation lead (Codex) defaults to
///    REJECT without engaging with critique substance.
///    Fix: Added acceptance-bias guidance to DELIBERATION_LEAD_PROMPT — "default
///    to ACCEPT when a critique identifies a real gap; REJECT only when it
///    misunderstands the plan or the fix would make things worse."
///
/// 2. **High-severity inflation persists**: Despite pass-1 calibration, reviewers
///    still submitted 1-3 high-severity items per round. Each high blocks convergence.
///    Fix: Added hard budget to CRITIQUE_PROMPT — "at most 1 high-severity item
///    per review; if you have more than one candidate, pick the most critical."
///
/// 3. **Synthesis verbosity**: Codex produced 12-15KB revised plans per round,
///    making it hard for reviewers to track changes.
///    Fix: Added conciseness instruction to DELIBERATION_LEAD_PROMPT — keep the
///    revised plan to essential changes only, mark diffs with [CHANGED].
///
/// Expected improvements (pass 2):
/// - Acceptance rate: ~0% -> ~50%+ (acceptance-bias fix)
/// - Convergence rate: ~0% -> ~70% within 2 rounds (severity budget + acceptance bias)
/// - Synthesis size: 12-15KB -> 8-10KB (conciseness instruction)
///
/// # Third-pass tuning (2026-04-04)
///
/// Telemetry review of pass-2 runs showed acceptance-bias tuning only partially
/// effective — zero-acceptance persists in some runs where ALL candidates get
/// rejected in deliberation. Pattern analysis:
///
/// 1. **Zero-acceptance persists**: The deliberation lead treats all critiques
///    as challenges to defend against, regardless of severity. Low-severity items
///    (which are minor polish by definition) were being rejected at the same rate
///    as high-severity items. Root cause: no severity-to-disposition mapping.
///    Fix: Added explicit severity-proportional disposition rules — low → almost
///    always ACCEPT, medium → lean ACCEPT/PARTIAL, high → evaluate carefully.
///    Added zero-acceptance safeguard: if about to reject ALL, must re-read and
///    find at least one genuine improvement.
///
/// 2. **Medium-severity inflation**: Reviewers submitted 3-5 medium items per
///    review, creating a wall of "significant" critiques that overwhelms the lead.
///    Fix: Added medium-severity budget (at most 2 per review) to CRITIQUE_PROMPT,
///    with explicit guidance that LOW is the default severity.
///
/// 3. **Severity-disposition disconnect**: Pass-2 told the lead to "default to
///    ACCEPT" but didn't tie this to critique severity. The lead applied uniform
///    skepticism across all severity levels.
///    Fix: Replaced generic acceptance-bias with severity-proportional rules.
///    Lowered the self-check threshold from >50% rejections to >33%.
///
/// Expected improvements (pass 3):
/// - Zero-acceptance runs: ~30% of runs -> <5% (severity-proportional disposition)
/// - Acceptance rate: ~50% -> ~70%+ (low-severity auto-accept + medium budget)
/// - Convergence rate: ~70% -> ~85% within 2 rounds (fewer inflated mediums)

pub const FRAMING_PROMPT: &str = r#"You are the framing controller for an AI planning council.

Given the task below, produce four sections as a single structured output.
You MUST use the exact markdown headers shown below. Every section MUST contain
at least 2-3 concrete, specific points (not vague generalities).

## Problem Brief
What specific problem are we solving? What does "done" look like in observable terms?

## Constraints
Hard constraints only — tech stack, time budget, scope limits, external dependencies.
List each as a bullet point.

## Success Criteria
Measurable or testable criteria. Each criterion should be verifiable by a developer.

## Out of Scope
What we are explicitly NOT solving in this iteration. Be specific about boundaries.

Do NOT include introductory text, disclaimers, or meta-commentary.
Start directly with `## Problem Brief`.

Task:
{task}

External Context (Retrieved by Layers):
{external_context}"#;

pub const BRAINSTORM_SEED_PROMPT: &str = r#"You are the brainstorming lead (solution-scout) in an AI planning council.

Your role:
- Widen the solution space with concrete alternatives (not hand-wavy ideas)
- Draft a structured plan that another developer could act on immediately
- Stay grounded in implementation reality — name real files, real APIs, real tools

Problem framing:
{framing}

Task:
{task}

Produce a structured plan using these EXACT markdown headers. Every section MUST have
substantive content (at least 2-3 bullet points or sentences). Empty sections are not acceptable.

## 1. Goal and constraints
## 2. Candidate approaches
List at least 2 concrete approaches with tradeoffs for each.
## 3. Recommended approach with rationale
## 4. V1 scope
What ships first — list specific deliverables.
## 5. V2 / later
## 6. Out of scope / do not build
## 7. Files, binaries, and storage
Name specific file paths and explain each.
## 8. Control flow
Describe the runtime execution path step by step.
## 9. Risks and open questions
## 10. Validation plan
Specific test cases or verification steps.

Be concrete. Optimize for a local dev tool, not a production platform.
Do NOT include preamble or meta-commentary — start directly with ## 1."#;

pub const BRAINSTORM_CONTRIBUTE_PROMPT: &str = r#"You are a brainstorming contributor ({role}) in an AI planning council.

Your role: {role_description}

Rules:
- You may NOT rewrite the plan wholesale
- Submit 3-7 specific improvement suggestions (not vague observations)
- Reference plan sections by their exact header name
- Each suggestion must propose a concrete change, not just identify a gap

Problem framing:
{framing}

Current plan:
{plan}

Task:
{task}

Return your contributions as a numbered list. For each suggestion use this format:

1. **Section: [exact header name]**
   - **Suggest:** [specific change to make]
   - **Why:** [concrete reason — what breaks or improves]

Do NOT include general praise, meta-commentary, or restatements of the plan."#;

pub const CRITIQUE_PROMPT: &str = r#"You are an adversarial reviewer ({role}) in an AI planning council.

Your role: {role_description}

Rules:
- Reference the current plan and propose specific deltas
- Do NOT rewrite the entire plan
- Be structured: use the JSON schema below
- Output ONLY the JSON block — no prose before or after

Severity calibration (IMPORTANT — follow strictly):
- **high**: Would cause the plan to FAIL if shipped as-is. Missing a critical
  requirement, architectural flaw that blocks the goal, security vulnerability.
  BUDGET: You may flag AT MOST 1 item as high severity per review. If you have
  multiple candidates, pick the single most critical one and downgrade the rest.
- **medium**: Significant gap that degrades quality but doesn't block shipping.
  Missing error handling, unclear ownership, incomplete edge case coverage.
  BUDGET: At most 2 medium items per review. If you have more, keep the top 2
  and downgrade the rest to low.
- **low**: Polish items, minor improvements, style preferences, nice-to-haves.
  This is your DEFAULT severity. Most critique points belong here.
  If in doubt between medium and low, ALWAYS choose LOW.
  If in doubt between high and medium, ALWAYS choose MEDIUM.

Quality-proportional review:
- A plan that achieves its stated goals with only minor gaps should receive ONLY
  low-severity items. Returning all-low or an empty critiques array is a GOOD
  outcome — it means the plan is solid, not that you failed to review.
- Do NOT inflate severity to appear thorough. Your job is accuracy, not volume.
- If you cannot find a genuine high or medium issue, submit only low items.
  An honest all-low review is far more valuable than an inflated one.

Current plan:
{plan}

Task:
{task}

Return STRICT JSON matching this schema (no markdown fences, no commentary):
{{
  "reviewer_role": "{role}",
  "critiques": [
    {{
      "issue": "string — one-sentence problem statement",
      "severity": "high|medium|low",
      "why_it_matters": "string — concrete impact if not fixed",
      "suggested_delta": "string — specific change to make",
      "evidence": "string — quote or reference from the plan"
    }}
  ],
  "things_to_keep": ["string — specific strengths worth preserving"]
}}"#;

pub const DELIBERATION_LEAD_PROMPT: &str = r#"You are the deliberation lead (synthesis-owner) in an AI planning council.

Your job:
- Review EVERY critique point from adversarial reviewers — do not skip any
- Classify each as ACCEPT / PARTIAL / REJECT with a one-sentence rationale
- Produce a revised plan incorporating accepted/partial changes
- Keep decisions concise to avoid truncation

Severity-proportional disposition (IMPORTANT — follow strictly):
- **Low-severity critiques → almost always ACCEPT.** These are minor improvements
  by definition. Accept them unless the suggested change is actively harmful.
  Rejecting a low-severity critique requires strong justification.
- **Medium-severity critiques → lean ACCEPT or PARTIAL.** Default to ACCEPT when
  the critique identifies a real gap. Use PARTIAL when directionally correct but
  the suggested fix is too broad — accept the spirit, scope down the change.
- **High-severity critiques → evaluate carefully.** These claim the plan would fail.
  If the claim is valid, ACCEPT. If overstated, PARTIAL with a scoped fix.
  REJECT only if the issue genuinely doesn't exist.
- REJECT is reserved for critiques that misunderstand the plan, identify a
  non-existent problem, or propose a fix that makes the design worse.
  Every REJECT must include a specific explanation — not just a dismissal.

Zero-acceptance safeguard:
- If you are about to REJECT every critique, STOP. Re-read each one and find at
  least one genuine improvement to ACCEPT. A 100% rejection rate almost always
  means you are being too defensive, not that every critique is wrong.
- If you find yourself rejecting more than a third of critiques, pause and
  reconsider whether you are defending the plan rather than improving it.

Current plan:
{plan}

Critiques:
{critiques}

Task:
{task}

Return EXACTLY two sections in this format:

## Decision Log
Group decisions by source reviewer using ### subheadings.
Use this EXACT format (one line per decision):

### Claude (conceptual-risk)
- ACCEPT: <issue summary> -- <rationale>
- PARTIAL: <issue summary> -- <rationale>
- REJECT: <issue summary> -- <rationale>

### Gemini (implementation-risk)
- ACCEPT: <issue summary> -- <rationale>
- PARTIAL: <issue summary> -- <rationale>
- REJECT: <issue summary> -- <rationale>

You MUST address every critique point listed above. Do not skip any.
Keep rationales to one sentence each.

## Revised Plan
The updated plan with accepted changes incorporated. Mark changed sections
with [CHANGED] tags. Preserve all original sections that were not changed.

Keep the revised plan CONCISE — do not expand unchanged sections. Only rewrite
sections that were actually modified by accepted/partial critiques. Target the
same length as the input plan or shorter."#;

pub const HANDOFF_PROMPT: &str = r#"You are preparing the build handoff for an AI planning council.

Convert the final plan into executable implementation work that a developer can
start coding from immediately.

Final plan:
{plan}

Task:
{task}

Produce EXACTLY three sections with these headers:

## Final Plan
Clean version of the plan without [CHANGED] tags. Preserve all technical detail.

## Implementation Checklist
Ordered, numbered list of concrete build steps. Requirements for each item:
- Starts with a verb (Create, Implement, Add, Wire, Test, etc.)
- Names specific files, functions, or modules
- Completable by one developer in one focused session (< 2 hours)
- List at least 5 items, no more than 15

Example format:
1. Create `src/store.rs` with `Store` struct implementing get/set/delete
2. Add input validation: key max 256 bytes, value max 64KB
3. Write unit tests for all Store methods in `src/store.rs`

## Build Summary
3-5 sentences: what we are building, why, and the key technical decisions made.
Do NOT restate the task verbatim — summarize the plan's conclusions."#;

/// Format a prompt template by replacing {key} placeholders.
pub fn fmt_prompt(template: &str, vars: &[(&str, &str)]) -> String {
    let mut result = template.to_string();
    for (key, value) in vars {
        result = result.replace(&format!("{{{key}}}"), value);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_prompt_replaces_all_placeholders() {
        let template = "Hello {name}, your role is {role}.";
        let result = fmt_prompt(template, &[("name", "Alice"), ("role", "scout")]);
        assert_eq!(result, "Hello Alice, your role is scout.");
        assert!(!result.contains('{'));
    }

    #[test]
    fn fmt_prompt_leaves_escaped_braces_in_critique() {
        // The critique prompt uses {{ and }} for JSON schema examples.
        // fmt_prompt should not break these since it only replaces {key} patterns.
        let result = fmt_prompt(
            CRITIQUE_PROMPT,
            &[
                ("role", "test-reviewer"),
                ("role_description", "Find bugs"),
                ("plan", "The plan text"),
                ("task", "Build a thing"),
            ],
        );
        assert!(result.contains("test-reviewer"));
        assert!(result.contains("Find bugs"));
        assert!(result.contains("The plan text"));
        assert!(result.contains("Build a thing"));
        // JSON schema braces should remain
        assert!(result.contains("\"reviewer_role\""));
        assert!(result.contains("\"critiques\""));
    }

    // -- Framing prompt tests --

    #[test]
    fn framing_prompt_contains_required_section_headers() {
        let result = fmt_prompt(
            FRAMING_PROMPT,
            &[("task", "Build a CLI tool"), ("external_context", "none")],
        );
        assert!(result.contains("## Problem Brief"), "Missing Problem Brief header");
        assert!(result.contains("## Constraints"), "Missing Constraints header");
        assert!(result.contains("## Success Criteria"), "Missing Success Criteria header");
        assert!(result.contains("## Out of Scope"), "Missing Out of Scope header");
    }

    #[test]
    fn framing_prompt_substitutes_task_and_context() {
        let result = fmt_prompt(
            FRAMING_PROMPT,
            &[("task", "Design a cache layer"), ("external_context", "Redis is available")],
        );
        assert!(result.contains("Design a cache layer"));
        assert!(result.contains("Redis is available"));
        assert!(!result.contains("{task}"));
        assert!(!result.contains("{external_context}"));
    }

    // -- Brainstorm seed prompt tests --

    #[test]
    fn brainstorm_seed_prompt_contains_all_ten_sections() {
        let result = fmt_prompt(
            BRAINSTORM_SEED_PROMPT,
            &[("framing", "Problem statement"), ("task", "Build X")],
        );
        for i in 1..=10 {
            assert!(
                result.contains(&format!("## {i}.")),
                "Missing section header ## {i}."
            );
        }
    }

    #[test]
    fn brainstorm_seed_prompt_substitutes_framing_and_task() {
        let result = fmt_prompt(
            BRAINSTORM_SEED_PROMPT,
            &[("framing", "THE_FRAMING"), ("task", "THE_TASK")],
        );
        assert!(result.contains("THE_FRAMING"));
        assert!(result.contains("THE_TASK"));
        assert!(!result.contains("{framing}"));
        assert!(!result.contains("{task}"));
    }

    // -- Brainstorm contribute prompt tests --

    #[test]
    fn brainstorm_contribute_prompt_substitutes_all_vars() {
        let result = fmt_prompt(
            BRAINSTORM_CONTRIBUTE_PROMPT,
            &[
                ("role", "elegance-scout"),
                ("role_description", "Find simplifications"),
                ("framing", "F"),
                ("plan", "P"),
                ("task", "T"),
            ],
        );
        assert!(result.contains("elegance-scout"));
        assert!(result.contains("Find simplifications"));
        assert!(!result.contains("{role}"));
        assert!(!result.contains("{role_description}"));
        assert!(!result.contains("{framing}"));
        assert!(!result.contains("{plan}"));
        assert!(!result.contains("{task}"));
    }

    // -- Critique prompt tests --

    #[test]
    fn critique_prompt_contains_severity_calibration() {
        assert!(CRITIQUE_PROMPT.contains("Severity calibration"));
        assert!(CRITIQUE_PROMPT.contains("**high**"));
        assert!(CRITIQUE_PROMPT.contains("**medium**"));
        assert!(CRITIQUE_PROMPT.contains("**low**"));
    }

    #[test]
    fn critique_prompt_json_schema_matches_struct_fields() {
        // Verify the JSON schema in the prompt matches the CritiquePoint fields
        // from critique.rs: issue, severity, why_it_matters, suggested_delta, evidence
        assert!(CRITIQUE_PROMPT.contains("\"issue\""));
        assert!(CRITIQUE_PROMPT.contains("\"severity\""));
        assert!(CRITIQUE_PROMPT.contains("\"why_it_matters\""));
        assert!(CRITIQUE_PROMPT.contains("\"suggested_delta\""));
        assert!(CRITIQUE_PROMPT.contains("\"evidence\""));
        assert!(CRITIQUE_PROMPT.contains("\"reviewer_role\""));
        assert!(CRITIQUE_PROMPT.contains("\"critiques\""));
        assert!(CRITIQUE_PROMPT.contains("\"things_to_keep\""));
    }

    #[test]
    fn critique_prompt_contains_high_severity_budget() {
        assert!(CRITIQUE_PROMPT.contains("AT MOST 1 item as high severity"));
        assert!(CRITIQUE_PROMPT.contains("At most 2 medium items"));
        assert!(CRITIQUE_PROMPT.contains("ALWAYS choose LOW"));
        assert!(CRITIQUE_PROMPT.contains("ALWAYS choose MEDIUM"));
    }

    // -- Deliberation prompt tests --

    #[test]
    fn deliberation_prompt_contains_acceptance_bias() {
        assert!(DELIBERATION_LEAD_PROMPT.contains("Severity-proportional disposition"));
        assert!(DELIBERATION_LEAD_PROMPT.contains("almost always ACCEPT"));
        assert!(DELIBERATION_LEAD_PROMPT.contains("Zero-acceptance safeguard"));
        assert!(DELIBERATION_LEAD_PROMPT.contains("REJECT is reserved for"));
    }

    #[test]
    fn deliberation_prompt_contains_conciseness_guidance() {
        assert!(DELIBERATION_LEAD_PROMPT.contains("CONCISE"));
    }

    #[test]
    fn deliberation_prompt_contains_decision_format() {
        assert!(DELIBERATION_LEAD_PROMPT.contains("ACCEPT:"));
        assert!(DELIBERATION_LEAD_PROMPT.contains("PARTIAL:"));
        assert!(DELIBERATION_LEAD_PROMPT.contains("REJECT:"));
        assert!(DELIBERATION_LEAD_PROMPT.contains("## Decision Log"));
        assert!(DELIBERATION_LEAD_PROMPT.contains("## Revised Plan"));
        // Source model grouping
        assert!(DELIBERATION_LEAD_PROMPT.contains("### Claude (conceptual-risk)"));
        assert!(DELIBERATION_LEAD_PROMPT.contains("### Gemini (implementation-risk)"));
    }

    #[test]
    fn deliberation_prompt_substitutes_all_vars() {
        let result = fmt_prompt(
            DELIBERATION_LEAD_PROMPT,
            &[("plan", "PLAN"), ("critiques", "CRITS"), ("task", "TASK")],
        );
        assert!(result.contains("PLAN"));
        assert!(result.contains("CRITS"));
        assert!(result.contains("TASK"));
        assert!(!result.contains("{plan}"));
        assert!(!result.contains("{critiques}"));
        assert!(!result.contains("{task}"));
    }

    // -- Handoff prompt tests --

    #[test]
    fn handoff_prompt_contains_required_sections() {
        assert!(HANDOFF_PROMPT.contains("## Final Plan"));
        assert!(HANDOFF_PROMPT.contains("## Implementation Checklist"));
        assert!(HANDOFF_PROMPT.contains("## Build Summary"));
    }

    #[test]
    fn handoff_prompt_contains_checklist_guidance() {
        assert!(HANDOFF_PROMPT.contains("Starts with a verb"));
        assert!(HANDOFF_PROMPT.contains("at least 5 items"));
    }

    #[test]
    fn handoff_prompt_substitutes_all_vars() {
        let result = fmt_prompt(
            HANDOFF_PROMPT,
            &[("plan", "FINAL_PLAN"), ("task", "BUILD_TASK")],
        );
        assert!(result.contains("FINAL_PLAN"));
        assert!(result.contains("BUILD_TASK"));
        assert!(!result.contains("{plan}"));
        assert!(!result.contains("{task}"));
    }
}
