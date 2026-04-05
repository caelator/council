use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Build the council binary once (reused across tests).
fn build_council() -> PathBuf {
    let cargo = env::var("CARGO").unwrap_or_else(|_| {
        let home = env::var("HOME").unwrap_or_else(|_| "/Users/bri".into());
        format!("{}/.cargo/bin/cargo", home)
    });

    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let status = Command::new(&cargo)
        .args(["build", "--manifest-path"])
        .arg(&manifest)
        .status()
        .expect("failed to build council");
    assert!(status.success(), "cargo build failed");

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("debug")
        .join("council")
}

/// Create mock CLI scripts in the given directory.
/// Each mock reads its arguments and returns canned responses based on stage keywords.
fn create_mock_scripts(mock_dir: &Path) {
    // --- mock gemini ---
    // NOTE: adversarial/critique must be checked BEFORE brainstorm/seed, because
    // the critique prompt embeds the full plan which may contain brainstorm keywords.
    let gemini_script = r#"#!/bin/bash
# Mock gemini CLI for integration tests
# All args are passed via -p <prompt>
PROMPT="$*"

if echo "$PROMPT" | grep -qi "adversarial.*reviewer\|STRICT JSON.*critiques"; then
    cat <<'CRITIQUE'
```json
{
  "reviewer_role": "adversarial-reviewer/implementation-risk",
  "critiques": [
    {
      "issue": "No input validation on key size",
      "severity": "low",
      "why_it_matters": "Could allow oversized keys",
      "suggested_delta": "Add key size limit",
      "evidence": "Section 4"
    }
  ],
  "things_to_keep": ["Simple mutex approach is appropriate for v1"]
}
```
CRITIQUE
elif echo "$PROMPT" | grep -qi "framing controller"; then
    cat <<'FRAMING'
## Problem Brief
Design a simple in-memory key-value store for local development.

## Constraints
- Must be a single binary
- Must support GET, SET, DELETE operations
- No persistence required for v1

## Success Criteria
- All CRUD operations work correctly
- Sub-millisecond latency for in-memory ops

## Out of Scope
- Clustering, replication, persistence to disk
FRAMING
elif echo "$PROMPT" | grep -qi "synthesiz\|integrate.*contributions"; then
    cat <<'SYNTH'
# Revised Key-Value Store Plan [CHANGED]

## 1. Goal and constraints
Build an in-memory KV store. [CHANGED] Added input validation.

## 2. Recommended approach
HashMap behind a Mutex with input validation layer.

## 3. V1 scope
GET, SET, DELETE with size limits on keys/values.

## 4. Implementation
- src/main.rs — entry point
- src/store.rs — KV store logic
- src/protocol.rs — TCP command parsing
SYNTH
elif echo "$PROMPT" | grep -qi "brainstorming lead\|solution-scout"; then
    cat <<'SEED'
# Key-Value Store Plan

## 1. Goal and constraints
Build an in-memory KV store as a single Rust binary.

## 2. Candidate approaches
- A) HashMap behind a Mutex
- B) Lock-free concurrent map (dashmap)

## 3. Recommended approach
Approach A for simplicity.

## 4. V1 scope
GET, SET, DELETE over a simple TCP protocol.

## 5. V2 / later
Persistence, TTL support.

## 6. Out of scope
Clustering.

## 7. Files
- src/main.rs, src/store.rs

## 8. Control flow
Client connects -> parse command -> execute on store -> respond

## 9. Risks
Mutex contention under high concurrency.

## 10. Validation plan
Unit tests for store operations, integration test for TCP protocol.
SEED
else
    echo "Mock gemini: unrecognized prompt pattern"
    echo "$PROMPT" | head -3
fi
"#;

    // --- mock claude ---
    // NOTE: adversarial/critique must be checked BEFORE contribute, because
    // the critique prompt embeds the full plan which may contain contribution keywords.
    let claude_script = r#"#!/bin/bash
# Mock claude CLI for integration tests
PROMPT="$*"

if echo "$PROMPT" | grep -qi "adversarial.*reviewer\|STRICT JSON.*critiques"; then
    cat <<'CRITIQUE'
```json
{
  "reviewer_role": "adversarial-reviewer/conceptual-risk",
  "critiques": [
    {
      "issue": "No consideration for concurrent access patterns",
      "severity": "low",
      "why_it_matters": "Single mutex may bottleneck",
      "suggested_delta": "Document expected concurrency model",
      "evidence": "Section 9 mentions risk but no mitigation"
    }
  ],
  "things_to_keep": ["Scope is well-defined"]
}
```
CRITIQUE
elif echo "$PROMPT" | grep -qi "elegance\|contribut"; then
    cat <<'CONTRIB'
## Contributions (elegance-scout)

1. **Section: Recommended approach**
   - Suggest: Consider using a trait-based store interface for testability
   - Why: Makes it easy to swap implementations later

2. **Section: Control flow**
   - Suggest: Use an enum for commands instead of string parsing
   - Why: Compile-time safety and exhaustive matching
CONTRIB
else
    echo "Mock claude: generic response"
fi
"#;

    // --- mock codex ---
    // NOTE: handoff/deliberation must be checked BEFORE contribute, because
    // the deliberation prompt embeds the full plan which may contain contribution keywords.
    let codex_script = r#"#!/bin/bash
# Mock codex CLI for integration tests
PROMPT="$*"

if echo "$PROMPT" | grep -qi "build handoff\|implementation checklist\|build summary"; then
    cat <<'HANDOFF'
# Final Plan: In-Memory Key-Value Store

## Overview
A simple in-memory key-value store implemented as a single Rust binary with a line-based TCP protocol.

## Architecture
- HashMap<String, String> behind a Mutex for thread safety
- Line-based TCP protocol: `GET key`, `SET key value`, `DEL key`
- Input validation: key max 256 bytes, value max 64KB

## Implementation Checklist
- [ ] Create project skeleton with Cargo
- [ ] Implement Store struct with get/set/delete methods
- [ ] Implement TCP listener and command parser
- [ ] Add input validation (key/value size limits)
- [ ] Write unit tests for store operations
- [ ] Write integration test for TCP protocol
- [ ] Add graceful shutdown handling

## Build Summary
We are building a lightweight in-memory key-value store for local development use.
It provides GET, SET, and DELETE operations over a line-based TCP protocol.
The implementation prioritizes simplicity using a Mutex-protected HashMap.
Input validation prevents oversized keys and values.
This is a v1 focused on correctness; persistence and clustering are deferred.
HANDOFF
elif echo "$PROMPT" | grep -qi "deliberation lead\|synthesis-owner"; then
    cat <<'REVISION'
## Decision Log
- ACCEPT: No input validation on key size -- Added key size limit of 256 bytes
- ACCEPT: No consideration for concurrent access patterns -- Documented single-threaded model for v1

## Revised Plan
# Key-Value Store — Revised

## Goal
In-memory KV store, single binary, simple protocol.

## Implementation
- HashMap<String, String> behind Mutex
- Key size limit: 256 bytes
- Value size limit: 64KB
- Line-based TCP protocol

## Files
- src/main.rs
- src/store.rs
- src/protocol.rs
REVISION
elif echo "$PROMPT" | grep -qi "feasibility-scout\|brainstorming contributor"; then
    cat <<'CONTRIB'
## Contributions (feasibility-scout)

1. **Section: V1 scope**
   - Suggest: Start with a simple line-based protocol instead of full TCP framing
   - Why: Reduces implementation complexity significantly

2. **Section: Risks**
   - Suggest: Add explicit memory limit to prevent OOM
   - Why: Unbounded HashMap can exhaust memory
CONTRIB
else
    echo "Mock codex: generic response"
fi
"#;

    // --- mock which (so check_available returns true) ---
    // Actually, check_available uses `which <model>`, so the mock scripts on PATH suffice.

    for (name, content) in [
        ("gemini", gemini_script),
        ("claude", claude_script),
        ("codex", codex_script),
    ] {
        let path = mock_dir.join(name);
        fs::write(&path, content).expect("failed to write mock script");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))
            .expect("failed to set permissions");
    }

    // Also need a mock `which` that finds our scripts — actually, the real `which` will
    // find them if mock_dir is first on PATH. But we also need a `date` command available.
    // The real `date` is fine.
}

#[test]
fn e2e_full_pipeline_with_mock_models() {
    let council_bin = build_council();
    assert!(
        council_bin.exists(),
        "council binary not found at {:?}",
        council_bin
    );

    // Create temp directories
    let tmp = env::temp_dir().join(format!("council-e2e-{}", std::process::id()));
    let mock_dir = tmp.join("mock-bin");
    let workdir = tmp.join("workdir");
    let council_dir = tmp.join("council-output");

    fs::create_dir_all(&mock_dir).unwrap();
    fs::create_dir_all(&workdir).unwrap();
    fs::create_dir_all(&council_dir).unwrap();
    // memoryport is created by the binary, but inside workdir
    let memoryport = workdir.join("memoryport");

    create_mock_scripts(&mock_dir);

    // Prepend mock_dir to PATH so mock scripts are found
    let original_path = env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", mock_dir.display(), original_path);

    let output = Command::new(&council_bin)
        .args([workdir.to_str().unwrap(), "Design a key-value store"])
        .env("PATH", &new_path)
        .env("COUNCIL_DIR", council_dir.to_str().unwrap())
        .env("COUNCIL_ROUNDS", "2")
        .env("COUNCIL_NAME", "e2e-test")
        .output()
        .expect("failed to run council");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    eprintln!("--- council stdout ---\n{}", stdout);
    eprintln!("--- council stderr ---\n{}", stderr);

    // Assert: all 4 stages completed (check stderr for stage markers)
    assert!(
        stderr.contains("Stage 0: Framing"),
        "Stage 0 not found in stderr"
    );
    assert!(
        stderr.contains("Stage 1: Brainstorming"),
        "Stage 1 not found in stderr"
    );
    assert!(
        stderr.contains("Stage 2: Adversarial Deliberation"),
        "Stage 2 not found in stderr"
    );
    assert!(
        stderr.contains("Stage 3: Build Handoff"),
        "Stage 3 not found in stderr"
    );
    assert!(
        stderr.contains("Council Complete"),
        "Council Complete not found in stderr"
    );

    // Assert: final-plan.md exists and is non-empty
    let final_plan_path = council_dir.join("final-plan.md");
    assert!(final_plan_path.exists(), "final-plan.md not found");
    let final_plan = fs::read_to_string(&final_plan_path).unwrap();
    assert!(!final_plan.trim().is_empty(), "final-plan.md is empty");
    assert!(
        final_plan.contains("Key-Value"),
        "final-plan.md doesn't contain expected content"
    );

    // Assert: run-summary.json exists and parses with schema_version
    let summary_path = council_dir.join("run-summary.json");
    assert!(summary_path.exists(), "run-summary.json not found");
    let summary_text = fs::read_to_string(&summary_path).unwrap();
    let summary: serde_json::Value =
        serde_json::from_str(&summary_text).expect("run-summary.json is not valid JSON");
    assert!(
        summary.get("run_id").is_some(),
        "run-summary.json missing run_id"
    );
    assert!(
        summary.get("converged").is_some(),
        "run-summary.json missing converged"
    );
    assert!(
        summary.get("schema_version").is_some(),
        "run-summary.json missing schema_version"
    );
    assert_eq!(
        summary["schema_version"].as_u64().unwrap(),
        1,
        "run-summary.json schema_version should be 1"
    );

    // Assert: convergence was detected (all-low critiques)
    let converged = summary["converged"].as_bool().unwrap_or(false);
    assert!(
        converged,
        "pipeline did not converge (expected all-low critiques)"
    );

    // Assert: telemetry JSONL files were written
    let traces_path = memoryport.join("council-traces.jsonl");
    let plans_path = memoryport.join("council-plans.jsonl");
    let learnings_path = memoryport.join("council-learnings.jsonl");

    assert!(traces_path.exists(), "council-traces.jsonl not found");
    let traces = fs::read_to_string(&traces_path).unwrap();
    assert!(
        traces.lines().count() >= 1,
        "council-traces.jsonl has no entries"
    );

    assert!(plans_path.exists(), "council-plans.jsonl not found");
    let plans = fs::read_to_string(&plans_path).unwrap();
    assert!(
        plans.lines().count() >= 1,
        "council-plans.jsonl has no entries"
    );

    assert!(learnings_path.exists(), "council-learnings.jsonl not found");
    let learnings = fs::read_to_string(&learnings_path).unwrap();
    assert!(
        learnings.lines().count() >= 1,
        "council-learnings.jsonl has no entries"
    );

    // Verify telemetry is valid JSON lines
    for line in traces.lines() {
        let _: serde_json::Value =
            serde_json::from_str(line).expect("traces JSONL line is not valid JSON");
    }
    for line in plans.lines() {
        let _: serde_json::Value =
            serde_json::from_str(line).expect("plans JSONL line is not valid JSON");
    }

    // Assert: decision-log.json in round dirs has schema_version
    for round in 1..=2 {
        let round_dir = council_dir.join(format!("stage2-round{round}"));
        let dlog_path = round_dir.join("decision-log.json");
        if dlog_path.exists() {
            let dlog_text = fs::read_to_string(&dlog_path).unwrap();
            let dlog: serde_json::Value =
                serde_json::from_str(&dlog_text).expect("decision-log.json is not valid JSON");
            assert!(
                dlog.get("schema_version").is_some(),
                "decision-log.json round {round} missing schema_version"
            );
        }
    }

    // Clean up
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn e2e_resume_reads_v1_payload_without_schema_version() {
    // Verify that resume mode gracefully handles a run-summary.json written
    // without a schema_version field (v1 backward compatibility).
    let council_bin = build_council();

    let tmp = env::temp_dir().join(format!("council-e2e-resume-{}", std::process::id()));
    let mock_dir = tmp.join("mock-bin");
    let workdir = tmp.join("workdir");
    let council_dir = tmp.join("council-output");

    fs::create_dir_all(&mock_dir).unwrap();
    fs::create_dir_all(&workdir).unwrap();
    fs::create_dir_all(&council_dir).unwrap();

    create_mock_scripts(&mock_dir);

    // Write a v1 run-summary.json WITHOUT schema_version (simulates old format)
    let v1_summary = r#"{
        "run_id": "council-legacy",
        "task": "Design a key-value store",
        "rounds": 1,
        "converged": false,
        "council_dir": "/tmp/old",
        "final_plan": "/tmp/old/final-plan.md"
    }"#;
    fs::write(council_dir.join("run-summary.json"), v1_summary).unwrap();

    let original_path = env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", mock_dir.display(), original_path);

    // Run in resume mode — should not crash, should log schema info
    let output = Command::new(&council_bin)
        .args([
            "--resume",
            workdir.to_str().unwrap(),
            "Design a key-value store",
        ])
        .env("PATH", &new_path)
        .env("COUNCIL_DIR", council_dir.to_str().unwrap())
        .env("COUNCIL_ROUNDS", "1")
        .env("COUNCIL_NAME", "resume-test")
        .output()
        .expect("failed to run council");

    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("--- resume stderr ---\n{}", stderr);

    // Should complete successfully
    assert!(
        stderr.contains("Council Complete"),
        "Council did not complete on resume"
    );

    // Should have logged that it loaded the v1 summary
    assert!(
        stderr.contains("[resume] Loaded run-summary.json"),
        "Did not log loading of v1 run-summary.json"
    );

    // The new run-summary.json should now have schema_version
    let new_summary_text = fs::read_to_string(council_dir.join("run-summary.json")).unwrap();
    let new_summary: serde_json::Value = serde_json::from_str(&new_summary_text).unwrap();
    assert_eq!(
        new_summary["schema_version"].as_u64().unwrap(),
        1,
        "New run-summary.json should have schema_version=1"
    );

    let _ = fs::remove_dir_all(&tmp);
}
