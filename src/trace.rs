use crate::model::Model;
use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize)]
pub struct RunTrace {
    pub run_id: String,
    pub timestamp: String,
    pub task: String,
    pub task_type: String,
    pub phases_completed: Vec<String>,
    pub models_used: Vec<String>,
    pub persona_assignments: Vec<PersonaAssignment>,
    pub rounds: u32,
    pub converged: bool,
    pub artifacts_dir: String,
    pub final_plan_path: String,
}

#[derive(Debug, Serialize)]
pub struct PersonaAssignment {
    pub model: String,
    pub role: String,
    pub phase: String,
}

#[derive(Debug, Serialize)]
pub struct PlanRecord {
    pub run_id: String,
    pub timestamp: String,
    pub task: String,
    pub task_type: String,
    pub plan_markdown: String,
    pub artifacts_dir: String,
}

#[derive(Debug, Serialize)]
pub struct LearningRecord {
    pub run_id: String,
    pub timestamp: String,
    pub kind: String,
    pub summary: String,
    pub task_type: String,
    pub model: String,
    pub role: String,
}

pub fn generate_run_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("council-{}", now.as_secs())
}

pub fn iso_now() -> String {
    // Use the `date` command for ISO format since we're avoiding chrono
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

pub fn append_jsonl(path: &Path, value: &impl Serialize) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string(value) {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{json}");
        }
    }
}

pub fn persona_assignment(model: Model, role: &str, phase: &str) -> PersonaAssignment {
    PersonaAssignment {
        model: model.name().into(),
        role: role.into(),
        phase: phase.into(),
    }
}
