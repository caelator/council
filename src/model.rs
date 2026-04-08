use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

use crate::provider::{self, OpenRouterFreeProvider};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Model {
    Gemini,
    Claude,
    Codex,
    Gemma31B,
    Qwen36Plus,
}

impl Model {
    pub fn name(self) -> &'static str {
        match self {
            Model::Gemini => "gemini",
            Model::Claude => "claude",
            Model::Codex => "codex",
            Model::Gemma31B => "gemma-llama",
            Model::Qwen36Plus => "qwen-36-plus",
        }
    }

    /// If this model runs via the OpenRouter provider, return its model ID.
    pub fn openrouter_model_id(self) -> Option<&'static str> {
        match self {
            Model::Qwen36Plus => Some("qwen/qwen3.6-plus"),
            _ => None,
        }
    }

    /// Whether this model runs through a remote provider (vs. local CLI).
    pub fn is_remote(self) -> bool {
        self.openrouter_model_id().is_some()
    }
}

pub struct ModelOutput {
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
}

pub fn run_model(model: Model, prompt: &str, workdir: &Path) -> io::Result<ModelOutput> {
    match model.openrouter_model_id() {
        Some(model_id) => run_openrouter_model(model_id, prompt),
        None => run_local_model(model, prompt, workdir),
    }
}

fn run_local_model(model: Model, prompt: &str, workdir: &Path) -> io::Result<ModelOutput> {
    let output = match model {
        Model::Gemini => Command::new("gemini")
            .args(["--approval-mode", "plan", "-o", "text", "-p", prompt])
            .current_dir(workdir)
            .output(),
        Model::Claude => Command::new("claude")
            .args(["--dangerously-skip-permissions", "--print", prompt])
            .current_dir(workdir)
            .output(),
        Model::Codex => Command::new("codex")
            .args([
                "exec",
                "--full-auto",
                "-C",
                workdir.to_str().unwrap_or("."),
                "--skip-git-repo-check",
                "--add-dir",
                workdir.to_str().unwrap_or("."),
                "--",
                prompt,
            ])
            .output(),
        Model::Gemma31B => Command::new("gemma-llama")
            .args(["-o", "text", "-p", prompt])
            .current_dir(workdir)
            .output(),
        _ => unreachable!("remote models handled in run_model"),
    }?;

    Ok(ModelOutput {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        success: output.status.success(),
    })
}

// ── OpenRouter API integration (via provider) ─────────────────────────

/// Call the OpenRouter chat completions API for a specific model.
///
/// Delegates to `OpenRouterFreeProvider`, which enforces the free-only gate.
fn run_openrouter_model(model_id: &str, prompt: &str) -> io::Result<ModelOutput> {
    let provider = OpenRouterFreeProvider::from_env().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::Other,
            "No OpenRouter API key found (set OPENROUTER_API_KEY or configure ~/.openclaw/openclaw.json)",
        )
    })?;

    let content = provider::ChatProvider::complete(&provider, model_id, prompt, 8192)?;

    Ok(ModelOutput {
        stdout: content,
        stderr: String::new(),
        success: true,
    })
}

// ── Availability cache ─────────────────────────────────────────────

static AVAILABILITY_CACHE: std::sync::LazyLock<Mutex<HashMap<Model, bool>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn check_available(model: Model) -> bool {
    let mut cache = AVAILABILITY_CACHE.lock().unwrap();
    if let Some(&cached) = cache.get(&model) {
        return cached;
    }
    let available = if model.is_remote() {
        provider::resolve_openrouter_api_key().is_some()
    } else {
        Command::new("sh")
            .args(["-c", &format!("command -v {}", model.name())])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };
    cache.insert(model, available);
    available
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_qwen() {
        assert_eq!(Model::Qwen36Plus.name(), "qwen-36-plus");
    }

    #[test]
    fn qwen_is_remote() {
        assert!(Model::Qwen36Plus.is_remote());
        assert_eq!(
            Model::Qwen36Plus.openrouter_model_id(),
            Some("qwen/qwen3.6-plus")
        );
    }

    #[test]
    fn local_models_are_not_remote() {
        assert!(!Model::Gemini.is_remote());
        assert!(!Model::Claude.is_remote());
        assert!(!Model::Codex.is_remote());
        assert!(!Model::Gemma31B.is_remote());
    }

    #[test]
    fn openrouter_model_id_none_for_local() {
        assert!(Model::Gemini.openrouter_model_id().is_none());
    }
}
