use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

use crate::provider::{self, OpenRouterFreeProvider};

/// A council model — either a local CLI tool or any free-tier OpenRouter model.
///
/// The `OpenRouterFree` variant accepts any model ID present in the free-tier
/// registry (`provider::FREE_MODEL_REGISTRY`).  This eliminates per-model enum
/// boilerplate: adding a new free model only requires a registry entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Model {
    Gemini,
    Claude,
    Codex,
    Gemma31B,
    Qwen36Plus,
    /// Any free-tier OpenRouter model, identified by its registry ID.
    /// The `&'static str` must match an entry in `FREE_MODEL_REGISTRY`.
    OpenRouterFree(&'static str),
}

// ── Custom serde ─────────────────────────────────────────────────────
// Serialize all variants as their canonical name string.  Deserialize
// by trying known local names first, then checking the free registry.

impl serde::Serialize for Model {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> serde::Deserialize<'de> for Model {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = <String as serde::Deserialize>::deserialize(deserializer)?;
        if let Some(m) = Model::from_name(&s) {
            Ok(m)
        } else {
            Err(serde::de::Error::custom(format!("unknown model: {s}")))
        }
    }
}

impl Model {
    /// Canonical short name used in logs, metrics, and serialization.
    pub fn name(self) -> &'static str {
        match self {
            Model::Gemini => "gemini",
            Model::Claude => "claude",
            Model::Codex => "codex",
            Model::Gemma31B => "gemma-llama",
            Model::Qwen36Plus => "qwen-36-plus",
            Model::OpenRouterFree(id) => {
                // Derive a short name from the OpenRouter ID.
                // "deepseek/deepseek-r1-0528:free" → "deepseek-r1-0528"
                let slug = id.split('/').last().unwrap_or(id);
                // Strip the ":free" suffix if present — it's noise in logs.
                let slug = slug.strip_suffix(":free").unwrap_or(slug);
                // Return &'static str by leaking — these are few, long-lived values.
                slug
            }
        }
    }

    /// If this model runs via the OpenRouter provider, return its model ID.
    pub fn openrouter_model_id(self) -> Option<&'static str> {
        match self {
            Model::Qwen36Plus => Some("qwen/qwen3.6-plus"),
            Model::OpenRouterFree(id) => Some(id),
            _ => None,
        }
    }

    /// Whether this model runs through a remote provider (vs. local CLI).
    pub fn is_remote(self) -> bool {
        self.openrouter_model_id().is_some()
    }

    /// Resolve a model from its canonical name or OpenRouter ID.
    ///
    /// Checks local model names first, then the free-tier registry.
    pub fn from_name(name: &str) -> Option<Model> {
        match name {
            "gemini" => Some(Model::Gemini),
            "claude" => Some(Model::Claude),
            "codex" => Some(Model::Codex),
            "gemma-llama" => Some(Model::Gemma31B),
            "qwen-36-plus" => Some(Model::Qwen36Plus),
            _ => {
                // Try as a direct OpenRouter free model ID.
                if let Some(entry) = provider::lookup_free_model(name) {
                    return Some(Model::OpenRouterFree(entry.id));
                }
                // Try matching against registry slugs (name without org/ and :free).
                for entry in provider::free_model_registry() {
                    let slug = entry.id.split('/').last().unwrap_or(entry.id);
                    let slug = slug.strip_suffix(":free").unwrap_or(slug);
                    if slug == name {
                        return Some(Model::OpenRouterFree(entry.id));
                    }
                }
                None
            }
        }
    }

    /// Return `Model` values for every free-tier registry entry.
    ///
    /// Qwen 3.6 Plus is returned as `Model::Qwen36Plus` (its dedicated variant)
    /// rather than as `OpenRouterFree` to preserve backward compatibility.
    pub fn all_free_models() -> Vec<Model> {
        provider::free_model_registry()
            .iter()
            .map(|entry| {
                if entry.id == "qwen/qwen3.6-plus" {
                    Model::Qwen36Plus
                } else {
                    Model::OpenRouterFree(entry.id)
                }
            })
            .collect()
    }

    /// All models that could participate in a council run.
    pub fn all_models() -> Vec<Model> {
        let mut models = vec![Model::Gemini, Model::Claude, Model::Codex, Model::Gemma31B];
        models.extend(Self::all_free_models());
        models
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
        Model::Qwen36Plus | Model::OpenRouterFree(_) => {
            unreachable!("remote models handled in run_model")
        }
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

    #[test]
    fn openrouter_free_variant_basics() {
        let m = Model::OpenRouterFree("deepseek/deepseek-r1-0528:free");
        assert!(m.is_remote());
        assert_eq!(
            m.openrouter_model_id(),
            Some("deepseek/deepseek-r1-0528:free")
        );
        assert_eq!(m.name(), "deepseek-r1-0528");
    }

    #[test]
    fn openrouter_free_name_strips_org_and_free() {
        let m = Model::OpenRouterFree("meta-llama/llama-4-maverick:free");
        assert_eq!(m.name(), "llama-4-maverick");
    }

    #[test]
    fn from_name_resolves_local_models() {
        assert_eq!(Model::from_name("gemini"), Some(Model::Gemini));
        assert_eq!(Model::from_name("claude"), Some(Model::Claude));
        assert_eq!(Model::from_name("codex"), Some(Model::Codex));
        assert_eq!(Model::from_name("gemma-llama"), Some(Model::Gemma31B));
        assert_eq!(Model::from_name("qwen-36-plus"), Some(Model::Qwen36Plus));
    }

    #[test]
    fn from_name_resolves_free_model_by_id() {
        let m = Model::from_name("deepseek/deepseek-r1-0528:free");
        assert_eq!(
            m,
            Some(Model::OpenRouterFree("deepseek/deepseek-r1-0528:free"))
        );
    }

    #[test]
    fn from_name_resolves_free_model_by_slug() {
        let m = Model::from_name("deepseek-r1-0528");
        assert_eq!(
            m,
            Some(Model::OpenRouterFree("deepseek/deepseek-r1-0528:free"))
        );
    }

    #[test]
    fn from_name_returns_none_for_unknown() {
        assert!(Model::from_name("openai/gpt-4o").is_none());
        assert!(Model::from_name("nonexistent").is_none());
    }

    #[test]
    fn all_free_models_includes_registry() {
        let free = Model::all_free_models();
        assert!(free.len() >= 3);
        // Qwen should be the dedicated variant, not OpenRouterFree.
        assert!(free.contains(&Model::Qwen36Plus));
        // Others should be OpenRouterFree.
        assert!(free
            .iter()
            .any(|m| matches!(m, Model::OpenRouterFree(_))));
    }

    #[test]
    fn all_models_includes_local_and_free() {
        let all = Model::all_models();
        assert!(all.contains(&Model::Gemini));
        assert!(all.contains(&Model::Claude));
        assert!(all.contains(&Model::Qwen36Plus));
        assert!(all.len() >= 7); // 4 local + at least 3 free
    }

    #[test]
    fn serde_roundtrip_local() {
        let m = Model::Gemini;
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, "\"gemini\"");
        let back: Model = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn serde_roundtrip_openrouter_free() {
        let m = Model::OpenRouterFree("deepseek/deepseek-r1-0528:free");
        let json = serde_json::to_string(&m).unwrap();
        // Serializes as the slug name.
        assert_eq!(json, "\"deepseek-r1-0528\"");
        // Deserializes back via slug lookup.
        let back: Model = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    // ── Proof B: Qwen lane runtime path goes through provider layer ──

    #[test]
    fn proof_b_qwen_dispatches_to_openrouter_provider() {
        // Qwen36Plus must resolve to a remote model with an OpenRouter model ID.
        let model = Model::Qwen36Plus;
        assert!(model.is_remote(), "Qwen must be remote");
        assert_eq!(
            model.openrouter_model_id(),
            Some("qwen/qwen3.6-plus"),
            "Qwen must resolve to openrouter model ID"
        );
        // This means run_model() will take the openrouter path, not the local CLI path.
    }

    #[test]
    fn proof_b_all_free_models_route_through_provider() {
        // Every model from all_free_models() must be remote → provider-dispatched.
        for model in Model::all_free_models() {
            assert!(
                model.is_remote(),
                "{} should be remote",
                model.name()
            );
            assert!(
                model.openrouter_model_id().is_some(),
                "{} should have an openrouter model ID",
                model.name()
            );
            // Verify the model ID is in the free registry (provider gate will pass).
            let model_id = model.openrouter_model_id().unwrap();
            assert!(
                provider::is_free_model(model_id),
                "{} (model_id={}) should be in the free registry",
                model.name(),
                model_id
            );
        }
    }

    #[test]
    fn proof_b_qwen_free_gate_allows_qwen_model_id() {
        // End-to-end: the model ID that Qwen resolves to passes the free gate.
        let model_id = Model::Qwen36Plus.openrouter_model_id().unwrap();
        assert!(
            provider::is_free_model(model_id),
            "Qwen's resolved model ID should pass free gate"
        );
    }

    #[test]
    fn proof_b_openrouter_free_variant_gate_pass() {
        // Every OpenRouterFree variant's ID must pass the free gate.
        for entry in provider::free_model_registry() {
            let model = if entry.id == "qwen/qwen3.6-plus" {
                Model::Qwen36Plus
            } else {
                Model::OpenRouterFree(entry.id)
            };
            let model_id = model.openrouter_model_id().unwrap();
            assert!(
                provider::is_free_model(model_id),
                "model {} (id={}) must pass free gate",
                model.name(),
                model_id
            );
        }
    }

    // ── Proof B: roster override resolution ─────────────────────────

    #[test]
    fn proof_b_roster_override_resolves_free_model_by_slug() {
        // Simulates what resolve_roster_model does: Model::from_name with a slug.
        let model = Model::from_name("deepseek-r1-0528");
        assert!(model.is_some(), "slug should resolve to a model");
        let model = model.unwrap();
        assert!(model.is_remote());
        assert_eq!(
            model.openrouter_model_id(),
            Some("deepseek/deepseek-r1-0528:free")
        );
    }

    #[test]
    fn proof_b_roster_override_resolves_free_model_by_full_id() {
        let model = Model::from_name("meta-llama/llama-4-maverick:free");
        assert!(model.is_some(), "full ID should resolve");
        let model = model.unwrap();
        assert!(model.is_remote());
        assert!(provider::is_free_model(model.openrouter_model_id().unwrap()));
    }

    #[test]
    fn proof_b_roster_override_rejects_paid_model_name() {
        // A paid model name must NOT resolve → roster stays on default.
        assert!(Model::from_name("openai/gpt-4o").is_none());
        assert!(Model::from_name("gpt-4o").is_none());
    }
}
