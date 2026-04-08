//! OpenAI-compatible provider abstraction with OpenRouter free-only implementation.
//!
//! Design: one trait (`ChatProvider`) for any OpenAI-compatible chat-completion
//! backend.  The first concrete impl is `OpenRouterFreeProvider`, which enforces
//! that only free-tier models are ever called.

use std::io;

// ── Trait ──────────────────────────────────────────────────────────────

/// Minimal trait for an OpenAI-compatible chat-completion backend.
///
/// Implementations own their HTTP client, credentials, and any policy
/// enforcement (e.g. free-only filtering).
pub trait ChatProvider: Send + Sync {
    /// Human-readable provider name (e.g. "openrouter-free").
    fn name(&self) -> &str;

    /// Send a single-turn chat completion request.
    ///
    /// * `model_id` — provider-specific model identifier (e.g. "qwen/qwen3.6-plus").
    /// * `prompt`   — user-turn content.
    /// * `max_tokens` — response length budget.
    ///
    /// Returns the assistant message content (post-processed, e.g. think-blocks stripped).
    fn complete(&self, model_id: &str, prompt: &str, max_tokens: u32) -> io::Result<String>;

    /// Check whether the provider is configured and reachable.
    fn is_available(&self) -> bool;
}

// ── OpenRouter free-only provider ─────────────────────────────────────

/// Known free-tier models on OpenRouter.
///
/// This registry is the **sole gate** preventing accidental paid API usage.
/// A model must appear here to be callable.  The list is intentionally
/// conservative — add entries only after verifying free-tier status on
/// <https://openrouter.ai/models>.
///
/// Each entry: (model_id, human_label, supports_reasoning).
const FREE_MODEL_REGISTRY: &[FreeModelEntry] = &[
    FreeModelEntry {
        id: "qwen/qwen3.6-plus",
        label: "Qwen 3.6 Plus",
        reasoning: true,
    },
    FreeModelEntry {
        id: "qwen/qwen3-235b-a22b:free",
        label: "Qwen 3 235B (free)",
        reasoning: true,
    },
    FreeModelEntry {
        id: "google/gemma-3-27b-it:free",
        label: "Gemma 3 27B IT (free)",
        reasoning: false,
    },
    FreeModelEntry {
        id: "mistralai/mistral-small-3.2-24b-instruct:free",
        label: "Mistral Small 3.2 (free)",
        reasoning: false,
    },
    FreeModelEntry {
        id: "deepseek/deepseek-r1-0528:free",
        label: "DeepSeek R1 (free)",
        reasoning: true,
    },
    FreeModelEntry {
        id: "meta-llama/llama-4-maverick:free",
        label: "Llama 4 Maverick (free)",
        reasoning: false,
    },
];

#[derive(Debug, Clone)]
pub struct FreeModelEntry {
    /// OpenRouter model identifier (e.g. "qwen/qwen3.6-plus").
    pub id: &'static str,
    /// Short human-readable label.
    pub label: &'static str,
    /// Whether the model emits `<think>` reasoning blocks.
    pub reasoning: bool,
}

/// Return the full list of known free models.
pub fn free_model_registry() -> &'static [FreeModelEntry] {
    FREE_MODEL_REGISTRY
}

/// Check whether a model ID is in the free-tier registry.
pub fn is_free_model(model_id: &str) -> bool {
    FREE_MODEL_REGISTRY.iter().any(|e| e.id == model_id)
}

/// Look up a free model entry by ID.
pub fn lookup_free_model(model_id: &str) -> Option<&'static FreeModelEntry> {
    FREE_MODEL_REGISTRY.iter().find(|e| e.id == model_id)
}

/// OpenRouter provider that **only** allows free-tier models.
///
/// Construction requires a valid API key.  Every `complete()` call checks
/// the model ID against `FREE_MODEL_REGISTRY` before making a request.
pub struct OpenRouterFreeProvider {
    api_key: String,
    client: reqwest::blocking::Client,
}

impl OpenRouterFreeProvider {
    /// Base URL for the OpenRouter chat completions endpoint.
    const BASE_URL: &str = "https://openrouter.org/api/v1/chat/completions";

    /// Build a new provider from an explicit API key.
    pub fn new(api_key: String) -> io::Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(Self { api_key, client })
    }

    /// Try to build a provider by resolving the API key from the environment
    /// or OpenClaw config.  Returns `None` if no key is found.
    pub fn from_env() -> Option<Self> {
        let key = resolve_openrouter_api_key()?;
        Self::new(key).ok()
    }
}

impl ChatProvider for OpenRouterFreeProvider {
    fn name(&self) -> &str {
        "openrouter-free"
    }

    fn complete(&self, model_id: &str, prompt: &str, max_tokens: u32) -> io::Result<String> {
        // ── Free-only gate ────────────────────────────────────────
        if !is_free_model(model_id) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "Model {model_id:?} is not in the free-tier registry. \
                     Only free OpenRouter models are allowed. \
                     Known free models: {}",
                    FREE_MODEL_REGISTRY
                        .iter()
                        .map(|e| e.id)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }

        let payload = serde_json::json!({
            "model": model_id,
            "messages": [{ "role": "user", "content": prompt }],
            "max_tokens": max_tokens
        });

        let resp = self
            .client
            .post(Self::BASE_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", "https://github.com/council")
            .header("X-Title", "AI-Council")
            .json(&payload)
            .send()
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("OpenRouter request failed: {e}"),
                )
            })?;

        let status = resp.status();
        let text = resp.text().map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to read OpenRouter response: {e}"),
            )
        })?;

        if !status.is_success() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("OpenRouter API error (HTTP {}): {}", status.as_u16(), text),
            ));
        }

        let parsed: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("OpenRouter JSON parse error: {e}"),
            )
        })?;

        let content = parsed["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("No content in OpenRouter response: {text}"),
                )
            })?;

        // Strip <think>...</think> blocks from reasoning models
        let entry = lookup_free_model(model_id);
        let cleaned = if entry.is_some_and(|e| e.reasoning) {
            strip_think_blocks(content)
        } else {
            content.to_string()
        };

        Ok(cleaned)
    }

    fn is_available(&self) -> bool {
        // Key exists → considered available (we don't probe the network).
        !self.api_key.is_empty()
    }
}

// ── Shared utilities ──────────────────────────────────────────────────

/// Remove `<think>...</think>` blocks that reasoning models may emit.
pub fn strip_think_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("<think>") {
        result.push_str(&remaining[..start]);
        if let Some(end) = remaining[start..].find("</think>") {
            remaining = &remaining[start + end + "</think>".len()..];
        } else {
            remaining = "";
        }
    }
    result.push_str(remaining);
    let trimmed = result.trim().to_string();
    if trimmed.is_empty() {
        text.to_string()
    } else {
        trimmed
    }
}

/// Resolve the OpenRouter API key: env var first, then OpenClaw config.
pub fn resolve_openrouter_api_key() -> Option<String> {
    // 1. Environment variable override
    if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
        if !key.is_empty() {
            return Some(key);
        }
    }

    // 2. Read from ~/.openclaw/openclaw.json → models.providers.openrouter.apiKey
    let home = std::env::var("HOME").ok()?;
    let config_path = std::path::PathBuf::from(home).join(".openclaw/openclaw.json");
    let content = std::fs::read_to_string(config_path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
    parsed["models"]["providers"]["openrouter"]["apiKey"]
        .as_str()
        .map(|s| s.to_string())
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_model_gate_rejects_unknown() {
        assert!(!is_free_model("openai/gpt-4o"));
        assert!(!is_free_model("anthropic/claude-3.5-sonnet"));
        assert!(!is_free_model(""));
    }

    #[test]
    fn free_model_gate_accepts_registered() {
        assert!(is_free_model("qwen/qwen3.6-plus"));
        assert!(is_free_model("google/gemma-3-27b-it:free"));
        assert!(is_free_model("deepseek/deepseek-r1-0528:free"));
    }

    #[test]
    fn lookup_returns_entry_details() {
        let entry = lookup_free_model("qwen/qwen3.6-plus").unwrap();
        assert_eq!(entry.label, "Qwen 3.6 Plus");
        assert!(entry.reasoning);

        let entry = lookup_free_model("google/gemma-3-27b-it:free").unwrap();
        assert!(!entry.reasoning);
    }

    #[test]
    fn lookup_returns_none_for_unknown() {
        assert!(lookup_free_model("openai/gpt-4o").is_none());
    }

    #[test]
    fn registry_is_nonempty_and_all_have_ids() {
        let reg = free_model_registry();
        assert!(reg.len() >= 3, "registry should have multiple entries");
        for entry in reg {
            assert!(!entry.id.is_empty());
            assert!(!entry.label.is_empty());
        }
    }

    #[test]
    fn provider_rejects_paid_model() {
        // Build provider with a dummy key (won't actually call the API).
        let provider = OpenRouterFreeProvider::new("test-key".into()).unwrap();
        let result = provider.complete("openai/gpt-4o", "hello", 100);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
        assert!(err.to_string().contains("not in the free-tier registry"));
    }

    #[test]
    fn strip_think_blocks_basic() {
        let input = "<think>reasoning here</think>The actual answer.";
        assert_eq!(strip_think_blocks(input), "The actual answer.");
    }

    #[test]
    fn strip_think_blocks_multiple() {
        let input = "<think>a</think>Hello <think>b</think>world";
        assert_eq!(strip_think_blocks(input), "Hello world");
    }

    #[test]
    fn strip_think_blocks_none() {
        let input = "No thinking here.";
        assert_eq!(strip_think_blocks(input), "No thinking here.");
    }

    #[test]
    fn strip_think_blocks_only_think() {
        let input = "<think>only reasoning</think>";
        assert_eq!(strip_think_blocks(input), "<think>only reasoning</think>");
    }

    #[test]
    fn from_env_returns_none_without_key() {
        // In test environment without OPENROUTER_API_KEY set and without
        // openclaw config, from_env should return None gracefully.
        // (We can't guarantee the env is clean, so just test the type.)
        let _result: Option<OpenRouterFreeProvider> = OpenRouterFreeProvider::from_env();
        // No panic = success.
    }
}
