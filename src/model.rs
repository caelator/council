use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

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
}

pub struct ModelOutput {
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
}

pub fn run_model(model: Model, prompt: &str, workdir: &Path) -> io::Result<ModelOutput> {
    match model {
        Model::Qwen36Plus => run_openrouter_model(prompt),
        _ => run_local_model(model, prompt, workdir),
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
            .args(["exec", "--full-auto", "-C", workdir.to_str().unwrap_or("."), "--skip-git-repo-check", "--add-dir", workdir.to_str().unwrap_or("."), "--", prompt])
            .output(),
        Model::Gemma31B => Command::new("gemma-llama")
            .args(["-o", "text", "-p", prompt])
            .current_dir(workdir)
            .output(),
        Model::Qwen36Plus => unreachable!("handled in run_model"),
    }?;

    Ok(ModelOutput {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        success: output.status.success(),
    })
}

// ── OpenRouter API integration ─────────────────────────────────────

/// OpenRouter model ID for Qwen 3.6 Plus.
const OPENROUTER_MODEL_ID: &str = "qwen/qwen3.6-plus";

/// Call the OpenRouter chat completions API.
fn run_openrouter_model(prompt: &str) -> io::Result<ModelOutput> {
    let api_key = resolve_openrouter_api_key()
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "No OpenRouter API key found (set OPENROUTER_API_KEY or configure ~/.openclaw/openclaw.json)"))?;

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    let payload = serde_json::json!({
        "model": OPENROUTER_MODEL_ID,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": 8192
    });

    let resp = client
        .post("https://openrouter.org/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .header("HTTP-Referer", "https://github.com/council")
        .header("X-Title", "AI-Council")
        .json(&payload)
        .send()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("OpenRouter request failed: {e}")))?;

    let status = resp.status();
    let text = resp.text()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to read OpenRouter response: {e}")))?;

    if !status.is_success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("OpenRouter API error (HTTP {}): {}", status.as_u16(), text),
        ));
    }

    let parsed: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("OpenRouter JSON parse error: {e}")))?;

    let content = parsed["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, format!("No content in OpenRouter response: {text}")))?;

    // Strip chain-of-thought <think>...</think> blocks if present
    let cleaned = strip_think_blocks(content);

    Ok(ModelOutput {
        stdout: cleaned,
        stderr: String::new(),
        success: true,
    })
}

/// Remove `<think>...</think>` blocks that reasoning models may emit.
fn strip_think_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("<think>") {
        result.push_str(&remaining[..start]);
        if let Some(end) = remaining[start..].find("</think>") {
            remaining = &remaining[start + end + "</think>".len()..];
        } else {
            // Unclosed <think> tag — skip the rest
            remaining = "";
        }
    }
    result.push_str(remaining);
    let trimmed = result.trim().to_string();
    if trimmed.is_empty() { text.to_string() } else { trimmed }
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

// ── Availability cache ─────────────────────────────────────────────

static AVAILABILITY_CACHE: std::sync::LazyLock<Mutex<HashMap<Model, bool>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn check_available(model: Model) -> bool {
    let mut cache = AVAILABILITY_CACHE.lock().unwrap();
    if let Some(&cached) = cache.get(&model) {
        return cached;
    }
    let available = match model {
        Model::Qwen36Plus => resolve_openrouter_api_key().is_some(),
        _ => Command::new("sh")
            .args(["-c", &format!("command -v {}", model.name())])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false),
    };
    cache.insert(model, available);
    available
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // If the entire response is a think block, return original text
        let input = "<think>only reasoning</think>";
        assert_eq!(strip_think_blocks(input), "<think>only reasoning</think>");
    }

    #[test]
    fn model_name_qwen() {
        assert_eq!(Model::Qwen36Plus.name(), "qwen-36-plus");
    }
}
