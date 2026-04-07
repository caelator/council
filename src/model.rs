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
}

impl Model {
    pub fn name(self) -> &'static str {
        match self {
            Model::Gemini => "gemini",
            Model::Claude => "claude",
            Model::Codex => "codex",
        }
    }
}

pub struct ModelOutput {
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
}

pub fn run_model(model: Model, prompt: &str, workdir: &Path) -> io::Result<ModelOutput> {
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
    }?;

    Ok(ModelOutput {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        success: output.status.success(),
    })
}

static AVAILABILITY_CACHE: std::sync::LazyLock<Mutex<HashMap<Model, bool>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn check_available(model: Model) -> bool {
    let mut cache = AVAILABILITY_CACHE.lock().unwrap();
    if let Some(&cached) = cache.get(&model) {
        return cached;
    }
    let available = Command::new("sh")
        .args(["-c", &format!("command -v {}", model.name())])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    cache.insert(model, available);
    available
}
