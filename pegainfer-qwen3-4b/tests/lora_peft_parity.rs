use std::path::PathBuf;
use std::process::Command;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate must live under workspace root")
        .to_path_buf()
}

#[test]
#[ignore = "requires Qwen3-4B weights, CUDA, transformers, peft, and safetensors"]
fn qwen3_lora_matches_hf_peft_reference() {
    let root = repo_root();
    let model_path =
        std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.into());
    let python = std::env::var("PEGAINFER_LORA_PARITY_PYTHON").unwrap_or_else(|_| {
        let venv_python = root.join(".venv").join("bin").join("python");
        if venv_python.exists() {
            venv_python.display().to_string()
        } else {
            "python3".to_string()
        }
    });

    let output = Command::new(&python)
        .current_dir(&root)
        .arg(root.join("tools/lora/qwen3_lora_live_parity.py"))
        .arg("--model-path")
        .arg(&model_path)
        .arg("--max-tokens")
        .arg("8")
        .arg("--logprob-mean-tol")
        .arg("0.08")
        .arg("--logprob-max-tol")
        .arg("0.30")
        .output()
        .unwrap_or_else(|err| panic!("failed to run {python}: {err}"));

    if !output.status.success() {
        panic!(
            "Qwen3 LoRA PEFT parity gate failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
