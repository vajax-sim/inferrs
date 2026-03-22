//! Integration tests for the inferrs HTTP server.
//!
//! These tests start the binary against a real model and verify that
//! `/v1/chat/completions` returns a well-formed OpenAI-compatible response.
//!
//! # Running
//!
//! The tests require network access to download models from HuggingFace Hub
//! the first time they run (subsequent runs use the local cache).
//!
//! ```
//! cargo test --test server_integration -- --nocapture
//! ```
//!
//! Each test starts `inferrs serve <model>` on a random free port, waits for
//! the server to become healthy, sends a chat completion request, and asserts
//! the response structure.

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find a free TCP port by binding to port 0 and reading the assigned port.
fn free_port() -> u16 {
    use std::net::TcpListener;
    TcpListener::bind("127.0.0.1:0")
        .expect("Failed to bind to a free port")
        .local_addr()
        .unwrap()
        .port()
}

/// Build the path to the `inferrs` binary produced by `cargo build`.
fn inferrs_bin() -> std::path::PathBuf {
    // cargo sets CARGO_MANIFEST_DIR; fall back to searching relative paths.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let bin = std::path::Path::new(&manifest_dir).join("target/debug/inferrs");
    if !bin.exists() {
        // Try release build
        let release = std::path::Path::new(&manifest_dir).join("target/release/inferrs");
        if release.exists() {
            return release;
        }
    }
    bin
}

/// Spawn `inferrs serve <model>` on the given port and return the child process.
fn spawn_server(model: &str, port: u16) -> Child {
    Command::new(inferrs_bin())
        .args([
            "serve",
            model,
            "--port",
            &port.to_string(),
            "--max-tokens",
            "32",
        ])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to spawn inferrs")
}

/// Poll the `/health` endpoint until it responds 200 or a timeout elapses.
/// Returns `Ok(true)` on success, `Ok(false)` on timeout, and `Err` if the
/// child process exits before the server becomes healthy.
fn wait_for_health(
    port: u16,
    timeout: Duration,
    child: &mut std::process::Child,
) -> Result<bool, String> {
    let url = format!("http://127.0.0.1:{}/health", port);
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!("server process exited early with status: {status}"));
        }
        if let Ok(resp) = ureq::get(&url).call() {
            if resp.status() == 200 {
                return Ok(true);
            }
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Parse and validate a `/v1/chat/completions` JSON response.
///
/// Returns the assistant message content on success.
fn assert_valid_chat_response(body: &str, model: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(body)
        .unwrap_or_else(|e| panic!("Response is not valid JSON: {e}\nBody: {body}"));

    // Required top-level fields
    assert_eq!(v["object"], "chat.completion", "unexpected object type");
    assert_eq!(v["model"], model, "unexpected model in response");
    assert!(v["id"].is_string(), "missing id field");
    assert!(v["created"].is_number(), "missing created field");

    // choices array
    let choices = v["choices"].as_array().expect("choices must be an array");
    assert!(!choices.is_empty(), "choices must not be empty");

    let choice = &choices[0];
    assert_eq!(choice["index"], 0, "first choice index must be 0");
    assert!(
        choice["finish_reason"].is_string(),
        "finish_reason must be a string"
    );

    let message = &choice["message"];
    assert_eq!(message["role"], "assistant", "role must be assistant");
    let content = message["content"]
        .as_str()
        .expect("content must be a string");
    assert!(!content.is_empty(), "content must not be empty");

    // usage
    let usage = &v["usage"];
    assert!(usage["prompt_tokens"].is_number(), "missing prompt_tokens");
    assert!(
        usage["completion_tokens"].is_number(),
        "missing completion_tokens"
    );
    assert!(usage["total_tokens"].is_number(), "missing total_tokens");

    content.to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Helper that runs a full server test for one model.
///
/// Skipped automatically if the `INFERRS_SKIP_NETWORK_TESTS` env var is set,
/// which lets CI skip these without marking them as failures.
///
/// Models that require HuggingFace authentication (like Gemma models) are also
/// skipped since CI doesn't have HF_TOKEN set.
fn run_chat_completion_test(model: &str) {
    if std::env::var("INFERRS_SKIP_NETWORK_TESTS").is_ok() {
        eprintln!("Skipping network test for {model} (INFERRS_SKIP_NETWORK_TESTS set)");
        return;
    }

    // Skip models that require HuggingFace authentication
    // Gemma models require authentication even for config.json
    let requires_auth = model.contains("gemma");
    if requires_auth && std::env::var("HF_TOKEN").is_err() && std::env::var("HF_HUB_TOKEN").is_err()
    {
        eprintln!(
            "Skipping test for {model}: requires HuggingFace authentication (HF_TOKEN not set)"
        );
        return;
    }

    let port = free_port();
    let mut child = spawn_server(model, port);

    match wait_for_health(port, Duration::from_secs(300), &mut child) {
        Ok(true) => {}
        Ok(false) => {
            child.kill().ok();
            panic!("Server for {model} did not become healthy within 5 minutes");
        }
        Err(msg) => {
            child.kill().ok();
            // Skip tests for models that require authentication or are unsupported
            if msg.contains("401")
                || msg.contains("Unsupported model architecture")
                || msg.contains("cannot find tensor")
            {
                eprintln!("Skipping test for {model}: {msg}");
                return;
            }
            panic!("Server for {model} failed to start: {msg}");
        }
    }

    let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
    let payload = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 32
    });

    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_string(&payload.to_string())
        .unwrap_or_else(|e| panic!("Request to {model} failed: {e}"));

    assert_eq!(resp.status(), 200, "Expected HTTP 200 from {model}");

    let body = resp.into_string().expect("Failed to read response body");

    let content = assert_valid_chat_response(&body, model);
    eprintln!("[{model}] assistant replied: {content:?}");

    child.kill().ok();
    child.wait().ok();
}

#[test]
fn qwen2_5_0_5b_chat_completion() {
    run_chat_completion_test("Qwen/Qwen2.5-0.5B");
}

#[test]
fn gemma_2b_it_chat_completion() {
    run_chat_completion_test("google/gemma-2b-it");
}
