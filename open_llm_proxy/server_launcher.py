import os
import sys
print("SERVER_LAUNCHER: Python started, importing modules...", flush=True)
import tempfile
import logging
import uvicorn
import yaml
from pathlib import Path

# Wire callback registration
print("SERVER_LAUNCHER: Importing litellm...", flush=True)
import litellm
print("SERVER_LAUNCHER: Importing open_llm_proxy callbacks/config...", flush=True)
from open_llm_proxy.callbacks import GeminiThinkingBudgetCallback, FallbackChainCommaRewriterCallback
from open_llm_proxy.config_gen import generate_config
print("SERVER_LAUNCHER: All imports done.", flush=True)

log = logging.getLogger("open_llm_proxy.server_launcher")

def setup_callbacks():
    """
    Registers the required gemini thinking budget and fallback chain comma rewriter callbacks.
    """
    if not hasattr(litellm, "callbacks") or litellm.callbacks is None:
        litellm.callbacks = []
    
    # Register Gemini thinking budget callback
    if not any(isinstance(c, GeminiThinkingBudgetCallback) for c in litellm.callbacks):
        gemini_callback = GeminiThinkingBudgetCallback()
        litellm.callbacks.append(gemini_callback)
        log.info("GeminiThinkingBudgetCallback registered.")

    # Register Fallback chain comma rewriter callback
    if not any(isinstance(c, FallbackChainCommaRewriterCallback) for c in litellm.callbacks):
        rewriter_callback = FallbackChainCommaRewriterCallback()
        litellm.callbacks.append(rewriter_callback)
        log.info("FallbackChainCommaRewriterCallback registered.")

def find_agent_config() -> Path:
    """
    Finds agent-config.yml by traversing upwards or looking in the dotfiles workspace.
    """
    # Prefer non-TCC fallback path under .config if it exists
    config_in_home = Path.home() / ".config" / "kilo-claude-proxy" / "agent-config.yml"
    if config_in_home.exists():
        return config_in_home

    # 1. Check direct env var or relative to current working directory
    paths_to_check = [
        Path(os.getcwd()) / "config/agent-runtime/agent-config.yml",
        Path(__file__).resolve().parents[3] / "config/agent-runtime/agent-config.yml",
    ]
    for p in paths_to_check:
        if p.exists():
            return p
    # Fallback default
    return Path("/Users/alanshum/Documents/dotfiles/config/agent-runtime/agent-config.yml")

def launch_server(host: str = "0.0.0.0", port: int = 8765):
    """
    Launches the programmatic LiteLLM proxy server on the specified port.
    """
    print("SERVER_LAUNCHER: Calling setup_callbacks()...", flush=True)
    setup_callbacks()
    
    print("SERVER_LAUNCHER: Finding agent config...", flush=True)
    config_path = find_agent_config()
    if not config_path.exists():
        print(f"Error: agent-config.yml not found at {config_path}", file=sys.stderr, flush=True)
        sys.exit(1)
        
    print(f"Generating LiteLLM config from: {config_path}", flush=True)
    config_dict = generate_config(str(config_path))
    
    # Use Ephemeral tempfile fallback as LiteLLM 1.83.7 expects a file path
    # write to temp file, register on CONFIG_FILE_PATH, then launch.
    with tempfile.NamedTemporaryFile(mode="w", suffix="_litellm_config.yaml", delete=False) as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
        temp_config_path = f.name
        
    print(f"Created ephemeral config file at: {temp_config_path}")
    
    # Set the environment variable so LiteLLM's module load catches it
    os.environ["CONFIG_FILE_PATH"] = temp_config_path
    
    # Also set litellm-specific settings
    os.environ["LITELLM_MASTER_KEY"] = "sk-local"
    
    try:
        # Import app after CONFIG_FILE_PATH has been set to ensure the module-level load of
        # litellm.proxy.proxy_server parses the correct configuration.
        from litellm.proxy.proxy_server import app
        
        # Add healthz endpoint for sync-agents validation
        @app.get("/healthz")
        async def healthz():
            return {"status": "ok"}
        
        print(f"Starting programmatically on {host}:{port}...", flush=True)
        uvicorn.run(app, host=host, port=port)
    finally:
        # Clean up the ephemeral config file on shutdown
        if os.path.exists(temp_config_path):
            try:
                os.remove(temp_config_path)
                print(f"Deleted ephemeral config file: {temp_config_path}")
            except Exception as e:
                print(f"Warning: failed to delete ephemeral config file {temp_config_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Allow passing host/port from cli
    import argparse
    parser = argparse.ArgumentParser(description="Launch open-llm-proxy programmatically")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    args = parser.parse_args()
    launch_server(host=args.host, port=args.port)
