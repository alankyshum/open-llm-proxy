import os
import sys
import json
import hmac
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
from open_llm_proxy.callbacks import (
    FallbackChainCommaRewriterCallback,
    GeminiThinkingBudgetCallback,
    OllamaReasoningStripCallback,
    ServedByCallback,
    StickyRoutingCallback,
)
from open_llm_proxy.config_gen import (
    configured_model_tokens,
    configured_model_tokens_from_data,
    generate_config_from_data,
    parse_agent_config,
)
from open_llm_proxy.rate_limit_state import (
    PersistentRateLimitCallback,
    load_rate_limit_policy,
    load_rate_limit_policy_from_data,
)
from open_llm_proxy.streaming_safety import (
    install_non_stream_attribution,
    install_pre_first_chunk_fallback_only,
)
from open_llm_proxy.gemini_isolation import install_gemini_shared_state_isolation
from open_llm_proxy.reloader import ConfigReloader
print("SERVER_LAUNCHER: All imports done.", flush=True)

log = logging.getLogger("open_llm_proxy.server_launcher")


def register_attribution_endpoint(app):
    """Register authenticated loopback lookup for request attribution."""
    from fastapi import HTTPException, Request, Response
    from open_llm_proxy.attribution import get_attribution_token, global_attribution_store

    @app.get("/internal/attribution/v1/{attribution_id}")
    async def get_attribution(attribution_id: str, request: Request):
        if not request.client or request.client.host not in ("127.0.0.1", "::1"):
            raise HTTPException(status_code=403, detail="Forbidden")

        token = get_attribution_token()
        authorization = request.headers.get("Authorization", "")
        if not token or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized")
        if not hmac.compare_digest(authorization[7:], token):
            raise HTTPException(status_code=401, detail="Unauthorized")

        served_by = global_attribution_store.get(attribution_id)
        if not served_by:
            raise HTTPException(status_code=404, detail="Not Found")

        return Response(
            content=json.dumps({"servedBy": served_by}),
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )

    return get_attribution


def register_config_reload_endpoint(app, config_reloader):
    """Register loopback-only routing reload endpoint."""
    from fastapi import HTTPException, Request

    @app.post("/internal/config/reload")
    async def reload_config(request: Request):
        if not request.client or request.client.host not in ("127.0.0.1", "::1"):
            raise HTTPException(status_code=403, detail="Forbidden")

        try:
            payload = await request.json()
        except Exception as error:
            raise HTTPException(status_code=400, detail="Invalid JSON") from error

        expected_hash = payload.get("expected_hash") if isinstance(payload, dict) else None
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(char not in "0123456789abcdef" for char in expected_hash)
        ):
            raise HTTPException(status_code=400, detail="Invalid expected_hash")

        applied = await config_reloader.reload(expected_hash=expected_hash)
        if not applied or config_reloader.active_hash != expected_hash:
            raise HTTPException(
                status_code=409,
                detail="Routing config reload rejected; full restart required",
            )

        return {"status": "ok", "config_hash": config_reloader.active_hash}

    return reload_config

# ---------------------------------------------------------------------------
# Stable config lifecycle  — replaces ephemeral NamedTemporaryFile path
# (LiteLLM's APScheduler re-reads the config every 30 s; a missing temp
# path causes repeated "Config file not found" / credential DB errors.)
# ---------------------------------------------------------------------------
STABLE_CONFIG_BASENAME = "generated-litellm-config.yaml"

def _resolve_stable_config_dir() -> Path:
    """Directory for the generated LiteLLM config.

    Respects OPEN_LLM_PROXY_CONFIG_DIR env var if set, otherwise
    ``~/.config/open-llm-proxy`` (the canonical config directory already
    used for agent-config.yml, env, and the rate-limit SQLite DB).
    """
    env_dir = os.environ.get("OPEN_LLM_PROXY_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".config" / "open-llm-proxy"

def resolve_stable_config_path() -> Path:
    """Persistent config path scoped to this proxy process."""
    stem = Path(STABLE_CONFIG_BASENAME).stem
    suffix = Path(STABLE_CONFIG_BASENAME).suffix
    return _resolve_stable_config_dir() / f"{stem}-{os.getpid()}{suffix}"

def write_config_atomic(config_dict: dict, path: str | Path | None = None) -> str:
    """Write *config_dict* as YAML to *path* atomically with mode 0o600.

    Uses a same‑directory temp file + ``os.replace`` so the target is
    never torn (partial write).  Mode is enforced on the temp before
    replacement and again on the final path after replacement.

    Returns the final path as a string.
    """
    if path is None:
        path = resolve_stable_config_path()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f".{STABLE_CONFIG_BASENAME}.",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, str(path))
        # Mode may not survive replace on some filesystems; re‑apply.
        path.chmod(0o600)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return str(path)

def setup_callbacks(
    config_path: str | Path | None = None, *, config_data: dict | None = None
):
    """Register request transforms and persistent rate-limit tracking."""
    install_pre_first_chunk_fallback_only()
    install_gemini_shared_state_isolation()
    if not hasattr(litellm, "callbacks") or litellm.callbacks is None:
        litellm.callbacks = []

    rate_limit_callback = next(
        (c for c in litellm.callbacks if isinstance(c, PersistentRateLimitCallback)),
        None,
    )
    if (config_path is not None or config_data is not None) and rate_limit_callback is None:
        if config_data is not None:
            policy = load_rate_limit_policy_from_data(config_data)
            policy["model_keys"] = configured_model_tokens_from_data(config_data)
        else:
            policy = load_rate_limit_policy(config_path)
            policy["model_keys"] = configured_model_tokens(config_path)
        rate_limit_callback = PersistentRateLimitCallback(**policy)
        litellm.callbacks.append(rate_limit_callback)
        log.info("PersistentRateLimitCallback registered.")

    # Register StickyRoutingCallback after PersistentRateLimitCallback
    if not any(isinstance(c, StickyRoutingCallback) for c in litellm.callbacks):
        litellm.callbacks.append(StickyRoutingCallback())
        log.info("StickyRoutingCallback registered.")

    # Register Gemini thinking budget callback
    if not any(isinstance(c, GeminiThinkingBudgetCallback) for c in litellm.callbacks):
        gemini_callback = GeminiThinkingBudgetCallback()
        litellm.callbacks.append(gemini_callback)
        log.info("GeminiThinkingBudgetCallback registered.")

    # Register Fallback chain comma rewriter callback
    if not any(
        isinstance(c, FallbackChainCommaRewriterCallback) for c in litellm.callbacks
    ):
        rewriter_callback = FallbackChainCommaRewriterCallback()
        litellm.callbacks.append(rewriter_callback)
        log.info("FallbackChainCommaRewriterCallback registered.")

    if not any(isinstance(c, ServedByCallback) for c in litellm.callbacks):
        litellm.callbacks.append(ServedByCallback())
        log.info("ServedByCallback registered.")

    if not any(isinstance(c, OllamaReasoningStripCallback) for c in litellm.callbacks):
        litellm.callbacks.append(OllamaReasoningStripCallback())
        log.info("OllamaReasoningStripCallback registered.")

    return rate_limit_callback

def find_agent_config() -> Path:
    """
    Finds agent-config.yml by traversing upwards or looking in the dotfiles workspace.
    """
    # Prefer non-TCC fallback path under .config if it exists
    config_in_home = Path.home() / ".config" / "open-llm-proxy" / "agent-config.yml"
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

def launch_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    disable_admin_ui: bool | None = None,
    database_url: str | None = None,
    master_key: str | None = None,
    ui_username: str | None = None,
    ui_password: str | None = None,
    config_path: str | Path | None = None,
):
    """
    Launches the programmatic LiteLLM proxy server on the specified port.
    """
    print("SERVER_LAUNCHER: Finding agent config...", flush=True)
    if config_path is None:
        config_path = find_agent_config()
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(
            f"Error: agent-config.yml not found at {config_path}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    # --- Resolve Admin UI configuration (argument > environment) BEFORE doing any
    # side effects (tempfile, app import) so validation failures leave no debris. ---
    #
    # The Admin UI is a first-class, ON-by-default feature. It only requires a
    # PostgreSQL DATABASE_URL and a master key. When a database is configured we
    # auto-provision a master key so the proxy never crash-loops on a missing
    # secret. UI is disabled only when explicitly requested OR when no database
    # is configured (LiteLLM cannot serve the Admin UI without Postgres).
    env_disable_ui = os.environ.get("DISABLE_ADMIN_UI", "").lower() in ("true", "1")

    resolved_database_url = database_url if database_url is not None else os.environ.get("DATABASE_URL")
    resolved_master_key = master_key if master_key is not None else os.environ.get("LITELLM_MASTER_KEY")
    resolved_ui_username = ui_username if ui_username is not None else os.environ.get("UI_USERNAME")
    resolved_ui_password = ui_password if ui_password is not None else os.environ.get("UI_PASSWORD")

    # Determine whether the UI should run.
    if disable_admin_ui is True:
        ui_disabled = True
    elif env_disable_ui:
        ui_disabled = True
    elif not resolved_database_url:
        # No database => UI cannot function; fall back to DB-less proxy mode
        # instead of crashing. This keeps `serve` working out of the box.
        log.warning(
            "DATABASE_URL not set; starting in DB-less mode with the Admin UI "
            "disabled. Start PostgreSQL (podman compose up -d) and set "
            "DATABASE_URL to enable the Admin UI at /ui."
        )
        print(
            "UI Status: Disabled (no DATABASE_URL — DB-less mode)",
            flush=True,
        )
        ui_disabled = True
    else:
        ui_disabled = False

    if not resolved_master_key:
        import secrets

        resolved_master_key = f"sk-{secrets.token_urlsafe(32)}"
        log.warning(
            "LITELLM_MASTER_KEY not set; generated an ephemeral master key "
            "for this run. Set LITELLM_MASTER_KEY in "
            "~/.config/open-llm-proxy/env to make it stable."
        )

    # ALWAYS ensure LITELLM_MASTER_KEY is set in env before importing proxy_server
    os.environ["LITELLM_MASTER_KEY"] = resolved_master_key

    print(f"Generating LiteLLM config from: {config_path}", flush=True)
    config_source = config_path.read_bytes()
    config_data = parse_agent_config(config_source)
    config_dict = generate_config_from_data(config_data)

    print("SERVER_LAUNCHER: Calling setup_callbacks()...", flush=True)
    rate_limit_callback = setup_callbacks(config_data=config_data)

    # Keep each process's config available for LiteLLM scheduler re-reads.
    stable_config_path = write_config_atomic(config_dict)
    print(f"Generated LiteLLM config at: {stable_config_path}")

    # Set the environment variable so LiteLLM's module load catches it
    os.environ["CONFIG_FILE_PATH"] = stable_config_path

    if ui_disabled:
        os.environ["DISABLE_ADMIN_UI"] = "True"
    else:
        # Clear any stale disable flag so the UI actually serves.
        os.environ.pop("DISABLE_ADMIN_UI", None)

    # Publish resolved environment variables for LiteLLM's proxy module.
    if resolved_database_url:
        os.environ["DATABASE_URL"] = resolved_database_url
    if resolved_master_key:
        os.environ["LITELLM_MASTER_KEY"] = resolved_master_key
    if resolved_ui_username:
        os.environ["UI_USERNAME"] = resolved_ui_username
    if resolved_ui_password:
        os.environ["UI_PASSWORD"] = resolved_ui_password

    if not ui_disabled:
        print("UI Status: Enabled", flush=True)
        print(f"Admin UI: http://{host}:{port}/ui", flush=True)
        print(
            f"UI Username: {resolved_ui_username if resolved_ui_username else 'admin (LiteLLM default)'}",
            flush=True,
        )
    
    try:
        # Import app after CONFIG_FILE_PATH has been set to ensure the module-level load of
        # litellm.proxy.proxy_server parses the correct configuration.
        from litellm.proxy.proxy_server import app
        install_non_stream_attribution()
        from open_llm_proxy.usage_reporting import install_usage_reporting
        install_usage_reporting(app)
        from open_llm_proxy.model_chain_middleware import install_model_chain_middleware
        install_model_chain_middleware(app)

        config_reloader = ConfigReloader(
            source_path=config_path,
            generated_path=stable_config_path,
            initial_source=config_source,
            initial_config=config_dict,
            write_config=write_config_atomic,
            rate_limit_callback=rate_limit_callback,
        )
        
        # Add healthz endpoint for sync-agents validation
        @app.get("/healthz")
        async def healthz():
            return {"status": "ok", "config_hash": config_reloader.active_hash}

        register_attribution_endpoint(app)
        register_config_reload_endpoint(app, config_reloader)
        
        if ui_disabled:
            from fastapi import Response
            @app.middleware("http")
            async def block_ui_middleware(request, call_next):
                if request.url.path == "/ui" or request.url.path.startswith("/ui/") or request.url.path.startswith("/_next"):
                    return Response("Admin UI is disabled.", status_code=404)
                return await call_next(request)
        
        print(f"Starting programmatically on {host}:{port}...", flush=True)
        uvicorn.run(app, host=host, port=port)
    except Exception:
        log.exception("Fatal error during server startup")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch open-llm-proxy programmatically")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument(
        "--disable-admin-ui",
        action="store_true",
        default=None,
        help="Disable the LiteLLM Admin UI (DB-less proxy mode).",
    )
    parser.add_argument("--database-url", default=None, help="PostgreSQL URL for the Admin UI.")
    parser.add_argument("--master-key", default=None, help="Master key securing the Admin UI.")
    parser.add_argument("--ui-username", default=None, help="Admin UI login username.")
    parser.add_argument("--ui-password", default=None, help="Admin UI login password.")
    parser.add_argument("--config-path", default=None, help="Path to agent-config.yml.")
    args = parser.parse_args()
    launch_server(
        host=args.host,
        port=args.port,
        disable_admin_ui=args.disable_admin_ui,
        database_url=args.database_url,
        master_key=args.master_key,
        ui_username=args.ui_username,
        ui_password=args.ui_password,
        config_path=args.config_path,
    )
