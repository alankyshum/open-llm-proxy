import json

from fastapi import FastAPI, Request


def rewrite_model_chain_body(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body

    if not isinstance(data, dict) or not isinstance(data.get("model"), str):
        return body

    model = data["model"]
    if model.startswith("open-llm-proxy/"):
        model = model[len("open-llm-proxy/") :]
    if not (model.startswith("[") and model.endswith("]") and "," in model):
        return body

    data["model"] = model.replace(",", ";")
    return json.dumps(data).encode("utf-8")


def install_model_chain_middleware(app: FastAPI):
    """Rewrite public comma-form chains to LiteLLM's internal semicolon alias."""

    @app.middleware("http")
    async def rewrite_model_chain_middleware(request: Request, call_next):
        if request.method == "POST" and request.url.path in (
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/chat/completions/",
            "/chat/completions/",
        ):
            body = await request.body()
            request._body = rewrite_model_chain_body(body)
            if hasattr(request, "_json"):
                delattr(request, "_json")
        return await call_next(request)
