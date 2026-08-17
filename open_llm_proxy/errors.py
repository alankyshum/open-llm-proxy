from __future__ import annotations

from litellm.llms.custom_llm import CustomLLMError


class RateLimitError(RuntimeError):
    def __init__(
        self, message: str, retry_after: float | None = None, headers: dict[str, str] | None = None
    ) -> None:  # intentional long protocol text or compatibility message
        super().__init__(message)
        self.retry_after = retry_after
        self.headers = headers or {}


class TranslationError(ValueError):
    """Raised when translation fails."""


def custom_rate_limit_error(
    message: str,
    *,
    retry_after: float | None = None,
    headers: dict[str, str] | None = None,
    rate_limit_origin_key: str | None = None,
) -> CustomLLMError:
    error = CustomLLMError(status_code=429, message=message)
    error.retry_after = retry_after
    error.headers = headers or {}
    error.rate_limit_origin_key = rate_limit_origin_key
    return error


def upstream_http_error(status_code: int, message: str) -> Exception:
    """Build the right exception type for a non-200 upstream HTTP response.

    A genuine client error (HTTP 400) must NOT be retried: opencode chains and
    LiteLLM's router treat retriable errors by rotating deployments and re-sending
    the SAME payload, which for a deterministic 400 just loops forever.

    LiteLLM's ``exception_type`` only maps ``CustomLLMError(status_code=400)`` to a
    retriable ``APIConnectionError`` for custom providers (github-copilot falls
    into the generic ``else`` branch that always yields APIConnectionError). So for
    400 we raise LiteLLM's own ``BadRequestError`` directly, which ``exception_type``
    passes through unchanged as a non-retriable 400. All other statuses keep the
    existing ``CustomLLMError`` behaviour.
    """
    if status_code == 400:
        from litellm.exceptions import BadRequestError

        return BadRequestError(
            message=message or "Bad Request",
            model="github-copilot",
            llm_provider="github-copilot",
        )
    return CustomLLMError(status_code=status_code, message=message or f"HTTP {status_code}")


def map_rate_limit_error(
    err: Exception, *, rate_limit_origin_key: str | None = None
) -> CustomLLMError:
    if isinstance(err, RateLimitError):
        return custom_rate_limit_error(
            str(err),
            retry_after=err.retry_after,
            headers=err.headers,
            rate_limit_origin_key=rate_limit_origin_key,
        )
    return CustomLLMError(status_code=500, message=str(err))
