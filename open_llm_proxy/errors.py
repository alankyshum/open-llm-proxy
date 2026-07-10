from __future__ import annotations

from litellm.llms.custom_llm import CustomLLMError


class RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None, headers: dict[str, str] | None = None) -> None:
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
