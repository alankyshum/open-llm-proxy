from __future__ import annotations

from litellm.llms.custom_llm import CustomLLMError


class RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.headers = headers or {}


class TranslationError(ValueError):
    """Raised when translation fails."""


def map_rate_limit_error(err: Exception) -> CustomLLMError:
    if isinstance(err, RateLimitError):
        return CustomLLMError(status_code=429, message=str(err))
    return CustomLLMError(status_code=500, message=str(err))

