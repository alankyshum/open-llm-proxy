import time
import uuid
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Callable, Optional


def get_attribution_token() -> Optional[str]:
    """Load dedicated attribution token from env or its mode-0600 file."""
    token = os.environ.get("OPEN_LLM_PROXY_ATTRIBUTION_TOKEN", "").strip()
    if token:
        return token

    path = Path(
        os.environ.get(
            "OPEN_LLM_PROXY_ATTRIBUTION_TOKEN_FILE",
            "~/.config/open-llm-proxy/attribution-token",
        )
    ).expanduser()
    try:
        metadata = path.stat()
        if not path.is_file() or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            return None
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token if token and len(token) <= 1024 else None

class AttributionStore:
    def __init__(
        self,
        capacity: int = 2048,
        ttl: float = 600.0,
        clock: Optional[Callable[[], float]] = None,
    ):
        if capacity < 1 or ttl <= 0:
            raise ValueError("capacity and ttl must be positive")
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock or time.monotonic
        self._lock = Lock()
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def set(self, attr_id: str, served_by: str) -> None:
        normalized_id = self._normalize_uuid(attr_id)
        if normalized_id is None or not isinstance(served_by, str):
            return
        served_by = served_by.strip()
        if not served_by or len(served_by) > 256:
            return
        now = self.clock()
        with self._lock:
            self._evict_expired(now)
            self._store.pop(normalized_id, None)
            self._store[normalized_id] = (served_by, now + self.ttl)
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def get(self, attr_id: str) -> Optional[str]:
        normalized_id = self._normalize_uuid(attr_id)
        if normalized_id is None:
            return None
        now = self.clock()
        with self._lock:
            self._evict_expired(now)
            value = self._store.get(normalized_id)
            return value[0] if value is not None else None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @staticmethod
    def _normalize_uuid(value: object) -> Optional[str]:
        if not isinstance(value, str):
            return None
        try:
            parsed = uuid.UUID(value)
        except (ValueError, TypeError, AttributeError):
            return None
        return str(parsed) if value.lower() == str(parsed) else None

    def _evict_expired(self, now: float) -> None:
        expired = [key for key, value in self._store.items() if value[1] <= now]
        for key in expired:
            self._store.pop(key, None)

# Global store instance
global_attribution_store = AttributionStore()
