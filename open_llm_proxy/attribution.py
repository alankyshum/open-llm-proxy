import time
import uuid
import os
import re
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional


ATTRIBUTION_HEADER = "x-open-llm-proxy-attribution-id"
_SERVED_BY_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,256}$")


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
        ttl: float = 86400.0,
        clock: Optional[Callable[[], float]] = None,
    ):
        if capacity < 1 or ttl <= 0:
            raise ValueError("capacity and ttl must be positive")
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock or time.monotonic
        self._lock = Lock()
        # UUID -> (latest winner, last announced winner, expiry)
        self._store: OrderedDict[str, tuple[str, Optional[str], float]] = OrderedDict()

    def set(self, attr_id: str, served_by: str) -> None:
        normalized_id = self._normalize_uuid(attr_id)
        served_by = self._normalize_served_by(served_by)
        if normalized_id is None or served_by is None:
            return
        now = self.clock()
        with self._lock:
            self._evict_expired(now)
            previous = self._store.pop(normalized_id, None)
            announced = previous[1] if previous is not None else None
            self._put(normalized_id, served_by, announced, now)

    def announce_if_changed(self, attr_id: str, served_by: str) -> bool:
        """Record *served_by* and atomically claim a changed-winner banner."""
        normalized_id = self._normalize_uuid(attr_id)
        served_by = self._normalize_served_by(served_by)
        if normalized_id is None or served_by is None:
            return False
        now = self.clock()
        with self._lock:
            self._evict_expired(now)
            previous = self._store.pop(normalized_id, None)
            announced = previous[1] if previous is not None else None
            changed = announced != served_by
            self._put(normalized_id, served_by, served_by, now)
            return changed

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

    @staticmethod
    def _normalize_served_by(value: object) -> Optional[str]:
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if _SERVED_BY_RE.fullmatch(value) else None

    def _put(
        self,
        attr_id: str,
        served_by: str,
        announced: Optional[str],
        now: float,
    ) -> None:
        self._store[attr_id] = (served_by, announced, now + self.ttl)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def _evict_expired(self, now: float) -> None:
        expired = [key for key, value in self._store.items() if value[2] <= now]
        for key in expired:
            self._store.pop(key, None)


def attribution_id_from_headers(headers: object) -> Optional[str]:
    if not isinstance(headers, Mapping):
        return None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == ATTRIBUTION_HEADER:
            return AttributionStore._normalize_uuid(value)
    return None


def attribution_id_from_data(data: object) -> Optional[str]:
    """Find the OpenCode session attribution ID in LiteLLM request metadata."""
    if not isinstance(data, Mapping):
        return None

    containers: list[Mapping[str, Any]] = [data]
    litellm_params = data.get("litellm_params")
    if isinstance(litellm_params, Mapping):
        containers.append(litellm_params)

    for container in containers:
        proxy_request = container.get("proxy_server_request")
        if isinstance(proxy_request, Mapping):
            attr_id = attribution_id_from_headers(proxy_request.get("headers"))
            if attr_id:
                return attr_id
        metadata = container.get("metadata")
        if isinstance(metadata, Mapping):
            attr_id = attribution_id_from_headers(metadata.get("headers"))
            if attr_id:
                return attr_id
    return None


def served_by_from_data(data: object) -> Optional[str]:
    """Extract the concrete deployment key from LiteLLM request metadata."""
    if not isinstance(data, Mapping):
        return None

    containers: list[Mapping[str, Any]] = [data]
    for name in ("deployment", "litellm_params"):
        value = data.get(name)
        if isinstance(value, Mapping):
            containers.append(value)

    for container in containers:
        model_info = container.get("model_info")
        if isinstance(model_info, Mapping):
            key = AttributionStore._normalize_served_by(model_info.get("rate_limit_key"))
            if key:
                return key
    return None

# Global store instance
global_attribution_store = AttributionStore()
