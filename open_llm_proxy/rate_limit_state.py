from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml
from litellm.integrations.custom_logger import CustomLogger

from open_llm_proxy.rate_limit_catalog import (
    DEFAULT_PLANS,
    SOURCE_CHECKED_AT,
    get_plan_policy,
)

log = logging.getLogger("open_llm_proxy.rate_limit_state")

_DEFAULT_DATABASE_PATH = (
    Path.home() / ".config" / "kilo-claude-proxy" / "state.sqlite3"
)


def load_rate_limit_policy(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as config_file:
        data = yaml.safe_load(config_file) or {}

    raw_policy = data.get("rate_limit_policy") or {}
    if not isinstance(raw_policy, dict):
        raise ValueError("rate_limit_policy must be a mapping")

    database_path = Path(
        os.path.expandvars(
            os.path.expanduser(
                str(raw_policy.get("database", _DEFAULT_DATABASE_PATH))
            )
        )
    )
    plans = raw_policy.get("plans") or {}
    if not isinstance(plans, dict):
        raise ValueError("rate_limit_policy.plans must be a mapping")
    for provider, plan in plans.items():
        if not isinstance(provider, str) or not isinstance(plan, str):
            raise ValueError("rate_limit_policy.plans must map strings to strings")
        get_plan_policy(provider, plan)

    return {"database_path": database_path, "configured_plans": plans}


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


class RateLimitStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS provider_plans (
                    provider TEXT PRIMARY KEY,
                    plan TEXT NOT NULL,
                    label TEXT NOT NULL,
                    default_cooldown_seconds INTEGER NOT NULL,
                    quota_limited INTEGER NOT NULL,
                    limits_json TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    source_checked_at TEXT NOT NULL,
                    configured_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS models (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    PRIMARY KEY (provider, model)
                );

                CREATE TABLE IF NOT EXISTS rate_limits (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    last_rate_limited_at TEXT NOT NULL,
                    retry_at TEXT NOT NULL,
                    retry_source TEXT NOT NULL,
                    PRIMARY KEY (provider, model)
                );
                """
            )

    def configure_plan(
        self, provider: str, plan: str, *, overwrite: bool = True
    ) -> None:
        policy = get_plan_policy(provider, plan)
        now = _utc_text(datetime.now(timezone.utc))
        statement = (
            "INSERT OR REPLACE"
            if overwrite
            else "INSERT OR IGNORE"
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                f"""
                {statement} INTO provider_plans (
                    provider, plan, label, default_cooldown_seconds,
                    quota_limited, limits_json, source_url,
                    source_checked_at, configured_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider,
                    plan,
                    policy.label,
                    policy.default_cooldown_seconds,
                    int(policy.quota_limited),
                    json.dumps(policy.limits, sort_keys=True),
                    policy.source_url,
                    SOURCE_CHECKED_AT,
                    now,
                ),
            )

    def seed_plans(self, configured_plans: Mapping[str, str]) -> None:
        for provider, plan in configured_plans.items():
            self.configure_plan(provider, plan, overwrite=False)

    def register_models(self, model_keys: Iterable[str]) -> None:
        now = _utc_text(datetime.now(timezone.utc))
        with self._lock, self._connect() as connection:
            for key in model_keys:
                if "/" not in key:
                    continue
                provider, model = key.split("/", 1)
                connection.execute(
                    """
                    INSERT INTO models (provider, model, first_seen_at, last_seen_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(provider, model)
                    DO UPDATE SET last_seen_at = excluded.last_seen_at
                    """,
                    (provider, model, now, now),
                )

    def configured_plan(self, provider: str) -> sqlite3.Row | None:
        with self._lock, self._connect() as connection:
            return connection.execute(
                "SELECT * FROM provider_plans WHERE provider = ?", (provider,)
            ).fetchone()

    def inventory(self) -> list[sqlite3.Row]:
        with self._lock, self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    m.provider, m.model, p.plan, p.label,
                    p.default_cooldown_seconds, p.quota_limited,
                    p.limits_json, p.source_url, p.source_checked_at
                FROM models AS m
                LEFT JOIN provider_plans AS p USING (provider)
                ORDER BY m.provider, m.model
                """
            ).fetchall()

    def ensure_default_plan(self, provider: str) -> sqlite3.Row:
        row = self.configured_plan(provider)
        if row is not None:
            return row
        try:
            plan = DEFAULT_PLANS[provider]
        except KeyError as exc:
            raise ValueError(f"no rate-limit policy for provider {provider}") from exc
        self.configure_plan(provider, plan, overwrite=False)
        row = self.configured_plan(provider)
        if row is None:
            raise RuntimeError(f"failed to configure rate-limit policy for {provider}")
        return row

    def record_rate_limit(
        self,
        key: str,
        *,
        occurred_at: datetime,
        retry_at: datetime | None,
        retry_source: str | None,
    ) -> None:
        provider, model = key.split("/", 1)
        plan = self.ensure_default_plan(provider)
        if retry_at is None:
            retry_at = occurred_at + timedelta(
                seconds=plan["default_cooldown_seconds"]
            )
            retry_source = f"plan:{plan['plan']}"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rate_limits (
                    provider, model, last_rate_limited_at, retry_at, retry_source
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(provider, model) DO UPDATE SET
                    last_rate_limited_at = excluded.last_rate_limited_at,
                    retry_at = excluded.retry_at,
                    retry_source = excluded.retry_source
                """,
                (
                    provider,
                    model,
                    _utc_text(occurred_at),
                    _utc_text(retry_at),
                    retry_source,
                ),
            )

    def retry_at(self, key: str) -> datetime | None:
        provider, model = key.split("/", 1)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT retry_at FROM rate_limits
                WHERE provider = ? AND model = ?
                """,
                (provider, model),
            ).fetchone()
        return _parse_timestamp(row["retry_at"]) if row is not None else None


def _rate_limit_key(deployment: dict[str, Any]) -> str | None:
    model_info = deployment.get("model_info") or {}
    if not isinstance(model_info, dict):
        return None
    key = model_info.get("rate_limit_key")
    return key if isinstance(key, str) and "/" in key else None


def _exception_chain(exception: Any) -> Iterable[Any]:
    seen: set[int] = set()
    pending = [exception]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(
            [
                getattr(current, "__cause__", None),
                getattr(current, "__context__", None),
                getattr(current, "original_exception", None),
            ]
        )


def _is_rate_limit_error(exception: Any) -> bool:
    for current in _exception_chain(exception):
        status_code = getattr(current, "status_code", None)
        if status_code == 429 or str(status_code) == "429":
            return True
        response_status = getattr(
            getattr(current, "response", None), "status_code", None
        )
        if response_status == 429 or str(response_status) == "429":
            return True
    return False


def _headers_from_exception(exception: Any) -> dict[str, str]:
    headers: dict[str, str] = {}
    for current in _exception_chain(exception):
        candidates = [
            getattr(current, "headers", None),
            getattr(getattr(current, "response", None), "headers", None),
        ]
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                headers.update(
                    {str(key).lower(): str(value) for key, value in candidate.items()}
                )
    return headers


def _retry_after_timestamp(value: str, now: datetime) -> datetime | None:
    try:
        seconds = float(value)
    except ValueError:
        try:
            return parsedate_to_datetime(value).astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None
    return now + timedelta(seconds=max(0, seconds))


def _reset_timestamp(value: str, now: datetime) -> datetime | None:
    try:
        number = float(value)
    except ValueError:
        try:
            return _parse_timestamp(value)
        except ValueError:
            return None
    if number > 10_000_000_000:
        return datetime.fromtimestamp(number / 1000, timezone.utc)
    if number > 1_000_000_000:
        return datetime.fromtimestamp(number, timezone.utc)
    return now + timedelta(seconds=max(0, number))


def retry_at_from_exception(
    exception: Any, now: datetime
) -> tuple[datetime | None, str | None]:
    candidates: list[tuple[datetime, str]] = []
    for current in _exception_chain(exception):
        retry_after = getattr(current, "retry_after", None)
        if isinstance(retry_after, (int, float)) and not isinstance(
            retry_after, bool
        ):
            candidates.append(
                (now + timedelta(seconds=max(0, retry_after)), "retry_after")
            )

    headers = _headers_from_exception(exception)
    if "retry-after" in headers:
        timestamp = _retry_after_timestamp(headers["retry-after"], now)
        if timestamp is not None:
            candidates.append((timestamp, "header:retry-after"))
    for name in ("x-ratelimit-reset", "x-rate-limit-reset"):
        if name in headers:
            timestamp = _reset_timestamp(headers[name], now)
            if timestamp is not None:
                candidates.append((timestamp, f"header:{name}"))

    return max(candidates, default=(None, None), key=lambda item: item[0])


class PersistentRateLimitCallback(CustomLogger):
    def __init__(
        self,
        *,
        database_path: str | Path,
        configured_plans: Mapping[str, str] | None = None,
        model_keys: Iterable[str] = (),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__()
        self.store = RateLimitStore(database_path)
        self.store.seed_plans(configured_plans or {})
        self.store.register_models(model_keys)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict[str, Any]],
        messages: Any,
        request_kwargs: dict[str, Any] | None = None,
        parent_otel_span: Any = None,
    ) -> list[dict[str, Any]]:
        now = self._clock().astimezone(timezone.utc)
        available: list[dict[str, Any]] = []
        for deployment in healthy_deployments:
            key = _rate_limit_key(deployment)
            retry_at = self.store.retry_at(key) if key is not None else None
            if retry_at is None or retry_at <= now:
                available.append(deployment)
            else:
                log.info("Skipping %s until %s", key, _utc_text(retry_at))
        return available

    def _record_failure(self, kwargs: dict[str, Any], response_obj: Any) -> None:
        exception = kwargs.get("exception") or response_obj
        if not _is_rate_limit_error(exception):
            return
        litellm_params = kwargs.get("litellm_params") or {}
        model_info = litellm_params.get("model_info") or {}
        key = model_info.get("rate_limit_key") if isinstance(model_info, dict) else None
        if not isinstance(key, str) or "/" not in key:
            return
        now = self._clock().astimezone(timezone.utc)
        retry_at, retry_source = retry_at_from_exception(exception, now)
        self.store.record_rate_limit(
            key,
            occurred_at=now,
            retry_at=retry_at,
            retry_source=retry_source,
        )
        log.warning("Rate limited %s; persisted retry policy in SQLite", key)

    def log_failure_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._record_failure(kwargs, response_obj)

    async def async_log_failure_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._record_failure(kwargs, response_obj)
