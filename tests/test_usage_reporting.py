from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from open_llm_proxy.usage_reporting import (
    aggregate_and_normalize_spend_logs,
    format_date,
    format_display_model,
    install_usage_reporting,
)


def test_format_display_model():
    # Model starting with provider
    assert format_display_model("github/gh-gemini", "github") == "github/gh-gemini"
    assert format_display_model("github/gh-gemini", "GITHUB") == "github/gh-gemini"
    # Model without provider prefix
    assert format_display_model("gh-gemini", "github") == "github/gh-gemini"
    # Edge cases
    assert format_display_model("", "github") == ""
    assert format_display_model("gh-gemini", "") == "gh-gemini"


def test_format_date():
    # Datetime object
    dt = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)
    assert format_date(dt) == "Jul 10"

    # ISO strings
    assert format_date("2026-07-10T12:00:00Z") == "Jul 10"
    assert format_date("2026-07-10T12:00:00+00:00") == "Jul 10"

    # Datetime string without T
    assert format_date("2026-07-10 12:00:00") == "Jul 10"

    # Unknown format
    assert format_date("invalid-date") == "invalid-date"


def test_aggregate_and_normalize_spend_logs():
    db_rows = [
        {
            "model": "gpt-4",
            "custom_llm_provider": "openai",
            "date": "2026-07-10T00:00:00Z",
            "api_requests": 5,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        {
            "model": "openai/gpt-4",
            "custom_llm_provider": "openai",
            "date": "2026-07-10T00:00:00Z",
            "api_requests": 2,
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        },
        {
            "model": "claude-3",
            "custom_llm_provider": "anthropic",
            "date": "2026-07-11T00:00:00Z",
            "api_requests": 10,
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        },
    ]

    result = aggregate_and_normalize_spend_logs(db_rows)

    # Top 10 sorting by sum_total_tokens desc: anthropic/claude-3 has 1500, openai/gpt-4 has 450
    assert len(result) == 2

    first = result[0]
    assert first["model"] == "anthropic/claude-3"
    assert first["provider"] == "anthropic"
    assert first["sum_api_requests"] == 10
    assert first["sum_total_tokens"] == 1500
    assert first["sum_prompt_tokens"] == 1000
    assert first["sum_completion_tokens"] == 500
    assert len(first["daily_data"]) == 1
    assert first["daily_data"][0] == {
        "date": "Jul 11",
        "api_requests": 10,
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
    }

    second = result[1]
    assert second["model"] == "openai/gpt-4"
    assert second["provider"] == "openai"
    assert second["sum_api_requests"] == 7
    assert second["sum_total_tokens"] == 450
    assert second["sum_prompt_tokens"] == 300
    assert second["sum_completion_tokens"] == 150
    assert len(second["daily_data"]) == 1
    assert second["daily_data"][0] == {
        "date": "Jul 10",
        "api_requests": 7,
        "prompt_tokens": 300,
        "completion_tokens": 150,
        "total_tokens": 450,
    }


def test_install_usage_reporting_removes_and_replaces():
    app = FastAPI()
    router = APIRouter()

    @router.get("/global/activity/model")
    def original_endpoint():
        return "original"

    app.include_router(router)

    # Verify original exists
    client = TestClient(app)
    assert client.get("/global/activity/model").json() == "original"

    # Mock user_api_key_auth dependency
    mock_auth_dict = MagicMock()
    mock_auth_dict.user_role = "admin"
    mock_auth_dict.user_id = "test-user"

    # Install usage reporting override
    with (
        patch("open_llm_proxy.usage_reporting.user_api_key_auth", lambda: mock_auth_dict),
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
    ):
        # Setup mock database response
        mock_db = AsyncMock()
        mock_db.query_raw.return_value = [
            {
                "model": "gpt-4",
                "custom_llm_provider": "openai",
                "date": "2026-07-10T00:00:00Z",
                "api_requests": 1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        ]
        mock_prisma.db = mock_db

        install_usage_reporting(app)

        # Re-initialize client to pick up updated routes list
        client = TestClient(app)
        resp = client.get("/global/activity/model?start_date=2026-07-01&end_date=2026-07-10")
        assert resp.status_code == 200

        data = resp.json()
        assert len(data) == 1
        assert data[0]["model"] == "openai/gpt-4"
        assert data[0]["sum_total_tokens"] == 15


def test_aggregate_does_not_collapse_different_years():
    # Two dates with same %b %d but different years
    db_rows = [
        {
            "model": "gpt-4",
            "custom_llm_provider": "openai",
            "date": "2025-07-10T00:00:00Z",
            "api_requests": 1,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        {
            "model": "gpt-4",
            "custom_llm_provider": "openai",
            "date": "2026-07-10T00:00:00Z",
            "api_requests": 2,
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        },
    ]
    result = aggregate_and_normalize_spend_logs(db_rows)
    assert len(result) == 1
    daily_data = result[0]["daily_data"]
    # Should have two distinct daily data entries because they are from different years!
    assert len(daily_data) == 2
    # Sorted chronologically: 2025, then 2026
    assert daily_data[0]["date"] == "Jul 10"
    assert daily_data[0]["total_tokens"] == 150
    assert daily_data[1]["date"] == "Jul 10"
    assert daily_data[1]["total_tokens"] == 300


def test_install_usage_reporting_exception_leakage_prevention():
    app = FastAPI()
    router = APIRouter()

    @router.get("/global/activity/model")
    def original_endpoint():
        return "original"

    app.include_router(router)

    mock_auth_dict = MagicMock()
    mock_auth_dict.user_role = "admin"
    mock_auth_dict.user_id = "test-user"

    with (
        patch("open_llm_proxy.usage_reporting.user_api_key_auth", lambda: mock_auth_dict),
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("open_llm_proxy.usage_reporting.logging.getLogger") as mock_get_logger,
    ):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # prisma_client throws a raw database exception
        mock_db = AsyncMock()
        mock_db.query_raw.side_effect = RuntimeError("Database crash!")
        mock_prisma.db = mock_db

        install_usage_reporting(app)

        client = TestClient(app)
        resp = client.get("/global/activity/model?start_date=2026-07-01&end_date=2026-07-10")
        assert resp.status_code == 500
        # Check that generic 500 error detail is returned, not the raw Exception message
        data = resp.json()
        assert "Database crash!" not in str(data)
        assert data["detail"]["error"] == "Internal server error"
        # Verify it logged the raw exception
        mock_logger.exception.assert_called_once()
