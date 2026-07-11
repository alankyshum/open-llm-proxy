"""Tests for open_llm_proxy.errors.upstream_http_error.

A genuine HTTP 400 must surface as a non-retriable litellm BadRequestError; all
other upstream statuses keep the existing CustomLLMError behaviour.
"""

from litellm.exceptions import BadRequestError
from litellm.llms.custom_llm import CustomLLMError

from open_llm_proxy.errors import upstream_http_error


def test_400_maps_to_bad_request_error():
    err = upstream_http_error(400, "Bad Request")
    assert isinstance(err, BadRequestError)
    assert "Bad Request" in err.message


def test_400_empty_message_defaults():
    err = upstream_http_error(400, "")
    assert isinstance(err, BadRequestError)
    assert err.message


def test_non_400_stays_custom_llm_error():
    for status in (401, 403, 500, 502, 429):
        err = upstream_http_error(status, "boom")
        assert isinstance(err, CustomLLMError)
        assert err.status_code == status
        assert err.message == "boom"


def test_non_400_empty_message_defaults_to_http_status():
    err = upstream_http_error(503, "")
    assert isinstance(err, CustomLLMError)
    assert err.message == "HTTP 503"
