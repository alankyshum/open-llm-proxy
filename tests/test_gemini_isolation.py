"""Regression tests for the Gemini shared-state isolation monkeypatch.

Root cause covered: LiteLLM's Gemini request transform mutates the shared
``optional_params``/``messages`` objects in place (``_build_vertex_schema``
deletes empty ``properties`` from tool schemas). When an opencode fallback chain
tries a ``google/gemini-*`` deployment before ``github-copilot/*``, that in-place
mutation corrupts the tools handed to the later Copilot fallback, which rejects
them with a bare HTTP 400 ``Bad Request`` and triggers an infinite retry loop.
"""

import copy

import pytest

from open_llm_proxy.gemini_isolation import (
    _isolating_wrapper,
    install_gemini_shared_state_isolation,
)


def test_build_vertex_schema_mutates_in_place():
    """Document LiteLLM's in-place corruption so the fix stays justified."""
    from litellm.llms.vertex_ai.common_utils import _build_vertex_schema

    schema = {"type": "object", "additionalProperties": False, "properties": {}}
    original = copy.deepcopy(schema)
    _build_vertex_schema(parameters=schema)
    # The bug: empty properties (and additionalProperties) are deleted in place,
    # leaving {"type": "object"} which Copilot's gemini endpoint rejects.
    assert schema != original
    assert "properties" not in schema


def test_isolating_wrapper_deepcopies_messages_and_optional_params():
    seen = {}

    def fake_transform(*, messages, optional_params, **kw):
        # Mutate what we were handed, the way Gemini does.
        optional_params["tools"][0]["function"]["parameters"].pop("properties", None)
        messages.append({"role": "system", "content": "popped"})
        seen["messages_id"] = id(messages)
        seen["params_id"] = id(optional_params)
        return "ok"

    wrapped = _isolating_wrapper(fake_transform)
    assert wrapped._open_llm_proxy_deepcopy_isolated is True

    messages = [{"role": "user", "content": "hi"}]
    optional_params = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "t",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    }
    msg_before = copy.deepcopy(messages)
    params_before = copy.deepcopy(optional_params)

    result = wrapped(messages=messages, optional_params=optional_params)

    assert result == "ok"
    # Caller's objects are pristine; the wrapper handed the callee throwaway copies.
    assert messages == msg_before
    assert optional_params == params_before
    assert seen["messages_id"] != id(messages)
    assert seen["params_id"] != id(optional_params)


def test_install_is_idempotent_and_wraps_entrypoints():
    install_gemini_shared_state_isolation()
    install_gemini_shared_state_isolation()  # second call is a no-op

    from litellm.llms.vertex_ai.gemini import (
        vertex_and_google_ai_studio_gemini as handler_mod,
    )

    for name in ("sync_transform_request_body", "async_transform_request_body"):
        fn = getattr(handler_mod, name)
        assert getattr(fn, "_open_llm_proxy_deepcopy_isolated", False) is True
        # Idempotent: not double-wrapped (unwrapping one layer reaches the sentinel-free original).
        assert not getattr(
            getattr(fn, "__wrapped__", None), "_open_llm_proxy_deepcopy_isolated", False
        )


def test_end_to_end_isolation_prevents_tool_corruption():
    """The real transform entrypoint must not corrupt the caller's tools."""
    install_gemini_shared_state_isolation()
    from litellm.llms.vertex_ai.gemini import (
        vertex_and_google_ai_studio_gemini as handler_mod,
    )

    tool = {
        "type": "function",
        "function": {
            "name": "notion_API-get-self",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        },
    }
    optional_params = {"tools": [copy.deepcopy(tool)]}
    before = copy.deepcopy(optional_params)

    try:
        handler_mod.sync_transform_request_body(
            gemini_api_key="x",
            messages=[{"role": "user", "content": "hi"}],
            api_base=None,
            model="gemini-3.5-flash",
            client=None,
            timeout=None,
            extra_headers=None,
            optional_params=optional_params,
            logging_obj=None,
            custom_llm_provider="gemini",
            litellm_params={},
            vertex_project=None,
            vertex_location=None,
            vertex_auth_header=None,
        )
    except Exception:
        # We don't care whether the (heavily-mocked) transform completes; we only
        # assert it did not corrupt the caller's shared tools object.
        pass

    assert optional_params == before, "caller's tools were mutated in place"
