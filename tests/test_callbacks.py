import pytest
from open_llm_proxy.callbacks import (
    GeminiThinkingBudgetCallback,
    TaskToolEnumInjectionCallback,
)

@pytest.mark.anyio
async def test_gemini_thinking_budget_callback():
    callback = GeminiThinkingBudgetCallback()

    # Case 1: Gemini model with small max_tokens -> should raise to 1024
    kwargs = {"max_tokens": 100}
    res = await callback.async_pre_request_hook("google/gemini-2.5-flash", [], kwargs)
    assert res is not None
    assert res["max_tokens"] == 1024

    # Case 2: Gemini model with large max_tokens -> should not modify
    kwargs_large = {"max_tokens": 2048}
    res_large = await callback.async_pre_request_hook("google/gemini-2.5-flash", [], kwargs_large)
    assert res_large is None

    # Case 3: Non-Gemini model with small max_tokens -> should not modify
    kwargs_non_gemini = {"max_tokens": 100}
    res_non_gemini = await callback.async_pre_request_hook("claude-3-5-sonnet", [], kwargs_non_gemini)
    assert res_non_gemini is None

@pytest.mark.anyio
async def test_task_tool_enum_injection_callback():
    # disabled by default
    callback_disabled = TaskToolEnumInjectionCallback(enabled=False)
    kwargs = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "task",
                    "description": "Available agent types:\n  - lead: orchestrator\n  - code: coder\n",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subagent_type": {"type": "string"}
                        }
                    }
                }
            }
        ]
    }
    res = await callback_disabled.async_pre_request_hook("gpt-4o", [], kwargs)
    assert res is None

    # enabled
    callback_enabled = TaskToolEnumInjectionCallback(enabled=True)
    res_enabled = await callback_enabled.async_pre_request_hook("gpt-4o", [], kwargs)
    assert res_enabled is not None
    tool = res_enabled["tools"][0]
    subagent_prop = tool["function"]["parameters"]["properties"]["subagent_type"]
    assert "enum" in subagent_prop
    assert subagent_prop["enum"] == ["lead", "code"]
