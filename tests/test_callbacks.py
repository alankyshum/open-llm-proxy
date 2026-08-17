import pytest

from open_llm_proxy.callbacks import (
    GeminiThinkingBudgetCallback,
    TaskToolEnumInjectionCallback,
)


@pytest.mark.anyio
async def test_gemini_thinking_budget_callback():
    callback = GeminiThinkingBudgetCallback()

    # Case 1: Gemini model with small max_tokens -> should raise to 1024
    data = {"model": "google/gemini-2.5-flash", "max_tokens": 100}
    res = await callback.async_pre_call_hook(None, None, data, "completion")
    assert res is not None
    assert res["max_tokens"] == 1024

    # Case 2: Gemini model with large max_tokens -> should not modify
    data_large = {"model": "google/gemini-2.5-flash", "max_tokens": 2048}
    res_large = await callback.async_pre_call_hook(None, None, data_large, "completion")
    assert res_large is None
    assert data_large["max_tokens"] == 2048  # unchanged

    # Case 3: Non-Gemini model with small max_tokens -> should not modify
    data_non = {"model": "claude-3-5-sonnet", "max_tokens": 100}
    res_non = await callback.async_pre_call_hook(None, None, data_non, "completion")
    assert res_non is None
    assert data_non["max_tokens"] == 100  # unchanged

    # Case 4: absent max_tokens -> should set to floor
    data_absent = {"model": "gemini-2.5-pro"}
    res_absent = await callback.async_pre_call_hook(None, None, data_absent, "completion")
    assert res_absent is not None
    assert res_absent["max_tokens"] == 1024


@pytest.mark.anyio
async def test_deepseek_thinking_budget_callback():
    callback = GeminiThinkingBudgetCallback()

    # deepseek-v4-flash-free with small max_tokens -> raise to 4096
    data = {"model": "opencode/deepseek-v4-flash-free", "max_tokens": 64}
    res = await callback.async_pre_call_hook(None, None, data, "completion")
    assert res is not None
    assert res["max_tokens"] == 4096

    # Already above floor -> no change
    data_large = {"model": "opencode/deepseek-v4-flash-free", "max_tokens": 8192}
    res_large = await callback.async_pre_call_hook(None, None, data_large, "completion")
    assert res_large is None
    assert data_large["max_tokens"] == 8192  # unchanged

    # openai/ prefix variant
    data_openai = {"model": "openai/deepseek-v4-flash-free", "max_tokens": 100}
    res_openai = await callback.async_pre_call_hook(None, None, data_openai, "completion")
    assert res_openai is not None
    assert res_openai["max_tokens"] == 4096

    # absent max_tokens -> set to floor
    data_absent = {"model": "deepseek-v4-flash-free"}
    res_absent = await callback.async_pre_call_hook(None, None, data_absent, "completion")
    assert res_absent is not None
    assert res_absent["max_tokens"] == 4096


@pytest.mark.anyio
async def test_nemotron_thinking_budget_callback():
    callback = GeminiThinkingBudgetCallback()

    # nemotron-3-ultra-free with small max_tokens -> raise to 4096
    data = {"model": "opencode/nemotron-3-ultra-free", "max_tokens": 64}
    res = await callback.async_pre_call_hook(None, None, data, "completion")
    assert res is not None
    assert res["max_tokens"] == 4096

    # Already above floor -> no change
    data_large = {"model": "opencode/nemotron-3-ultra-free", "max_tokens": 8192}
    res_large = await callback.async_pre_call_hook(None, None, data_large, "completion")
    assert res_large is None
    assert data_large["max_tokens"] == 8192  # unchanged

    # absent max_tokens -> set to floor
    data_absent = {"model": "nemotron-3-ultra-free"}
    res_absent = await callback.async_pre_call_hook(None, None, data_absent, "completion")
    assert res_absent is not None
    assert res_absent["max_tokens"] == 4096


@pytest.mark.anyio
async def test_nemotron_exact_thinking_budget_callback():
    callback = GeminiThinkingBudgetCallback()

    # exact model openrouter/nvidia/nemotron-3-ultra-550b-a55b:free with small max_tokens -> raise to 4096
    data = {"model": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", "max_tokens": 64}
    res = await callback.async_pre_call_hook(None, None, data, "completion")
    assert res is not None
    assert res["max_tokens"] == 4096

    # absent max_tokens -> set to floor
    data_absent = {"model": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"}
    res_absent = await callback.async_pre_call_hook(None, None, data_absent, "completion")
    assert res_absent is not None
    assert res_absent["max_tokens"] == 4096


@pytest.mark.anyio
async def test_gpt5_5_thinking_budget_callback():
    callback = GeminiThinkingBudgetCallback()

    # small max_tokens -> raise to 4096
    data = {"model": "github-copilot/gpt-5.5", "max_tokens": 64}
    res = await callback.async_pre_call_hook(None, None, data, "completion")
    assert res is not None
    assert res["max_tokens"] == 4096

    # Already above floor -> no change
    data_large = {"model": "github-copilot/gpt-5.5", "max_tokens": 8192}
    res_large = await callback.async_pre_call_hook(None, None, data_large, "completion")
    assert res_large is None
    assert data_large["max_tokens"] == 8192  # unchanged

    # absent max_tokens -> set to floor
    data_absent = {"model": "gpt-5.5"}
    res_absent = await callback.async_pre_call_hook(None, None, data_absent, "completion")
    assert res_absent is not None
    assert res_absent["max_tokens"] == 4096


@pytest.mark.anyio
async def test_gpt5_mini_thinking_budget_callback():
    callback = GeminiThinkingBudgetCallback()

    data = {"model": "github-copilot/gpt-5-mini", "max_tokens": 16}
    res = await callback.async_pre_call_hook(None, None, data, "completion")
    assert res is not None
    assert res["max_tokens"] == 4096

    data_large = {"model": "github-copilot/gpt-5-mini", "max_tokens": 8192}
    res_large = await callback.async_pre_call_hook(None, None, data_large, "completion")
    assert res_large is None
    assert data_large["max_tokens"] == 8192

    data_absent = {"model": "gpt-5-mini"}
    res_absent = await callback.async_pre_call_hook(None, None, data_absent, "completion")
    assert res_absent is not None
    assert res_absent["max_tokens"] == 4096


@pytest.mark.anyio
async def test_big_pickle_thinking_budget_callback():
    callback = GeminiThinkingBudgetCallback()

    # small max_tokens -> raise to 4096
    data = {"model": "opencode/big-pickle", "max_tokens": 64}
    res = await callback.async_pre_call_hook(None, None, data, "completion")
    assert res is not None
    assert res["max_tokens"] == 4096

    # Already above floor -> no change
    data_large = {"model": "opencode/big-pickle", "max_tokens": 8192}
    res_large = await callback.async_pre_call_hook(None, None, data_large, "completion")
    assert res_large is None
    assert data_large["max_tokens"] == 8192  # unchanged

    # absent max_tokens -> set to floor
    data_absent = {"model": "big-pickle"}
    res_absent = await callback.async_pre_call_hook(None, None, data_absent, "completion")
    assert res_absent is not None
    assert res_absent["max_tokens"] == 4096


@pytest.mark.anyio
async def test_glm5_2_thinking_budget_callback():
    callback = GeminiThinkingBudgetCallback()

    data = {"model": "openrouter/z-ai/glm-5.2", "max_tokens": 16}
    res = await callback.async_pre_call_hook(None, None, data, "completion")
    assert res is not None
    assert res["max_tokens"] == 4096

    data_large = {"model": "openrouter/z-ai/glm-5.2", "max_tokens": 8192}
    res_large = await callback.async_pre_call_hook(None, None, data_large, "completion")
    assert res_large is None
    assert data_large["max_tokens"] == 8192

    data_absent = {"model": "z-ai/glm-5.2"}
    res_absent = await callback.async_pre_call_hook(None, None, data_absent, "completion")
    assert res_absent is not None
    assert res_absent["max_tokens"] == 4096


@pytest.mark.anyio
async def test_kimi_k2_7_thinking_budget_callback():
    callback = GeminiThinkingBudgetCallback()

    data = {"model": "openrouter/moonshotai/kimi-k2.7-code", "max_tokens": 16}
    res = await callback.async_pre_call_hook(None, None, data, "completion")
    assert res is not None
    assert res["max_tokens"] == 4096

    data_large = {"model": "openrouter/moonshotai/kimi-k2.7-code", "max_tokens": 8192}
    res_large = await callback.async_pre_call_hook(None, None, data_large, "completion")
    assert res_large is None
    assert data_large["max_tokens"] == 8192

    data_absent = {"model": "moonshotai/kimi-k2.7-code"}
    res_absent = await callback.async_pre_call_hook(None, None, data_absent, "completion")
    assert res_absent is not None
    assert res_absent["max_tokens"] == 4096


@pytest.mark.anyio
async def test_qwen3_7_plus_thinking_budget_callback():
    callback = GeminiThinkingBudgetCallback()

    data = {"model": "openrouter/qwen/qwen3.7-plus", "max_tokens": 16}
    res = await callback.async_pre_call_hook(None, None, data, "completion")
    assert res is not None
    assert res["max_tokens"] == 4096

    data_large = {"model": "openrouter/qwen/qwen3.7-plus", "max_tokens": 8192}
    res_large = await callback.async_pre_call_hook(None, None, data_large, "completion")
    assert res_large is None
    assert data_large["max_tokens"] == 8192


@pytest.mark.anyio
async def test_task_tool_enum_injection_callback():
    # disabled by default
    callback_disabled = TaskToolEnumInjectionCallback(enabled=False)
    data = {
        "model": "gpt-4o",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "task",
                    "description": "Available agent types:\n  - lead: orchestrator\n  - code: coder\n",
                    "parameters": {
                        "type": "object",
                        "properties": {"subagent_type": {"type": "string"}},
                    },
                },
            }
        ],
    }
    res = await callback_disabled.async_pre_call_hook(None, None, data, "completion")
    assert res is None

    # enabled
    callback_enabled = TaskToolEnumInjectionCallback(enabled=True)
    data2 = {**data}  # shallow copy so orig stays clean
    res_enabled = await callback_enabled.async_pre_call_hook(None, None, data2, "completion")
    assert res_enabled is not None
    tool = res_enabled["tools"][0]
    subagent_prop = tool["function"]["parameters"]["properties"]["subagent_type"]
    assert "enum" in subagent_prop
    assert subagent_prop["enum"] == ["lead", "code"]
