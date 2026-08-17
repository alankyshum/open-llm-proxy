"""Isolate LiteLLM's in-place Gemini request mutation from shared fallback state.

Root cause of the intermittent Copilot ``Bad Request`` 400s
==========================================================

An opencode fallback chain such as::

    [opencode/deepseek-v4-flash-free,
     opencode/nemotron-3-ultra-free,
     google/gemini-3.5-flash,
     github-copilot/gemini-3.5-flash]

asks LiteLLM's router to try each deployment in turn with the *same* request
kwargs (``messages`` and ``optional_params`` — the latter carries ``tools``).

When the router reaches a ``google/gemini-*`` (or ``vertex_ai``) deployment it
runs the Gemini request transformation, which mutates those shared objects
**in place**:

* ``litellm.../vertex_ai/common_utils.py::_build_vertex_schema`` rewrites every
  tool's JSON schema in place (its docstring literally says "modified in
  place") — unpacking ``$defs``, dropping empty ``properties``/``required``,
  adding ``nullable``. This is what reduces Notion's ``get-self`` schema from
  ``{"type": "object", "additionalProperties": false, "properties": {}}`` to
  ``{"type": "object", "additionalProperties": false}`` (empty ``properties``
  deleted).
* ``_transform_system_message`` pops the system messages out of the shared
  ``messages`` list.

If that Gemini deployment then fails (free-tier quota, etc.) the router falls
back to the next deployment — ``github-copilot/gemini-*`` — but now hands it the
**already-Gemini-mangled** tools/messages. Copilot's endpoint rejects the
mangled schema with a bare-text HTTP 400 ``Bad Request``. Because our custom
provider surfaces that as ``CustomLLMError(status_code=400)`` which LiteLLM maps
to ``APIConnectionError`` (a retriable 5xx-class error), opencode retries and
deterministically re-hits the same corruption — the retry loop that killed the
PM digest job.

Fix
===

Wrap Gemini's request-transformation entrypoints so they receive a
``deepcopy`` of ``messages`` and ``optional_params``. All of Gemini's in-place
mutation then happens on throwaway copies, leaving the router's shared objects
pristine for any subsequent Copilot fallback.

This is installed as a monkeypatch in the same spirit as
``streaming_safety.py``. It is idempotent and defensive: if LiteLLM's internals
change and the target functions are missing, installation is a silent no-op
rather than an import error.

Worth reporting upstream to LiteLLM as a shared-state mutation bug.
"""

from __future__ import annotations

import copy
import logging
from functools import wraps

log = logging.getLogger("open_llm_proxy.gemini_isolation")

_SENTINEL = "_open_llm_proxy_deepcopy_isolated"


def _isolating_wrapper(original):
    """Return a wrapper that deepcopies mutable request kwargs before calling.

    ``messages`` and ``optional_params`` are the two objects Gemini mutates in
    place. Both sync and async transform entrypoints accept them as keyword
    arguments, so isolating them by keyword covers every call site.
    """

    @wraps(original)
    def wrapper(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = copy.deepcopy(kwargs["messages"])
        if "optional_params" in kwargs:
            kwargs["optional_params"] = copy.deepcopy(kwargs["optional_params"])
        return original(*args, **kwargs)

    wrapper._open_llm_proxy_deepcopy_isolated = True  # type: ignore[attr-defined]
    return wrapper


def install_gemini_shared_state_isolation() -> None:
    """Patch Gemini transform entrypoints to stop them mutating shared state.

    Idempotent and defensive against LiteLLM internal changes.
    """
    try:
        from litellm.llms.vertex_ai.gemini import transformation as transform_mod
        from litellm.llms.vertex_ai.gemini import (
            vertex_and_google_ai_studio_gemini as handler_mod,
        )
    except Exception as e:  # pragma: no cover - defensive
        log.warning("gemini isolation: could not import vertex modules: %s", e)
        return

    # The handler module imports these two functions *by value* and calls them
    # as bare names, so the wrapper must be installed in the handler module's
    # namespace. We also patch the defining module so any other importer that
    # resolves the name late (and _transform_request_body's internal callers)
    # get the isolated version too.
    targets = [
        (handler_mod, "sync_transform_request_body"),
        (handler_mod, "async_transform_request_body"),
        (transform_mod, "sync_transform_request_body"),
        (transform_mod, "async_transform_request_body"),
    ]

    patched = []
    for mod, name in targets:
        fn = getattr(mod, name, None)
        if fn is None:
            continue
        if getattr(fn, _SENTINEL, False):
            continue
        setattr(mod, name, _isolating_wrapper(fn))
        patched.append(f"{mod.__name__}.{name}")

    if patched:
        log.info("gemini isolation installed on: %s", ", ".join(patched))
    else:
        log.debug("gemini isolation: nothing to patch (already installed or absent)")
