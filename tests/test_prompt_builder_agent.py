"""Tests for :class:`PromptBuilderAgent` — the A2A prompt-assembly service.

Pure / GPU-free: they drive ``handle_task`` directly (no sandbox, no network) to
exercise the ``build`` / ``list_profiles`` contract, error handling and YAML
construction.
"""

from __future__ import annotations

import asyncio
import textwrap

from atrium_agents.prompt_builder_agent import (
    KIND_BUILD,
    KIND_LIST,
    STATUS_ERROR,
    STATUS_OK,
    PromptBuilderAgent,
)
from atrium_agents.prompt_memory import PromptLayer, PromptMemory
from atrium.protocol import (
    Role,
    data_message,
    get_message_data,
    get_message_text,
    metadata_dict,
)

SAMPLE_TOOL = {
    "type": "function",
    "function": {"name": "read_file", "description": "Read a file", "parameters": {}},
}


def _handle(agent: PromptBuilderAgent, payload: dict):
    msg = data_message(payload, role=Role.ROLE_USER)
    return asyncio.run(agent.handle_task(msg))


def test_build_returns_composed_profile_with_tools():
    agent = PromptBuilderAgent("pb-1")
    reply = _handle(agent, {"type": KIND_BUILD, "profile": "coder", "tools": [SAMPLE_TOOL]})
    meta = metadata_dict(reply)
    assert meta["status"] == STATUS_OK
    assert meta["profile"] == "coder"
    text = get_message_text(reply)
    assert "coding agent" in text
    assert "<tools>" in text and "read_file" in text


def test_reviewer_and_coder_are_distinct_profiles():
    agent = PromptBuilderAgent("pb-1")
    coder = get_message_text(_handle(agent, {"type": KIND_BUILD, "profile": "coder"}))
    reviewer = get_message_text(_handle(agent, {"type": KIND_BUILD, "profile": "reviewer"}))
    assert coder != reviewer
    assert "did NOT write" in reviewer
    assert "focused coding agent" not in reviewer


def test_build_defaults_to_the_default_profile():
    agent = PromptBuilderAgent("pb-1", default_profile="reviewer")
    reply = _handle(agent, {"type": KIND_BUILD})  # no profile named
    assert metadata_dict(reply)["profile"] == "reviewer"
    assert "code reviewer" in get_message_text(reply)


def test_context_fills_volatile_layers():
    mem = PromptMemory(order=("objective",))
    mem.record(PromptLayer("objective", template="Task: {objective}"))
    agent = PromptBuilderAgent("pb-1", profiles={"custom": mem})
    reply = _handle(
        agent,
        {"type": KIND_BUILD, "profile": "custom", "context": {"objective": "ship it"}},
    )
    assert "Task: ship it" in get_message_text(reply)


def test_include_exclude_filter_layers():
    agent = PromptBuilderAgent("pb-1")
    reply = _handle(agent, {"type": KIND_BUILD, "profile": "reviewer", "exclude": ["verdict"]})
    assert "VERDICT" not in get_message_text(reply)


def test_unknown_profile_yields_error_status():
    agent = PromptBuilderAgent("pb-1")
    reply = _handle(agent, {"type": KIND_BUILD, "profile": "does-not-exist"})
    assert metadata_dict(reply)["status"] == STATUS_ERROR
    assert "unknown profile" in get_message_text(reply)


def test_list_profiles_returns_registered_names():
    agent = PromptBuilderAgent("pb-1")
    reply = _handle(agent, {"type": KIND_LIST})
    assert metadata_dict(reply)["status"] == STATUS_OK
    assert get_message_data(reply) == [{"profiles": ["coder", "reviewer"]}]


def test_unknown_kind_yields_error_status():
    agent = PromptBuilderAgent("pb-1")
    reply = _handle(agent, {"type": "frobnicate"})
    assert metadata_dict(reply)["status"] == STATUS_ERROR


def test_register_profile_extends_the_registry():
    agent = PromptBuilderAgent("pb-1")
    agent.register_profile("tester", PromptMemory().record(PromptLayer("identity", content="tester")))
    assert "tester" in agent.profile_names()
    reply = _handle(agent, {"type": KIND_BUILD, "profile": "tester"})
    assert get_message_text(reply) == "tester"


def test_from_yaml_merges_over_builtins(tmp_path):
    doc = textwrap.dedent(
        """
        default_profile: coder
        profiles:
          coder:
            order: [identity]
            layers:
              identity: {content: "custom coder identity"}
          docs:
            order: [identity]
            layers:
              identity: {content: "documentation writer"}
        """
    )
    path = tmp_path / "profiles.yaml"
    path.write_text(doc, encoding="utf-8")

    agent = PromptBuilderAgent.from_yaml(str(path), "pb-1")
    # Built-in reviewer survives; coder is overridden; docs is added.
    assert agent.profile_names() == ["coder", "docs", "reviewer"]
    assert "custom coder identity" in get_message_text(
        _handle(agent, {"type": KIND_BUILD, "profile": "coder"})
    )
    assert "did NOT write" in get_message_text(
        _handle(agent, {"type": KIND_BUILD, "profile": "reviewer"})
    )


def test_from_yaml_without_builtins_serves_only_file_profiles(tmp_path):
    doc = "profiles:\n  only:\n    order: [identity]\n    layers:\n      identity: {content: hi}\n"
    path = tmp_path / "profiles.yaml"
    path.write_text(doc, encoding="utf-8")
    agent = PromptBuilderAgent.from_yaml(str(path), "pb-1", merge_builtins=False)
    assert agent.profile_names() == ["only"]
