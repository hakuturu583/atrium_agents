"""Wiring tests: an inference agent's system prompt comes from its injected
``PromptSource`` (local or A2A), and an explicit ``system=`` always overrides it.

These construct a ``TabbyLLMAgent`` (no sandbox is started) and capture the
outbound request instead of sending it, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio

from atrium_agents.prompt_builder_agent import PromptBuilderAgent
from atrium_agents.prompt_source import LocalPromptSource, RemotePromptSource
from atrium_agents.tabby_llm_agent.agent import TabbyLLMAgent
from atrium.protocol import Role, text_message

SAMPLE_TOOL = {
    "type": "function",
    "function": {"name": "read_file", "description": "Read a file", "parameters": {}},
}


def _agent_with_capture(prompt_source=None):
    """Build an agent whose ``_infer_with_retry`` records the request instead of
    sending it, returning a canned OK reply."""
    agent = TabbyLLMAgent("agent-test", "0.1.0", prompt_source=prompt_source)
    captured: dict = {}

    async def fake_send(request):
        captured["request"] = request
        return text_message("OK", role=Role.ROLE_AGENT, metadata={"status": "ok"})

    agent._infer_with_retry = fake_send  # type: ignore[assignment]
    return agent, captured


# --------------------------------------------------------------------------- #
# Local source (co-located builder, no network).                              #
# --------------------------------------------------------------------------- #
def test_infer_injects_role_prompt_from_local_source():
    builder = PromptBuilderAgent("pb-1")
    agent, captured = _agent_with_capture(LocalPromptSource(builder, "coder"))

    asyncio.run(agent.infer("hi", tools=[SAMPLE_TOOL]))

    system = captured["request"]["messages"][0]
    assert system["role"] == "system"
    assert "coding agent" in system["content"]
    assert "<tools>" in system["content"]  # tools rendered into the layer
    assert captured["request"]["messages"][-1] == {"role": "user", "content": "hi"}


def test_reviewer_source_differs_from_coder():
    builder = PromptBuilderAgent("pb-1")
    agent, captured = _agent_with_capture(LocalPromptSource(builder, "reviewer"))

    asyncio.run(agent.infer("evaluate this"))

    system = captured["request"]["messages"][0]["content"]
    assert "did NOT write" in system
    assert "focused coding agent" not in system


def test_explicit_system_overrides_source():
    builder = PromptBuilderAgent("pb-1")
    agent, captured = _agent_with_capture(LocalPromptSource(builder, "coder"))

    asyncio.run(agent.infer("hi", system="explicit override"))

    assert captured["request"]["messages"][0] == {"role": "system", "content": "explicit override"}


def test_no_source_yields_no_system_message():
    agent, captured = _agent_with_capture()  # no prompt_source
    asyncio.run(agent.infer("hi"))
    assert all(m["role"] != "system" for m in captured["request"]["messages"])


def test_chat_prepends_system_only_when_absent():
    builder = PromptBuilderAgent("pb-1")
    agent, captured = _agent_with_capture(LocalPromptSource(builder, "coder"))

    # No system turn -> role prompt is prepended.
    asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))
    assert captured["request"]["messages"][0]["role"] == "system"
    assert "coding agent" in captured["request"]["messages"][0]["content"]

    # Existing system turn -> left untouched.
    asyncio.run(
        agent.chat(
            [
                {"role": "system", "content": "caller system"},
                {"role": "user", "content": "hi"},
            ]
        )
    )
    systems = [m for m in captured["request"]["messages"] if m["role"] == "system"]
    assert systems == [{"role": "system", "content": "caller system"}]


# --------------------------------------------------------------------------- #
# Remote source: the prompt is fetched from a PromptBuilderAgent over A2A.     #
# --------------------------------------------------------------------------- #
def test_infer_sources_reviewer_prompt_over_a2a(monkeypatch):
    builder = PromptBuilderAgent("pb-1")

    async def fake_send_message(target, message):
        assert target == "http://pb.local"
        return await builder.handle_task(message)

    # Route the source's A2A send to the builder in-process.
    monkeypatch.setattr("atrium_agents.prompt_source.send_message", fake_send_message)

    source = RemotePromptSource("http://pb.local", "reviewer")
    agent, captured = _agent_with_capture(source)

    asyncio.run(agent.infer("evaluate this patch"))

    system = captured["request"]["messages"][0]
    assert system["role"] == "system"
    assert "did NOT write" in system["content"]
