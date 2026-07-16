"""Wiring tests: an inference agent's system prompt is composed from its ``role``'s
profile, and an explicit ``system=`` always overrides it.

These construct a ``TabbyLLMAgent`` (no sandbox is started) and capture the
outbound request instead of sending it, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio

from atrium_agents.role import coder_role, reviewer_role
from atrium_agents.tabby_llm_agent.agent import TabbyLLMAgent
from atrium.protocol import Role as A2ARole, text_message

SAMPLE_TOOL = {
    "type": "function",
    "function": {"name": "read_file", "description": "Read a file", "parameters": {}},
}


def _agent_with_capture(role=None):
    """Build an agent whose ``_infer_with_retry`` records the request instead of
    sending it, returning a canned OK reply."""
    agent = TabbyLLMAgent("agent-test", "0.1.0", role=role)
    captured: dict = {}

    async def fake_send(request):
        captured["request"] = request
        return text_message("OK", role=A2ARole.ROLE_AGENT, metadata={"status": "ok"})

    agent._infer_with_retry = fake_send  # type: ignore[assignment]
    return agent, captured


def test_infer_injects_role_prompt_composed_locally():
    agent, captured = _agent_with_capture(coder_role())
    asyncio.run(agent.infer("hi", tools=[SAMPLE_TOOL]))

    system = captured["request"]["messages"][0]
    assert system["role"] == "system"
    assert "coding agent" in system["content"]
    assert "<tools>" in system["content"]  # tools rendered into the layer
    assert captured["request"]["messages"][-1] == {"role": "user", "content": "hi"}


def test_reviewer_role_prompt_differs_from_coder():
    agent, captured = _agent_with_capture(reviewer_role())
    asyncio.run(agent.infer("evaluate this"))

    system = captured["request"]["messages"][0]["content"]
    assert "did NOT write" in system
    assert "focused coding agent" not in system


def test_explicit_system_overrides_role():
    agent, captured = _agent_with_capture(coder_role())
    asyncio.run(agent.infer("hi", system="explicit override"))
    assert captured["request"]["messages"][0] == {"role": "system", "content": "explicit override"}


def test_no_role_yields_no_system_message():
    agent, captured = _agent_with_capture()  # default role, no profile
    asyncio.run(agent.infer("hi"))
    assert all(m["role"] != "system" for m in captured["request"]["messages"])


def test_chat_prepends_system_only_when_absent():
    agent, captured = _agent_with_capture(coder_role())

    asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))
    assert captured["request"]["messages"][0]["role"] == "system"
    assert "coding agent" in captured["request"]["messages"][0]["content"]

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
