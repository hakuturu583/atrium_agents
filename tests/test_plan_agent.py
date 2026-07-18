"""Tests for the planner as a *role* on the shared inference engine.

A planner is not a distinct agent — it is a ``TabbyLLMAgent`` handed
``planner_role()`` (exactly as a coder/reviewer is its role). GPU/model-free: one
``handle_task`` integration through a ``TabbyLLMAgent`` whose bridge call is
scripted (no model, no network), asserting the reply is a well-formed
``plan_result`` and the model saw the planner system prompt. Mirrors the reviewer
integration in ``test_role.py``.
"""

from __future__ import annotations

import asyncio

from atrium_agents.role import planner_role
from atrium_agents.tabby_llm_agent.agent import STATUS_OK, TabbyLLMAgent
from atrium.agents.plan_agent_protocol import build_plan_request, parse_plan_result
from atrium.protocol import text_message

_REPLY = (
    "```python\n"
    "from prefect import flow\n"
    "from atrium_dispatch import atrium_dispatch\n\n"
    "@flow\n"
    "def main():\n"
    "    return atrium_dispatch('coder:active', 'write it', {})\n"
    "```\n"
    "```json\n"
    "{}\n"
    "```\n"
)


def test_planner_is_a_role_on_a_plain_engine():
    # No PlanAgent subclass: a planner is the tabby engine carrying planner_role().
    agent = TabbyLLMAgent("planner-1", "0.1.0", role=planner_role())
    assert agent.role.name == "planner"
    assert agent.slug_for() == "tabby_llm_agent"  # the engine's slug, not a planner slug


def test_planner_role_handle_task_end_to_end():
    agent = TabbyLLMAgent("planner-1", "0.1.0", role=planner_role())
    captured: dict = {}

    async def fake_send(request):
        captured["request"] = request
        return text_message(_REPLY, metadata={"status": STATUS_OK})

    agent._infer_with_retry = fake_send  # type: ignore[assignment]

    msg = build_plan_request(
        {"instruction": "build a report"},
        "build a report",
        constraints={"agents": ["coder:active"]},
    )
    reply = asyncio.run(agent.handle_task(msg))

    # The reply is a well-formed plan_result carrying the generated flow.
    parsed = parse_plan_result(reply)
    assert parsed["status"] == "ok"
    assert "def main()" in parsed["flow_source"]

    # The model saw the planner system prompt and the request + roster.
    messages = captured["request"]["messages"]
    assert "planning specialist" in messages[0]["content"]
    assert "coder:active" in messages[-1]["content"]
