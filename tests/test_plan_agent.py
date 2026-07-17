"""Tests for the PlanAgent — the planner engine with a registrable slug.

GPU/model-free: one ``handle_task`` integration through a ``PlanAgent`` whose
bridge call is scripted (no model, no network), asserting the reply is a
well-formed ``plan_result`` and the model saw the planner system prompt.
Mirrors the reviewer integration in ``test_role.py``.
"""

from __future__ import annotations

import asyncio

from atrium_agents.plan_agent import PlanAgent
from atrium_agents.tabby_llm_agent.agent import STATUS_OK
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


def test_plan_agent_defaults_to_planner_role():
    assert PlanAgent("plan-1", "0.1.0").role.name == "planner"
    assert PlanAgent.slug_for() == "plan_agent"


def test_plan_agent_handle_task_end_to_end():
    agent = PlanAgent("plan-1", "0.1.0")
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
