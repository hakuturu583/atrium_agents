"""Tests for agent roles — the coder/reviewer framing over the shared engine.

Pure / GPU-free: role framing (``frame_prompt`` / ``frame_reply``) and verdict
parsing are exercised directly, plus one ``handle_task`` integration through a
``TabbyLLMAgent`` whose bridge call is scripted (no model, no network).
"""

from __future__ import annotations

import asyncio

from atrium_agents.role import (
    ReviewerRole,
    build_review_prompt,
    coder_role,
    flow_reviewer_role,
    parse_plan_output,
    parse_verdict,
    planner_role,
    reviewer_role,
)
from atrium_agents.tabby_llm_agent.agent import STATUS_OK, TabbyLLMAgent
from atrium.agents.plan_agent_protocol import build_plan_request, parse_plan_result
from atrium.orchestration.protocol import build_node_request, extract_board_update
from atrium.orchestration.review import build_review_request
from atrium.orchestration.types import WorkNode
from atrium.protocol import get_message_text, text_message


# --------------------------------------------------------------------------- #
# Verdict parsing                                                              #
# --------------------------------------------------------------------------- #
def test_parse_verdict_approve():
    assert parse_verdict("Looks complete.\nVERDICT: approve").status == "ok"


def test_parse_verdict_request_changes_carries_reason():
    out = parse_verdict("Missing error handling.\nVERDICT: request-changes")
    assert out.status == "error" and "Missing error handling" in out.reason


def test_parse_verdict_synonyms_and_last_wins():
    assert parse_verdict("VERDICT: LGTM").status == "ok"
    assert parse_verdict("VERDICT: reject").status == "error"
    assert parse_verdict("VERDICT: request-changes\nVERDICT: approve").status == "ok"


def test_parse_verdict_fails_closed():
    assert parse_verdict("Looks fine, ship it.").status == "error"  # no VERDICT line
    assert parse_verdict("VERDICT: maybe?").status == "error"       # unknown token


def test_build_review_prompt_states_the_contract():
    p = build_review_prompt("task", "artifact")
    assert "VERDICT: approve" in p and "VERDICT: request-changes" in p


# --------------------------------------------------------------------------- #
# Role framing                                                                 #
# --------------------------------------------------------------------------- #
def test_coder_role_composes_profile_locally():
    system = coder_role().system_prompt()
    assert "coding agent" in system


def test_coder_role_folds_review_feedback_into_prompt():
    node = WorkNode(id="n1", agent="a", instruction="do it", payload={"review_feedback": "rename foo"})
    prompt = coder_role().frame_prompt(build_node_request(node))
    assert "do it" in prompt and "rename foo" in prompt


def test_reviewer_role_frames_prompt_from_review_request():
    node = WorkNode(id="n1", agent="a", instruction="Implement X")
    msg = build_review_request(node, "def x(): ...", {})
    prompt = reviewer_role().frame_prompt(msg)
    assert "Implement X" in prompt and "def x()" in prompt


def test_reviewer_role_frames_reply_as_workboard_verdict():
    role = reviewer_role()
    node = WorkNode(id="n1", agent="a", instruction="t")
    msg = build_review_request(node, "artifact", {})

    approve = role.frame_reply("All good.\nVERDICT: approve", msg)
    assert extract_board_update(approve).outcome.ok

    reject = role.frame_reply("Bug on line 3.\nVERDICT: request-changes", msg)
    outcome = extract_board_update(reject).outcome
    assert not outcome.ok and "Bug on line 3" in outcome.reason


# --------------------------------------------------------------------------- #
# handle_task integration (scripted bridge — no model)                        #
# --------------------------------------------------------------------------- #
def test_reviewer_agent_handle_task_end_to_end():
    agent = TabbyLLMAgent("reviewer-1", "0.1.0", role=reviewer_role())
    captured: dict = {}

    async def fake_send(request):
        captured["request"] = request
        return text_message("Solid.\nVERDICT: approve", metadata={"status": STATUS_OK})

    agent._infer_with_retry = fake_send  # type: ignore[assignment]

    node = WorkNode(id="n1", agent="coder", instruction="Implement X")
    review_msg = build_review_request(node, "def x(): return 1", {})
    reply = asyncio.run(agent.handle_task(review_msg))

    # the verdict is returned in the workboard protocol...
    assert extract_board_update(reply).outcome.ok
    # ...and the model saw the reviewer system prompt + the deliverable.
    messages = captured["request"]["messages"]
    assert "did NOT write" in messages[0]["content"]           # reviewer profile as system
    assert "def x(): return 1" in messages[-1]["content"]      # deliverable in user turn


# --------------------------------------------------------------------------- #
# Planner output parsing                                                       #
# --------------------------------------------------------------------------- #
_PLAN_REPLY = (
    "Here is the plan.\n\n"
    "```python\n"
    "from prefect import flow\n"
    "from atrium_dispatch import atrium_dispatch\n\n"
    "@flow\n"
    "def main():\n"
    "    return atrium_dispatch('coder:active', 'write it', {})\n"
    "```\n\n"
    "```json\n"
    '{"topic": "widgets", "requirements": ["prefect"]}\n'
    "```\n"
)


def test_parse_plan_output_extracts_blocks_and_requirements():
    flow_source, params, requirements = parse_plan_output(_PLAN_REPLY)
    assert "def main()" in flow_source and "atrium_dispatch" in flow_source
    assert params == {"topic": "widgets"}          # requirements popped out
    assert requirements == ["prefect"]


def test_parse_plan_output_fails_closed_without_python_block():
    flow_source, params, requirements = parse_plan_output("no code here, sorry")
    assert flow_source == ""
    assert params == {} and requirements == []


def test_parse_plan_output_tolerates_bad_json_params():
    text = "```python\nx=1\n```\n```json\nnot json\n```"
    flow_source, params, _ = parse_plan_output(text)
    assert flow_source == "x=1" and params == {}


# --------------------------------------------------------------------------- #
# Planner role framing                                                         #
# --------------------------------------------------------------------------- #
def test_planner_role_composes_profile_locally():
    system = planner_role().system_prompt()
    assert "planning specialist" in system
    assert "atrium_dispatch" in system


def test_planner_role_frames_prompt_from_plan_request():
    msg = build_plan_request(
        {"instruction": "build a report"}, "build a report", constraints={"agents": ["coder:active"]}
    )
    prompt = planner_role().frame_prompt(msg)
    assert "build a report" in prompt
    assert "coder:active" in prompt          # roster is presented to the model


def test_planner_role_frames_reply_as_plan_result():
    msg = build_plan_request({"instruction": "x"}, "x")
    reply = planner_role().frame_reply(_PLAN_REPLY, msg)
    parsed = parse_plan_result(reply)
    assert parsed["status"] == "ok"
    assert "def main()" in parsed["flow_source"]
    assert parsed["params"] == {"topic": "widgets"}
    assert parsed["requirements"] == ["prefect"]


def test_planner_role_reply_fails_closed():
    msg = build_plan_request({"instruction": "x"}, "x")
    reply = planner_role().frame_reply("I couldn't do it.", msg)
    parsed = parse_plan_result(reply)
    assert parsed["status"] == "error"
    assert parsed["flow_source"] == ""


# --------------------------------------------------------------------------- #
# Flow reviewer role (pre-execution source review)                            #
# --------------------------------------------------------------------------- #
def test_flow_reviewer_role_reviews_a_flow_and_verdicts():
    role = flow_reviewer_role()
    assert "safety reviewer" in role.system_prompt()
    # Dispatched like any reviewed node: instruction + deliverable in → verdict out.
    node = WorkNode(id="review_source", agent="flow_reviewer", instruction="review it")
    msg = build_review_request(node, "def main():\n    return 1", {})
    approve = role.frame_reply("Only orchestrates.\nVERDICT: approve", msg)
    assert extract_board_update(approve).outcome.ok
    reject = role.frame_reply("Opens a socket.\nVERDICT: request-changes", msg)
    assert not extract_board_update(reject).outcome.ok
