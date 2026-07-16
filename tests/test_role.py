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
    parse_verdict,
    reviewer_role,
)
from atrium_agents.tabby_llm_agent.agent import STATUS_OK, TabbyLLMAgent
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
