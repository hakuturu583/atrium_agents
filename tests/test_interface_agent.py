"""Tests for the InterfaceAgent base and the SlackInterfaceAgent.

No sandbox or network: the outward A2A seam (``send_a2a_message``) is scripted, so
the parse → context_id → single-forward → ack path, Slack normalization, retry
de-duplication, feedback relaying, and job-update delivery are exercised
in-process against a scripted control plane.
"""

from __future__ import annotations

import asyncio

import pytest

from atrium_agents.interface_agent import SlackInterfaceAgent
from atrium_agents.interface_agent.slack import SlackSession
from atrium.agents.control_plane import (
    build_job_update,
    build_submitted_reply,
    parse_submit_request,
)
from atrium.core.errors import AgentError
from atrium.core.types import NetworkMode
from atrium.protocol import data_part, get_message_text, metadata_dict, text_message


def _agent(poster=None):
    return SlackInterfaceAgent("slack-if-1", "0.1.0", control_plane="http://control.local", poster=poster)


def _script_control_plane(agent, reply):
    """Record the forwarded request; return a scripted control-plane reply."""
    sent: list = []

    async def fake_send(target, message):
        sent.append((target, message))
        return reply

    agent.send_a2a_message = fake_send  # type: ignore[assignment]
    return sent


def _slack_event(text="<@U01BOT> write hello world", *, channel="C1", ts="1699.0001", event_id="Ev1", thread=None):
    event = {"text": text, "user": "U9", "channel": channel, "ts": ts}
    if thread:
        event["thread_ts"] = thread
    return text_message(
        "", extra_parts=[data_part({"event": event, "event_id": event_id})]
    )


# --------------------------------------------------------------------------- #
# Slack normalization (pure)                                                   #
# --------------------------------------------------------------------------- #
def test_normalize_strips_mention_and_uses_ts_as_thread():
    turn = SlackInterfaceAgent.normalize_slack(
        {"event": {"text": "<@U01> do it", "user": "U9", "channel": "C1", "ts": "42.1"}}
    )
    assert turn.instruction == "do it"
    assert turn.channel == "C1"
    assert turn.thread == "42.1"  # thread defaults to the root ts


def test_normalize_prefers_thread_ts():
    turn = SlackInterfaceAgent.normalize_slack(
        {"event": {"text": "reply", "channel": "C1", "ts": "99.9", "thread_ts": "42.1"}}
    )
    assert turn.thread == "42.1"


def test_normalize_slash_command():
    turn = SlackInterfaceAgent.normalize_slack(
        {"command": "/build", "text": "make a thing", "user_id": "U5", "channel_id": "C2"}
    )
    assert turn.instruction == "make a thing"
    assert turn.channel == "C2"


def test_normalize_rejects_empty():
    with pytest.raises(AgentError, match="no instruction"):
        SlackInterfaceAgent.normalize_slack({"event": {"text": "<@U01>   "}})


# --------------------------------------------------------------------------- #
# Envelope + context_id                                                        #
# --------------------------------------------------------------------------- #
def test_envelope_is_wan_capable():
    assert _agent().sandbox_config.network is NetworkMode.BRIDGE


def test_context_id_is_derived_from_thread():
    agent = _agent()
    turn = SlackInterfaceAgent.normalize_slack(
        {"event": {"text": "hi", "channel": "C1", "ts": "42.1"}}
    )
    assert agent.context_id_for(turn) == "slack:C1:42.1"


# --------------------------------------------------------------------------- #
# Forward: turn -> single submit to the control plane                          #
# --------------------------------------------------------------------------- #
def test_forwards_turn_to_control_plane_and_acks():
    agent = _agent()
    sent = _script_control_plane(agent, build_submitted_reply("flow-run-7"))
    reply = asyncio.run(agent.dispatch(_slack_event()))

    target, forwarded = sent[0]
    assert target == "http://control.local"
    req = parse_submit_request(forwarded)
    assert req.instruction == "write hello world"
    assert req.agent == ""  # no explicit target ⇒ control plane routes (D5)
    assert req.context_id == "slack:C1:1699.0001"
    assert req.payload["reply_coords"] == {
        "source": "slack", "channel": "C1", "thread": "1699.0001", "user": "U9"
    }
    assert req.feedback_for is None

    # Ack is Slack-flavored and mentions the job id; session cached it.
    assert metadata_dict(reply)["status"] == "ok"
    assert "job `flow-run-7`" in get_message_text(reply)
    assert agent.sessions.get("slack:C1:1699.0001").active_job_id == "flow-run-7"


def test_steering_folds_into_payload():
    class Steered(SlackInterfaceAgent):
        def steering(self, turn):
            return {"lang": "python"}

    agent = Steered("s1", "0.1.0", control_plane="http://c")
    sent = _script_control_plane(agent, build_submitted_reply("j1"))
    asyncio.run(agent.dispatch(_slack_event()))
    assert parse_submit_request(sent[0][1]).payload["steering"] == {"lang": "python"}


# --------------------------------------------------------------------------- #
# Retry de-duplication                                                          #
# --------------------------------------------------------------------------- #
def test_duplicate_event_is_not_forwarded_twice():
    agent = _agent()
    sent = _script_control_plane(agent, build_submitted_reply("j1"))
    asyncio.run(agent.dispatch(_slack_event(event_id="EvDup")))
    asyncio.run(agent.dispatch(_slack_event(event_id="EvDup")))
    assert len(sent) == 1  # the redelivery was dropped


# --------------------------------------------------------------------------- #
# Feedback relay: a reply while a review is pending                             #
# --------------------------------------------------------------------------- #
def test_reply_while_review_pending_relays_feedback():
    agent = _agent()
    sent = _script_control_plane(agent, build_submitted_reply("j2"))
    # Simulate a review awaiting the human on this thread.
    sess = agent.sessions.get_or_create("slack:C1:1699.0001")
    assert isinstance(sess, SlackSession)
    sess.pending_review = "review-token-1"

    asyncio.run(agent.dispatch(_slack_event(text="looks good")))
    assert parse_submit_request(sent[0][1]).feedback_for == "review-token-1"


# --------------------------------------------------------------------------- #
# Egress: a job_update is delivered into the thread                            #
# --------------------------------------------------------------------------- #
def test_job_update_is_delivered_via_poster():
    posted: list = []

    async def poster(coords, text):
        posted.append((dict(coords), text))

    agent = _agent(poster=poster)
    update = build_job_update(
        "j1", status="ok", coords={"channel": "C1", "thread": "42"}, result={"digest": "sha256:abc"}
    )
    asyncio.run(agent.on_job_update(update))

    assert posted[0][0] == {"channel": "C1", "thread": "42"}
    assert ":white_check_mark:" in posted[0][1]
    assert "sha256:abc" in posted[0][1]


def test_job_update_failure_rendered():
    posted: list = []

    async def poster(coords, text):
        posted.append(text)

    agent = _agent(poster=poster)
    update = build_job_update("j1", status="error", coords={}, result={"reason": "boom"})
    asyncio.run(agent.on_job_update(update))
    assert ":x:" in posted[0] and "boom" in posted[0]


def test_deliver_without_poster_is_noop():
    agent = _agent(poster=None)
    update = build_job_update("j1", status="ok", coords={"channel": "C1"}, result={})
    # Must not raise when no transport is wired.
    asyncio.run(agent.on_job_update(update))


# --------------------------------------------------------------------------- #
# 動線2(b) — a review presentation marks the session for feedback relay         #
# --------------------------------------------------------------------------- #
def test_review_update_marks_pending_review_and_presents():
    posted: list = []

    async def poster(coords, text):
        posted.append(text)

    agent = _agent(poster=poster)
    update = build_job_update(
        "j1", status="review", coords={"channel": "C1", "thread": "9"},
        result={"token": "slack:C1:9", "instruction": "build a widget"},
    )
    asyncio.run(agent.on_job_update(update))

    # Session is marked so the human's next reply relays as feedback_for the token.
    assert agent.sessions.get("slack:C1:9").pending_review == "slack:C1:9"
    assert ":eyes:" in posted[0] and "build a widget" in posted[0]
