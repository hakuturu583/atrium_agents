"""Concrete agent roles — coder and reviewer — over the backend-agnostic base.

The :class:`~atrium_agents.inference_agent.Role` base (prompt profile + I/O
framing) lives on the ``InferenceAgent`` layer so any backend reuses it. This
module holds the *application* roles built on it:

* :func:`coder_role` — the default framing with the ``coder`` profile;
* :class:`ReviewerRole` / :func:`reviewer_role` — reads a ``review_request`` and
  emits a **workboard verdict**, which is what couples it to
  :mod:`atrium.orchestration`; kept out of the base layer for that reason.

A coder and a reviewer are then the *same* inference client fanned into one
model, differing only by the role handed to them — so their concurrent requests
share the backend and ride its continuous batching.
"""

from __future__ import annotations

import json
import re
from typing import Any

from atrium_agents.inference_agent import Role, _merged_data
from atrium_agents.prompt_profiles import (
    coder_profile,
    flow_reviewer_profile,
    planner_profile,
    reviewer_profile,
)
from atrium.agents.plan_agent_protocol import build_plan_result
from atrium.orchestration.protocol import board_update_message
from atrium.orchestration.types import NodeOutcome
from atrium.protocol import Message, get_message_text

__all__ = [
    "ReviewerRole",
    "PlannerRole",
    "coder_role",
    "reviewer_role",
    "planner_role",
    "flow_reviewer_role",
    "parse_verdict",
    "parse_plan_output",
    "build_review_prompt",
    "build_plan_prompt",
]


class ReviewerRole(Role):
    """The reviewer role: judge a deliverable, reply with a workboard verdict.

    Reads a ``review_request`` (task + deliverable) into a review prompt, and
    parses the model's ``VERDICT:`` line into an ``ok``/``error`` outcome returned
    in the workboard protocol — so the review gate (:mod:`atrium.orchestration`)
    reads this reply exactly like any node outcome. It judges only the artifact it
    is handed, never the doer's context, which is what makes the verdict independent.
    """

    def frame_prompt(self, message: Message) -> str:
        data = _merged_data(message)
        instruction = str(data.get("instruction") or get_message_text(message))
        deliverable = str(data.get("deliverable") or "")
        return build_review_prompt(instruction, deliverable)

    def frame_reply(self, text: str, message: Message) -> Message:
        return board_update_message(parse_verdict(text), text=text, request=message)


class PlannerRole(Role):
    """The planner role: read a request, emit a job's ``flow.py`` + params.

    Reads a ``plan_request`` (the human request JSON + planning constraints) into a
    planning prompt, and parses the model's two fenced blocks into a ``plan_result``
    the control plane reads exactly like any other contract reply. Kept out of the
    base layer (like :class:`ReviewerRole`) because it couples to the core plan
    contract (:mod:`atrium.agents.plan_agent_protocol`). It **fails closed**: a
    reply without a usable python block becomes an ``error`` plan_result, so a
    half-formed plan can never assemble a runnable job.
    """

    def frame_prompt(self, message: Message) -> str:
        data = _merged_data(message)
        request = data.get("request") or {}
        instruction = str(data.get("instruction") or get_message_text(message))
        constraints = data.get("constraints") or {}
        return build_plan_prompt(instruction, request, constraints)

    def frame_reply(self, text: str, message: Message) -> Message:
        flow_source, params, requirements = parse_plan_output(text)
        if not flow_source:
            return build_plan_result(
                status="error",
                reason="planner produced no python flow block",
                request=message,
            )
        return build_plan_result(
            flow_source, params, requirements=requirements, reason=text.strip()[:2000], request=message
        )


def coder_role() -> Role:
    """A **coder** role carrying the ``coder`` prompt profile (composed locally)."""
    return Role("coder", coder_profile())


def reviewer_role() -> ReviewerRole:
    """A **reviewer** role carrying the ``reviewer`` prompt profile (composed locally)."""
    return ReviewerRole("reviewer", reviewer_profile())


def planner_role() -> PlannerRole:
    """A **planner** role carrying the ``planner`` prompt profile (composed locally)."""
    return PlannerRole("planner", planner_profile())


def flow_reviewer_role() -> ReviewerRole:
    """A **flow reviewer** role: judge a generated flow before it runs.

    Reuses :class:`ReviewerRole`'s framing (review_request in → workboard verdict
    out) since the pre-execution source-review node is dispatched exactly like any
    reviewed node; only the *profile* (safety-first, flow-specific) differs.
    """
    return ReviewerRole("flow_reviewer", flow_reviewer_profile())


# --------------------------------------------------------------------------- #
# Reviewer verdict parsing (host-side of the workboard review gate)           #
# --------------------------------------------------------------------------- #
_APPROVE = {"approve", "approved", "accept", "accepted", "pass", "lgtm", "ok"}
_REJECT = {
    "request-changes", "request changes", "requestchanges", "reject", "rejected",
    "fail", "failed", "changes", "needs-work", "no",
}
#: Matches a ``VERDICT: <word...>`` line (the last one wins), case-insensitive.
_VERDICT_RE = re.compile(r"^\s*VERDICT\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_verdict(text: str) -> NodeOutcome:
    """Parse a reviewer's free text into an ``ok`` (approve) / ``error`` outcome.

    Reads the last ``VERDICT:`` line and maps it to approve/reject vocab. The full
    text rides along as ``result["review"]`` and, on rejection, as the ``reason``
    (which the runner feeds back to the doer as rework guidance). **Fail-closed**:
    an ambiguous review — no ``VERDICT:`` line or an unrecognized value — is
    treated as *request-changes*, never as approval.
    """
    review = text.strip()
    matches = _VERDICT_RE.findall(text)
    token = _normalize(matches[-1]) if matches else ""

    if token in _APPROVE:
        return NodeOutcome(status="ok", result={"review": review})
    reason = review or "reviewer returned no parseable verdict (failing closed)"
    return NodeOutcome(status="error", reason=reason, result={"review": review})


def _normalize(value: str) -> str:
    """Lowercase/strip a verdict token to its first word or known phrase."""
    v = value.strip().lower().strip(".!*`\"'")
    for phrase in ("request changes", "request-changes", "needs work", "needs-work"):
        if v.startswith(phrase):
            return phrase.replace(" ", "-")
    return v.split()[0] if v.split() else ""


def build_review_prompt(instruction: str, deliverable: str) -> str:
    """Compose the reviewer *user* prompt from the task and the deliverable.

    The reviewer's *system* prompt (identity, the VERDICT contract) comes from its
    ``reviewer`` profile; this is only the per-turn material, with the verdict line
    re-stated so parsing is robust across backends.
    """
    return (
        f"## Task\n{instruction.strip() or '(no task description provided)'}\n\n"
        f"## Deliverable to review\n{deliverable.strip() or '(empty deliverable)'}\n\n"
        "Evaluate the deliverable strictly against the task. Finish with a single "
        "line: `VERDICT: approve` or `VERDICT: request-changes`."
    )


# --------------------------------------------------------------------------- #
# Planner output parsing (host-side of the plan contract)                     #
# --------------------------------------------------------------------------- #
#: A fenced code block: ```<lang>\n<body>\n``` — lang is captured, body non-greedy.
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


def parse_plan_output(text: str) -> "tuple[str, dict[str, Any], list[str]]":
    """Parse the planner's fenced blocks into ``(flow_source, params, requirements)``.

    Extracts the first ```python block as the flow source and the first ```json
    block as the params object. **Fails closed**: a missing/empty python block
    yields an empty ``flow_source`` (which the caller turns into an error result),
    and unparseable/ non-object JSON params degrade to ``{}`` rather than raising.
    ``requirements`` is read from an optional ``requirements`` array in the params.
    """
    flow_source = ""
    params: dict[str, Any] = {}
    for lang, body in _FENCE_RE.findall(text):
        low = lang.lower()
        if not flow_source and low in ("python", "py"):
            flow_source = body.strip("\n")
        elif not params and low == "json":
            try:
                loaded = json.loads(body)
                if isinstance(loaded, dict):
                    params = loaded
            except (ValueError, TypeError):
                params = {}
    requirements = [str(r) for r in params.pop("requirements", []) if isinstance(r, str)] if isinstance(
        params.get("requirements"), list
    ) else []
    return flow_source, params, requirements


def build_plan_prompt(instruction: str, request: dict[str, Any], constraints: dict[str, Any]) -> str:
    """Compose the planner *user* prompt from the request and its constraints.

    The planner's *system* prompt (identity, the fenced-block output contract)
    comes from its ``planner`` profile; this is only the per-turn material — the
    request to plan, plus the roster of dispatchable subagents and any caps — with
    the output contract restated so parsing is robust across backends.
    """
    roster = constraints.get("agents") or constraints.get("subagents") or []
    roster_text = "\n".join(f"- {a}" for a in roster) if roster else "(none provided)"
    request_text = json.dumps(request, ensure_ascii=False, indent=2) if request else "(empty)"
    return (
        f"## Request to plan\n{instruction.strip() or '(no instruction)'}\n\n"
        f"## Request JSON\n{request_text}\n\n"
        f"## Dispatchable subagents (assign each task to one of these)\n{roster_text}\n\n"
        "Produce the job now: a ```python `flow.py` (single `@flow def main`) that "
        "dispatches to the subagents above via `atrium_dispatch`, and a ```json "
        "`params` object."
    )
