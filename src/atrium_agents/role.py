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

import re

from atrium_agents.inference_agent import Role, _merged_data
from atrium_agents.prompt_profiles import coder_profile, reviewer_profile
from atrium.orchestration.protocol import board_update_message
from atrium.orchestration.types import NodeOutcome
from atrium.protocol import Message, get_message_text

__all__ = [
    "ReviewerRole",
    "coder_role",
    "reviewer_role",
    "parse_verdict",
    "build_review_prompt",
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


def coder_role() -> Role:
    """A **coder** role carrying the ``coder`` prompt profile (composed locally)."""
    return Role("coder", coder_profile())


def reviewer_role() -> ReviewerRole:
    """A **reviewer** role carrying the ``reviewer`` prompt profile (composed locally)."""
    return ReviewerRole("reviewer", reviewer_profile())


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
