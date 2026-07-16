"""``SlackInterfaceAgent`` — Slack ingress/egress over :class:`InterfaceAgent`.

The concrete Slack boundary: normalize a Slack Events API message / slash command
into a :class:`Turn`, and render acks/updates as Slack ``mrkdwn``. Everything
channel-agnostic — the single forward to the control plane, ``context_id``
derivation, session/routing — is inherited. The outbound transport
(``chat.postMessage``) is an injected ``poster`` so the agent stays testable and
the deployment wires the actual Slack Web API call.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional

from atrium_agents.interface_agent.agent import InterfaceAgent, Session, Turn
from atrium.core.errors import AgentError
from atrium.core.types import SandboxConfig, VersionTag
from atrium.protocol import Message, Role, text_message

logger = logging.getLogger("atrium_agents.interface.slack")

__all__ = ["SlackInterfaceAgent", "SlackSession", "SlackPoster"]

#: Outbound transport: post ``text`` into the thread named by ``coords``.
SlackPoster = Callable[[Mapping[str, Any], str], Awaitable[None]]

#: Leading Slack bot mention, e.g. ``<@U0123ABCD> build a foo`` -> ``build a foo``.
_MENTION_RE = re.compile(r"^\s*<@[A-Z0-9]+>\s*")


@dataclass
class SlackSession(Session):
    """Slack presentation state on top of the shared :class:`Session`."""

    #: Message ts to edit as a job progresses (reserved for the deployment).
    working_msg_ts: Optional[str] = None
    #: Slack redelivers events on retry; de-duplicate on their ids.
    seen_event_ids: set[str] = field(default_factory=set)


class SlackInterfaceAgent(InterfaceAgent):
    """A Slack I/O boundary that forwards turns to the control plane."""

    AGENT_SLUG = "slack_interface_agent"
    SOURCE = "slack"

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        control_plane: Any,
        poster: Optional[SlackPoster] = None,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> None:
        from atrium_agents.interface_agent import __version__

        super().__init__(
            agent_id, version or __version__, control_plane=control_plane, sandbox_config=sandbox_config
        )
        self._poster = poster

    # ------------------------------------------------------------------ #
    # Session (adds Slack presentation state)                            #
    # ------------------------------------------------------------------ #
    def new_session(self, context_id: str) -> Session:
        return SlackSession(context_id=context_id)

    # ------------------------------------------------------------------ #
    # Ingress: Slack envelope -> Turn                                    #
    # ------------------------------------------------------------------ #
    def parse_turn(self, message: Message) -> Turn:
        return self.normalize_slack(self.merge_data_parts(message))

    @staticmethod
    def normalize_slack(payload: Mapping[str, Any]) -> Turn:
        """Normalize a raw Slack payload into a :class:`Turn` (pure; testable)."""
        event = payload.get("event") if isinstance(payload.get("event"), Mapping) else None
        if event is not None:  # Events API (message / app_mention)
            text = str(event.get("text", ""))
            user, channel = event.get("user"), event.get("channel")
            thread = event.get("thread_ts") or event.get("ts")
            event_id = payload.get("event_id")
        elif "command" in payload:  # slash command
            text = str(payload.get("text", ""))
            user, channel = payload.get("user_id"), payload.get("channel_id")
            thread = payload.get("thread_ts") or payload.get("trigger_id")
            event_id = payload.get("trigger_id")
        else:  # already-normalized / direct invocation
            text = str(payload.get("instruction") or payload.get("text") or "")
            user, channel = payload.get("user"), payload.get("channel")
            thread = payload.get("thread")
            event_id = payload.get("event_id")

        instruction = _MENTION_RE.sub("", text).strip()
        if not instruction:
            raise AgentError("Slack turn carried no instruction text")
        return Turn(
            source="slack",
            instruction=instruction,
            user=user,
            channel=channel,
            thread=thread,
            event_id=event_id,
            raw=dict(payload),
        )

    def is_duplicate(self, turn: Turn, sess: Session) -> bool:
        """Drop a Slack retry of an event we already accepted in this thread."""
        if not turn.event_id or not isinstance(sess, SlackSession):
            return False
        if turn.event_id in sess.seen_event_ids:
            return True
        sess.seen_event_ids.add(turn.event_id)
        return False

    # ------------------------------------------------------------------ #
    # Egress: Slack-flavored replies                                     #
    # ------------------------------------------------------------------ #
    def render_ack(self, turn: Turn, ack: Mapping[str, Any]) -> Message:
        status = ack.get("status")
        if status == "ok":
            text = f":hourglass_flowing_sand: Got it — working on it (job `{ack.get('job_id')}`)."
        elif status == "duplicate":
            text = ""  # already accepted; nothing new to say
        else:
            text = ":x: Couldn't queue that."
        return text_message(text, role=Role.ROLE_AGENT, metadata={"kind": "slack", "status": status or "error"})

    def render_update(self, update: Mapping[str, Any]) -> str:
        result = update.get("result") or {}
        if update.get("status") == "ok":
            digest = result.get("digest")
            return ":white_check_mark: Done" + (f" (`{digest}`)" if digest else "") + "."
        return f":x: Job failed: {result.get('reason', 'unknown error')}"

    async def deliver(self, coords: Mapping[str, Any], text: str) -> None:
        """Post into the Slack thread via the injected transport (no-op if unwired)."""
        if self._poster is None:
            logger.info("no Slack poster wired; would post to %s: %s", dict(coords), text)
            return
        await self._poster(coords, text)
