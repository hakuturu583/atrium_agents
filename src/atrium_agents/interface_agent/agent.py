"""``InterfaceAgent`` — a channel-agnostic human I/O boundary.

A human talks to Atrium through a chat app (Slack, Discord, …). An
``InterfaceAgent`` is the *one* boundary for a given app: it normalizes an
inbound turn, forwards it to a trusted **control plane** over A2A (its only
egress), and delivers results back into the originating thread. It authors no
code, kicks no workboard, and holds no authority — a propose-only edge (which is
why it lives here in the evolvable tier, not the fixed core). See the atrium core
design ``docs/design/interface-agent.md``.

The *shared "when"* lives on this base — turn → ``context_id`` → single forward to
the control plane → deliver the result to the thread — so a new chat app is one
concrete subclass filling the app-specific *how* (``parse_turn`` / ``render_*`` /
``deliver`` / ``SOURCE``). The inheritance chain::

    BaseAgent → InterfaceAgent → SlackInterfaceAgent, …
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from atrium.agents.control_plane import (
    build_submit_request,
    parse_job_update,
    parse_submitted_reply,
)
from atrium.core.base_agent import BaseAgent
from atrium.core.types import SandboxConfig, VersionTag, wan_sandbox_config
from atrium.protocol import Message
from atrium.protocol.a2a_transport import SendTarget

logger = logging.getLogger("atrium_agents.interface")

__all__ = ["InterfaceAgent", "Turn", "Session", "SessionStore"]

#: First version minted for an interface agent with no ledger history.
DEFAULT_INITIAL_VERSION = "0.1.0"


@dataclass
class Turn:
    """A user turn from any chat app, normalized (channel-agnostic)."""

    source: str
    instruction: str
    user: Optional[str] = None
    channel: Optional[str] = None
    thread: Optional[str] = None
    #: App event id, when the app redelivers (used for de-duplication).
    event_id: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Per-thread coordination state — non-authoritative and reconstructable.

    Keyed by ``context_id``. The authoritative job state lives in the control
    plane; this is a coordination/presentation cache the interface can rebuild
    (from the chat thread + ``workboard_state``) after a restart. Concrete apps
    subclass it for presentation state (e.g. a message ts to edit, seen ids).
    """

    context_id: str
    active_job_id: Optional[str] = None
    pending_review: Optional[str] = None


class SessionStore:
    """A ``context_id``-keyed, in-memory store of :class:`Session`\\ s.

    ``factory`` builds a session for a fresh key, so a subclass can store its own
    :class:`Session` subclass without the store knowing the concrete type.
    """

    def __init__(self, factory: Callable[[str], Session] = Session) -> None:
        self._factory = factory
        self._by_ctx: dict[str, Session] = {}

    def get_or_create(self, context_id: str) -> Session:
        sess = self._by_ctx.get(context_id)
        if sess is None:
            sess = self._factory(context_id)
            self._by_ctx[context_id] = sess
        return sess

    def get(self, context_id: str) -> Optional[Session]:
        return self._by_ctx.get(context_id)


class InterfaceAgent(BaseAgent, abc.ABC):
    """Channel-agnostic chat boundary: forward turns to a control plane, deliver back."""

    #: Set by the concrete app (``"slack"``, …); used in ``context_id`` and payload.
    SOURCE: str = ""

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        control_plane: SendTarget,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> None:
        super().__init__(
            agent_id, version or DEFAULT_INITIAL_VERSION, sandbox_config or wan_sandbox_config()
        )
        #: The one A2A egress: a trusted control plane that owns routing + kicks.
        self.control_plane = control_plane
        self.sessions = SessionStore(self.new_session)
        # WAN to reach the chat app; never the host Docker socket.
        self.forbid_docker_socket()

    # ------------------------------------------------------------------ #
    # Inbound: a user turn → single forward to the control plane          #
    # ------------------------------------------------------------------ #
    async def handle_task(self, message: Message) -> Message:
        turn = self.parse_turn(message)
        ctx = self.context_id_for(turn)
        sess = self.sessions.get_or_create(ctx)
        if self.is_duplicate(turn, sess):
            return self.render_ack(turn, {"status": "duplicate", "job_id": ""})
        request = build_submit_request(
            self.explicit_target(turn),  # "" ⇒ the control plane routes (D5)
            turn.instruction,
            context_id=ctx,
            payload=self.forward_payload(turn),
            feedback_for=sess.pending_review,  # a reply while a review is pending ⇒ relay it
        )
        reply = await self.send_a2a_message(self.control_plane, request)
        ack = parse_submitted_reply(reply)
        self._record_ack(sess, ack)
        return self.render_ack(turn, ack)

    def _record_ack(self, sess: Session, ack: Mapping[str, Any]) -> None:
        """Cache the accepted job id (control plane is authoritative) and clear review wait."""
        if ack.get("status") == "ok" and ack.get("job_id"):
            sess.active_job_id = str(ack["job_id"])
            sess.pending_review = None

    # ------------------------------------------------------------------ #
    # Egress: a job_update pushed by the control plane → the thread       #
    # ------------------------------------------------------------------ #
    async def on_job_update(self, message: Message) -> None:
        update = parse_job_update(message)
        if not update:
            return
        await self.deliver(update.get("coords") or {}, self.render_update(update))

    # ------------------------------------------------------------------ #
    # Common derivation (shared across apps)                             #
    # ------------------------------------------------------------------ #
    def context_id_for(self, turn: Turn) -> str:
        """The thread's session key: ``f"{SOURCE}:{channel}:{thread}"``."""
        return f"{self.SOURCE}:{turn.channel}:{turn.thread}"

    def forward_payload(self, turn: Turn) -> dict[str, Any]:
        """Ride-along reply coords (for the completion path) plus any steering.

        Coords go under the fixed, source-agnostic ``reply_coords`` key so the
        control plane can echo them into a ``job_update`` without knowing the app;
        ``source`` is tagged inside so the delivering interface knows they're its.
        """
        payload: dict[str, Any] = {"reply_coords": self.reply_coords(turn)}
        steering = self.steering(turn)
        if steering:
            payload["steering"] = steering
        return payload

    def reply_coords(self, turn: Turn) -> dict[str, Any]:
        return {"source": self.SOURCE, "channel": turn.channel, "thread": turn.thread, "user": turn.user}

    # ------------------------------------------------------------------ #
    # Overridable hooks (sensible defaults)                              #
    # ------------------------------------------------------------------ #
    def new_session(self, context_id: str) -> Session:
        return Session(context_id=context_id)

    def is_duplicate(self, turn: Turn, sess: Session) -> bool:
        return False

    def explicit_target(self, turn: Turn) -> str:
        """An explicit ``@agent`` override, or ``""`` to let the control plane route."""
        return ""

    def steering(self, turn: Turn) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------ #
    # Channel-specific seams (subclass responsibility)                   #
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def parse_turn(self, message: Message) -> Turn:
        """Unpack this app's inbound envelope into a :class:`Turn`."""
        raise NotImplementedError

    @abc.abstractmethod
    def render_ack(self, turn: Turn, ack: Mapping[str, Any]) -> Message:
        """Render the immediate reply to a turn (received / duplicate / error)."""
        raise NotImplementedError

    @abc.abstractmethod
    def render_update(self, update: Mapping[str, Any]) -> str:
        """Render a terminal/progress ``job_update`` for the thread."""
        raise NotImplementedError

    @abc.abstractmethod
    async def deliver(self, coords: Mapping[str, Any], text: str) -> None:
        """Push ``text`` into the thread identified by ``coords`` (app transport)."""
        raise NotImplementedError
