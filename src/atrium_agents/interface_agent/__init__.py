"""Interface agents — the human I/O boundary for chat apps.

* ``agent.py`` — the abstract :class:`InterfaceAgent` (channel-agnostic dispatch:
  turn → ``context_id`` → single forward to a control plane → deliver back), plus
  the :class:`Turn` / :class:`Session` / :class:`SessionStore` value objects.
* ``slack.py`` — the concrete :class:`SlackInterfaceAgent`.

A propose-only edge: it forwards turns to the trusted control plane and holds no
authority, which is why it lives in this evolvable tier. See the atrium core
design ``docs/design/interface-agent.md``.

``__version__`` is the single source of truth for the agent version and its image
tag ``local-registry/slack_interface_agent:<__version__>``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from atrium_agents.interface_agent.agent import (
    InterfaceAgent,
    Session,
    SessionStore,
    Turn,
)
from atrium_agents.interface_agent.slack import (
    SlackInterfaceAgent,
    SlackPoster,
    SlackSession,
)

__all__ = [
    "InterfaceAgent",
    "Turn",
    "Session",
    "SessionStore",
    "SlackInterfaceAgent",
    "SlackSession",
    "SlackPoster",
    "__version__",
]
