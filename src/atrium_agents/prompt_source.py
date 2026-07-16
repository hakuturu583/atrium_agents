"""``PromptSource`` — where an inference agent gets its system prompt from.

This is the seam that keeps the two concerns apart:

* an :class:`~atrium_agents.inference_agent.InferenceAgent` knows how to run a
  model, and *nothing* about prompt assembly;
* a :class:`~atrium_agents.prompt_builder_agent.PromptBuilderAgent` knows how to
  assemble role prompts, and *nothing* about models.

An agent is handed a ``PromptSource`` (dependency injection) and simply asks it
for the system prompt each turn. The source it receives *is* its role: give a
client a ``coder`` source and it is a coder; give another a ``reviewer`` source
and it is a reviewer — same backend, different injected prompt, unshared context.

Two implementations ship:

* :class:`RemotePromptSource` — the default: fetch the prompt from a
  ``PromptBuilderAgent`` over **A2A**, so several agents share one
  backend-independent prompt service.
* :class:`LocalPromptSource` — call a co-located builder directly, skipping the
  network (handy for single-process wiring and tests).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

from atrium_agents.prompt_builder_agent.agent import KIND_BUILD, STATUS_ERROR
from atrium.core.errors import AgentError
from atrium.protocol import (
    Role,
    data_message,
    get_message_text,
    metadata_dict,
    send_message,
)
from atrium.protocol.a2a_transport import SendTarget

if TYPE_CHECKING:
    from atrium_agents.prompt_builder_agent.agent import PromptBuilderAgent

__all__ = ["PromptSource", "RemotePromptSource", "LocalPromptSource"]


@runtime_checkable
class PromptSource(Protocol):
    """Something that yields a system prompt for one role.

    Implementations bind a single role (profile) at construction; a call passes
    only the per-turn ``tools`` and ``context`` the prompt may render. Returns
    ``None`` when the composed prompt is empty (no system message is sent).
    """

    async def system_prompt(
        self,
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]: ...


class RemotePromptSource:
    """Fetch a role prompt from a :class:`PromptBuilderAgent` over A2A.

    Parameters
    ----------
    target:
        The builder's A2A base URL (or resolved ``AgentCard``).
    profile:
        The role profile to request, e.g. ``"coder"`` or ``"reviewer"``.
    """

    def __init__(self, target: SendTarget, profile: str) -> None:
        self.target = target
        self.profile = profile

    async def system_prompt(
        self,
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        request: dict[str, Any] = {
            "type": KIND_BUILD,
            "profile": self.profile,
            "tools": tools or [],
        }
        if context:
            request["context"] = dict(context)
        message = data_message(request, role=Role.ROLE_USER, metadata={"kind": KIND_BUILD})
        reply = await send_message(self.target, message)
        if metadata_dict(reply).get("status") == STATUS_ERROR:
            raise AgentError(
                f"prompt builder rejected profile {self.profile!r}: {get_message_text(reply)}"
            )
        return get_message_text(reply) or None


class LocalPromptSource:
    """Compose a role prompt from a co-located builder, without A2A.

    The in-process counterpart to :class:`RemotePromptSource`: same role prompt,
    but resolved by calling :meth:`PromptBuilderAgent.compose` directly. Useful
    when the builder and the agent live in one process.
    """

    def __init__(self, builder: "PromptBuilderAgent", profile: str) -> None:
        self.builder = builder
        self.profile = profile

    async def system_prompt(
        self,
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        return self.builder.compose(self.profile, tools=tools, context=context) or None
