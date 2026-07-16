"""``PromptBuilderAgent`` — an LLM-agnostic prompt-assembly service over A2A.

The prompt an ``LLMAgent`` receives is not the model's concern: it is an
*assembly of reusable, ordered role sections* (see
:mod:`atrium_agents.prompt_memory`). PromptBuilderAgent pulls that assembly out
of any one inference agent and exposes it as its **own A2A endpoint**, so the
prompt logic lives in one place, independent of the backend behind it. Swap the
LLM (tabbyAPI, an OpenAI-style server, anything) and the role prompts are
unchanged — they are fetched, composed, over the wire.

Why a separate agent rather than a library call? Because it lets a **coder** and
a **reviewer** run as two distinct A2A agents with *unshared contexts*, each
drawing its role prompt from this one common source. The reviewer, judging a
deliverable it never authored and with no window into the coder's reasoning,
gives an independent second opinion — the accuracy win the split is for.

It holds no model and needs no GPU: composition is pure, host-side string work.
The A2A contract is small:

* **build** — ``{"type": "build", "profile": name, "context": {...},
  "tools": [...], "include": [...], "exclude": [...]}`` → a text reply carrying
  the composed system prompt (``metadata.status == "ok"``).
* **list_profiles** — ``{"type": "list_profiles"}`` → a data reply
  ``{"profiles": [...]}``.

    BaseAgent → PromptBuilderAgent
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from atrium_agents.prompt_memory import PromptMemory
from atrium_agents.prompt_profiles import builtin_profiles
from atrium.core.base_agent import BaseAgent
from atrium.core.types import NetworkMode, SandboxConfig, VersionTag
from atrium.protocol import (
    Message,
    Role,
    data_message,
    get_message_text,
    metadata_dict,
    text_message,
)

logger = logging.getLogger("atrium.agents.prompt_builder")

# --------------------------------------------------------------------------- #
# A2A contract — request "kinds" and reply statuses (shared with consumers).   #
# --------------------------------------------------------------------------- #
KIND_BUILD = "build"
KIND_LIST = "list_profiles"

STATUS_OK = "ok"
STATUS_ERROR = "error"

#: Default profile served when a ``build`` request names none.
DEFAULT_PROFILE = "coder"

__all__ = [
    "PromptBuilderAgent",
    "KIND_BUILD",
    "KIND_LIST",
    "STATUS_OK",
    "STATUS_ERROR",
    "DEFAULT_PROFILE",
]


def _prompt_builder_defaults() -> SandboxConfig:
    """A WAN-cut-off, GPU-free sandbox config: prompt assembly is offline work."""
    return SandboxConfig(network=NetworkMode.INTERNAL, internal=True)


class PromptBuilderAgent(BaseAgent):
    """Serve composed, role-specific system prompts over A2A (no model, no GPU).

    Parameters
    ----------
    agent_id, version:
        Standard :class:`BaseAgent` identity.
    profiles:
        Name → :class:`PromptMemory` registry. Defaults to the built-in
        ``coder`` / ``reviewer`` profiles
        (:func:`~atrium_agents.prompt_profiles.builtin_profiles`).
    default_profile:
        Profile used when a ``build`` request names none.
    """

    AGENT_SLUG = "prompt_builder_agent"

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        profiles: Optional[Mapping[str, PromptMemory]] = None,
        default_profile: str = DEFAULT_PROFILE,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> None:
        from atrium_agents.prompt_builder_agent import __version__

        super().__init__(
            agent_id,
            version or __version__,
            sandbox_config or _prompt_builder_defaults(),
        )
        self.profiles: dict[str, PromptMemory] = (
            dict(profiles) if profiles is not None else builtin_profiles()
        )
        self.default_profile = default_profile

    # ------------------------------------------------------------------ #
    # Registry management                                                #
    # ------------------------------------------------------------------ #
    def register_profile(self, name: str, memory: PromptMemory) -> "PromptBuilderAgent":
        """Add or replace the profile called ``name`` (chainable)."""
        self.profiles[name] = memory
        return self

    def profile_names(self) -> list[str]:
        """The names of every registered profile (sorted, stable for display)."""
        return sorted(self.profiles)

    # ------------------------------------------------------------------ #
    # Core: compose a named profile                                      #
    # ------------------------------------------------------------------ #
    def compose(
        self,
        profile: Optional[str] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> str:
        """Compose ``profile`` over the given context, exposing ``tools`` to it.

        ``tools`` is placed on the compose context as ``ctx["tools"]`` (so a tool
        layer can render its ``<tools>`` block); an explicit ``context["tools"]``
        wins if provided. Raises ``KeyError`` for an unknown profile.
        """
        name = profile or self.default_profile
        try:
            memory = self.profiles[name]
        except KeyError:
            raise KeyError(
                f"unknown profile {name!r}; known: {self.profile_names()}"
            ) from None
        ctx: dict[str, Any] = dict(context or {})
        ctx.setdefault("tools", tools or [])
        return memory.compose(
            ctx,
            include=set(include) if include is not None else None,
            exclude=set(exclude) if exclude is not None else None,
        )

    # ------------------------------------------------------------------ #
    # A2A glue                                                           #
    # ------------------------------------------------------------------ #
    async def handle_task(self, message: Message) -> Message:
        """Dispatch an inbound A2A request to ``build`` or ``list_profiles``."""
        request = self.merge_data_parts(message)
        kind = request.get("type") or metadata_dict(message).get("kind") or KIND_BUILD
        ctx_id = message.context_id or None
        task_id = message.task_id or None

        if kind == KIND_LIST:
            return data_message(
                {"profiles": self.profile_names()},
                role=Role.ROLE_AGENT,
                context_id=ctx_id,
                task_id=task_id,
                metadata={"status": STATUS_OK},
            )

        if kind == KIND_BUILD:
            try:
                composed = self.compose(
                    request.get("profile"),
                    context=request.get("context"),
                    tools=request.get("tools"),
                    include=request.get("include"),
                    exclude=request.get("exclude"),
                )
            except KeyError as exc:
                return self._error(str(exc), ctx_id, task_id)
            return text_message(
                composed,
                role=Role.ROLE_AGENT,
                context_id=ctx_id,
                task_id=task_id,
                metadata={
                    "status": STATUS_OK,
                    "profile": request.get("profile") or self.default_profile,
                },
            )

        return self._error(
            f"unknown request kind {kind!r}; expected {KIND_BUILD!r} or {KIND_LIST!r}",
            ctx_id,
            task_id,
        )

    @staticmethod
    def _error(reason: str, ctx_id: Optional[str], task_id: Optional[str]) -> Message:
        return text_message(
            reason,
            role=Role.ROLE_AGENT,
            context_id=ctx_id,
            task_id=task_id,
            metadata={"status": STATUS_ERROR},
        )

    # ------------------------------------------------------------------ #
    # YAML construction                                                  #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(
        cls,
        path: str,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        merge_builtins: bool = True,
    ) -> "PromptBuilderAgent":
        """Build an agent whose profiles are loaded from a YAML file.

        Shape — a top-level ``profiles:`` mapping of name → a ``prompt:`` block
        (:meth:`PromptMemory.from_mapping`); an optional ``default_profile:``::

            default_profile: coder
            profiles:
              coder:    {order: [identity, tools], layers: {identity: {content: "..."}}}
              reviewer: {order: [identity],        layers: {identity: {content: "..."}}}

        With ``merge_builtins`` (default) the file's profiles are layered on top
        of the built-in ``coder`` / ``reviewer`` set, overriding by name; set it
        False to serve *only* the file's profiles.
        """
        import yaml

        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, Mapping):
            raise ValueError(f"{path}: top-level YAML must be a mapping")

        profiles: dict[str, PromptMemory] = builtin_profiles() if merge_builtins else {}
        for name, spec in (doc.get("profiles") or {}).items():
            profiles[name] = PromptMemory.from_mapping(spec)
        return cls(
            agent_id,
            version,
            profiles=profiles,
            default_profile=doc.get("default_profile", DEFAULT_PROFILE),
        )
