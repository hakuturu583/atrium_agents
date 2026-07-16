"""``InferenceAgent`` — shared abstract base for GPU-bound inference agents.

An inference agent runs entirely inside a GPU sandbox that is *cut off from the
public internet* (LAN/A2A only). This is the core defense: even if a model or
prompt is compromised, a WAN-isolated, GPU-only container cannot exfiltrate data
or escalate off the host.

``InferenceAgent`` enforces that security envelope at construction time and
implements the A2A glue (``handle_task``) so concrete subclasses only implement
:meth:`infer`. The inheritance chain is preserved::

    BaseAgent → InferenceAgent → TabbyLLMAgent
"""

from __future__ import annotations

import abc
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, Optional

from atrium_agents.prompt_memory import PromptMemory
from atrium.core.base_agent import BaseAgent
from atrium.core.errors import PolicyViolationError
from atrium.core.types import GPURequest, NetworkMode, SandboxConfig, VersionTag
from atrium.protocol import (
    Message,
    Role as A2ARole,
    get_message_data,
    get_message_text,
    text_message,
)

__all__ = [
    "InferenceAgent",
    "InferenceSettings",
    "ChatMessage",
    "Role",
]


def _merged_data(message: Message) -> dict[str, Any]:
    """Merge every structured data part of ``message`` into one mapping."""
    merged: dict[str, Any] = {}
    for part in get_message_data(message):
        if isinstance(part, dict):
            merged.update(part)
    return merged


@dataclass
class Role:
    """What an inference agent *is for* — its prompt profile + how it frames I/O.

    A :class:`InferenceAgent` is a role-agnostic engine; a ``Role`` is the thin,
    **backend-independent** thing that turns it into a *coder* or a *reviewer*. It
    lives here (not on any concrete backend) precisely so a future backend — a
    different ``InferenceAgent`` subclass over another serving stack — reuses the
    exact same role machinery: same profiles, same request/reply framing.

    It bundles the two aspects that actually differ between roles:

    * :attr:`prompt` — the layered :class:`PromptMemory` (its profile) that it
      composes into the system prompt; ``None`` sends no system prompt;
    * :meth:`frame_prompt` / :meth:`frame_reply` — how an inbound A2A request
      becomes the user prompt, and how the model's text becomes the reply.

    The role is a *parameter*, not a subclass or a separate backend — so a coder
    and a reviewer are the same client fanned into one model, and their concurrent
    requests ride the backend's continuous batching. The prompt is composed
    locally (pure string assembly — no service, no network); the profiles are
    still shared, backend-independent data. The base is a passthrough assistant
    (request text in → text out) that also folds a rework ``review_feedback``
    payload into the prompt, so a doer re-dispatched by the workboard review gate
    actually sees the feedback. Subclass to specialize framing (see
    :class:`~atrium_agents.role.ReviewerRole`).
    """

    name: str = "default"
    prompt: Optional[PromptMemory] = None

    def system_prompt(
        self,
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        """Compose this role's system prompt (``None`` when it has no profile)."""
        if self.prompt is None:
            return None
        ctx: dict[str, Any] = dict(context or {})
        ctx.setdefault("tools", tools or [])
        return self.prompt.compose(ctx) or None

    def frame_prompt(self, message: Message) -> str:
        """Turn an inbound A2A request into the model's user prompt."""
        text = get_message_text(message)
        feedback = _merged_data(message).get("review_feedback")
        if feedback:
            return f"{text}\n\n## Reviewer feedback (address this before finishing)\n{feedback}"
        return text

    def frame_reply(self, text: str, message: Message) -> Message:
        """Turn the model's text output into the A2A reply message."""
        return text_message(
            text,
            role=A2ARole.ROLE_AGENT,
            context_id=message.context_id or None,
            task_id=message.task_id or None,
        )

# An OpenAI-style chat message: ``{"role": "user"|"assistant"|"system"|"tool", "content": str}``.
ChatMessage = dict[str, Any]


@dataclass(slots=True)
class InferenceSettings:
    """Tunable generation + context-management knobs shared by all inference agents.

    Two concerns live here:

    * **Per-generation limits** — how a single completion is sampled
      (:attr:`max_output_tokens`, :attr:`temperature`, :attr:`top_p`,
      :attr:`stop`).
    * **Context management** — keeping a growing multi-turn history inside the
      model's window by *compacting* (summarizing) the oldest turns once the
      prompt approaches the budget. See :meth:`InferenceAgent.compact_messages`.
    """

    # --- per-generation sampling limits ---------------------------------- #
    max_output_tokens: int = 512
    """Hard cap on tokens generated in a single :meth:`InferenceAgent.infer` call."""
    temperature: float = 0.7
    top_p: float = 1.0
    stop: tuple[str, ...] = ()

    # --- context window / history compaction ----------------------------- #
    context_window_tokens: int = 8192
    """Total token budget of the served model (prompt + output)."""
    compaction_enabled: bool = True
    """When True, oversized histories are summarized before sending."""
    compaction_trigger_ratio: float = 0.75
    """Compact once the prompt exceeds this fraction of the usable input budget
    (``context_window_tokens - max_output_tokens``)."""
    compaction_keep_last_turns: int = 4
    """Most-recent non-system turns kept verbatim; older ones get summarized."""
    compaction_summary_max_tokens: int = 512
    """Output cap for the summary generation itself."""
    compaction_summary_temperature: float = 0.2
    """Low temperature keeps summaries faithful/deterministic."""

    # --- reasoning-model handling ---------------------------------------- #
    reasoning_open_tag: str = "<think>"
    reasoning_close_tag: str = "</think>"
    """Delimiters of a reasoning/"thinking" span. Compaction summaries strip
    this scaffolding so chain-of-thought doesn't re-bloat the context. The exact
    tokens are model-specific (``<think>``/``</think>`` for Qwen/Ornith, others
    differ), so they are configurable; set both to empty strings to disable
    stripping entirely."""

    def __post_init__(self) -> None:
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.context_window_tokens <= self.max_output_tokens:
            raise ValueError(
                "context_window_tokens must exceed max_output_tokens "
                f"({self.context_window_tokens} <= {self.max_output_tokens})"
            )
        if not 0.0 < self.compaction_trigger_ratio <= 1.0:
            raise ValueError("compaction_trigger_ratio must be in (0, 1]")
        if self.compaction_keep_last_turns < 0:
            raise ValueError("compaction_keep_last_turns must be >= 0")

    @property
    def input_budget_tokens(self) -> int:
        """Tokens available for the prompt once room for the output is reserved."""
        return self.context_window_tokens - self.max_output_tokens

    @property
    def compaction_trigger_tokens(self) -> int:
        """Prompt size (in tokens) at which compaction kicks in."""
        return int(self.input_budget_tokens * self.compaction_trigger_ratio)

    # --- (de)serialization: YAML / dict ---------------------------------- #
    @classmethod
    def _field_names(cls) -> frozenset[str]:
        return frozenset(f.name for f in fields(cls))

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "InferenceSettings":
        """Build settings from a plain mapping (e.g. parsed YAML).

        Unknown keys raise ``ValueError`` so typos in a config file fail loudly
        rather than silently doing nothing. ``stop`` is coerced to a tuple.
        """
        if not data:
            return cls()
        known = cls._field_names()
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown InferenceSettings field(s): {sorted(unknown)}; "
                f"valid keys are {sorted(known)}"
            )
        kwargs = dict(data)
        if "stop" in kwargs and kwargs["stop"] is not None:
            kwargs["stop"] = tuple(kwargs["stop"])
        return cls(**kwargs)

    def merge(self, data: Optional[Mapping[str, Any]]) -> "InferenceSettings":
        """Return a copy of these settings with ``data`` overrides applied.

        Use this to layer a YAML/dict on top of tuned defaults (e.g. a coding
        preset) so only the keys present in ``data`` change. Unknown keys raise.
        """
        if not data:
            return self
        unknown = set(data) - self._field_names()
        if unknown:
            raise ValueError(
                f"unknown InferenceSettings field(s): {sorted(unknown)}; "
                f"valid keys are {sorted(self._field_names())}"
            )
        kwargs = dict(data)
        if "stop" in kwargs and kwargs["stop"] is not None:
            kwargs["stop"] = tuple(kwargs["stop"])
        return replace(self, **kwargs)

    @classmethod
    def from_yaml(
        cls, path: str, *, section: Optional[str] = "inference"
    ) -> "InferenceSettings":
        """Load settings from a YAML file.

        When ``section`` is given and present at the top level, that sub-mapping
        is used (so one file can carry several configs); otherwise the whole
        document is treated as the settings mapping.
        """
        import yaml  # lazy: keep YAML an optional dependency of this module

        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if section and isinstance(doc, Mapping) and section in doc:
            doc = doc[section]
        return cls.from_mapping(doc)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (``stop`` as a list) for YAML/JSON dumping."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["stop"] = list(out["stop"])
        return out


def _secure_inference_defaults() -> SandboxConfig:
    """A WAN-isolated, GPU-enabled sandbox config (the inference envelope)."""
    return SandboxConfig(
        network=NetworkMode.INTERNAL,
        internal=True,
        device_requests=[GPURequest()],
    )


class InferenceAgent(BaseAgent, abc.ABC):
    """Abstract inference agent: WAN-isolated, LAN-only, GPU-required.

    Subclasses implement :meth:`infer`; the A2A request/response plumbing is
    handled here.
    """

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag",
        sandbox_config: Optional[SandboxConfig] = None,
        *,
        require_gpu: bool = True,
        settings: Optional[InferenceSettings] = None,
        role: Optional[Role] = None,
    ) -> None:
        super().__init__(agent_id, version, sandbox_config or _secure_inference_defaults())
        self._require_gpu = require_gpu
        self.settings = settings or InferenceSettings()
        # The agent's *role* (coder, reviewer, …) is injected, not built in: it
        # supplies the system prompt (its profile) and frames the agent's I/O. A
        # coder and a reviewer are the same engine handed different roles — which
        # is what lets them fan into one shared backend and continuous-batch. The
        # default role is a passthrough assistant with no system prompt.
        self.role = role if role is not None else Role("default")
        self._enforce_isolation_policy()

    def _enforce_isolation_policy(self) -> None:
        """Refuse any configuration that breaks the inference security envelope."""
        cfg = self.sandbox_config
        if cfg.network == NetworkMode.BRIDGE or not cfg.internal:
            raise PolicyViolationError(
                f"{type(self).__name__} must not have WAN access "
                f"(network={cfg.network.value}, internal={cfg.internal})"
            )
        if self._require_gpu and not cfg.gpu_enabled:
            raise PolicyViolationError(
                f"{type(self).__name__} requires GPU passthrough "
                "(sandbox_config.device_requests is empty)"
            )

    # ------------------------------------------------------------------ #
    # A2A glue (concrete) — turns an inbound message into an infer() call #
    # ------------------------------------------------------------------ #
    async def handle_task(self, message: Message) -> Message:
        """Adapt an inbound A2A request to :meth:`infer`, framed by the agent's role.

        The role decides how the request becomes the user prompt
        (:meth:`Role.frame_prompt`) and how the model's text becomes the reply
        (:meth:`Role.frame_reply`) — so a coder returns text while a reviewer
        returns a workboard verdict, from the same engine. Generation knobs come
        from the agent's :class:`InferenceSettings`; the role's profile supplies
        the system prompt inside :meth:`infer`.
        """
        prompt = self.role.frame_prompt(message)
        result = await self.infer(prompt)
        return self.role.frame_reply(result, message)

    # ------------------------------------------------------------------ #
    # Generation parameters                                              #
    # ------------------------------------------------------------------ #
    def resolve_generation_params(self, **overrides: Any) -> dict[str, Any]:
        """Merge per-call ``overrides`` over the agent's :class:`InferenceSettings`.

        Only ``None``-valued overrides fall back to the settings default, so a
        caller may pass ``temperature=0`` and have it respected. ``max_tokens``
        is the canonical key (aliased from ``max_output_tokens``) since that is
        what OpenAI-style backends expect.
        """
        resolved: dict[str, Any] = dict(overrides)
        if resolved.get("max_tokens") is None:
            resolved["max_tokens"] = self.settings.max_output_tokens
        if resolved.get("temperature") is None:
            resolved["temperature"] = self.settings.temperature
        if resolved.get("top_p") is None and self.settings.top_p != 1.0:
            resolved["top_p"] = self.settings.top_p
        if resolved.get("stop") is None and self.settings.stop:
            resolved["stop"] = list(self.settings.stop)
        # Drop keys left explicitly None so they don't override backend defaults.
        return {k: v for k, v in resolved.items() if v is not None}

    def with_settings(self, **changes: Any) -> InferenceSettings:
        """Return a copy of the current settings with ``changes`` applied and
        install it on this agent (chainable convenience for ad-hoc tuning)."""
        self.settings = replace(self.settings, **changes)
        return self.settings

    # ------------------------------------------------------------------ #
    # System prompt — composed from the role's profile                   #
    # ------------------------------------------------------------------ #
    async def resolve_system_prompt(
        self,
        system: Optional[str] = None,
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        """The system prompt to send for this call.

        An explicit ``system`` always wins. Otherwise the agent's :attr:`role`
        composes its profile locally (:meth:`Role.system_prompt`) — ``None`` when
        the role has no profile, sending no system message. Async so a future
        backend could source it remotely without changing callers.
        """
        if system is not None:
            return system
        return self.role.system_prompt(tools=tools, context=context)

    # ------------------------------------------------------------------ #
    # Token accounting                                                   #
    # ------------------------------------------------------------------ #
    def count_tokens(self, text: str) -> int:
        """Estimate the token count of ``text``.

        The base implementation is a fast, model-agnostic heuristic (~4 chars /
        token). Subclasses that have access to the real tokenizer (or a backend
        token-count endpoint) should override this for accurate budgeting.
        """
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)

    def count_message_tokens(self, messages: list[ChatMessage]) -> int:
        """Estimate the prompt token count of an OpenAI-style ``messages`` list.

        Adds a small per-message overhead to approximate the role/delimiter
        tokens most chat templates insert.
        """
        total = 0
        for m in messages:
            content = m.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            total += self.count_tokens(content) + 4
        return total

    # ------------------------------------------------------------------ #
    # History compaction ("compress once tokens pile up")                #
    # ------------------------------------------------------------------ #
    def should_compact(self, messages: list[ChatMessage]) -> bool:
        """Whether ``messages`` exceed the compaction trigger threshold."""
        if not self.settings.compaction_enabled:
            return False
        return self.count_message_tokens(messages) > self.settings.compaction_trigger_tokens

    async def compact_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Summarize the oldest turns when the history grows too large.

        Keeps every ``system`` message and the most recent
        ``compaction_keep_last_turns`` turns verbatim, replacing the middle with
        a single model-generated summary. Returns ``messages`` unchanged when
        compaction is disabled, under threshold, or there is nothing old enough
        to fold away. This is a no-op-safe building block: subclasses call it at
        the top of their chat loop.
        """
        if not self.should_compact(messages):
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        convo = [m for m in messages if m.get("role") != "system"]
        keep = self.settings.compaction_keep_last_turns
        to_summarize = convo[:-keep] if keep else convo
        recent = convo[-keep:] if keep else []
        if not to_summarize:
            return messages  # history is all "recent"; can't shrink further

        summary = await self._summarize_turns(to_summarize)
        summary_msg: ChatMessage = {
            "role": "system",
            "content": "Summary of earlier conversation (older turns were compacted):\n" + summary,
        }
        return [*system_msgs, summary_msg, *recent]

    async def _summarize_turns(self, turns: list[ChatMessage]) -> str:
        """Render ``turns`` to a transcript and self-summarize it via :meth:`infer`."""
        transcript = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content') or ''}" for m in turns
        )
        instruction = (
            "Summarize the following conversation excerpt. Preserve facts, decisions, "
            "names, numbers, and any open questions or pending tasks. Be concise and "
            "write only the summary."
        )
        raw = await self.infer(
            transcript,
            system=instruction,
            max_tokens=self.settings.compaction_summary_max_tokens,
            temperature=self.settings.compaction_summary_temperature,
        )
        return self.strip_reasoning(raw)

    def strip_reasoning(self, text: str) -> str:
        """Remove a reasoning/"thinking" span delimited by the configured tags.

        Takes whatever follows the final close tag (handling a single leading
        reasoning span), then drops any remaining well-formed open…close blocks.
        Returns ``text`` stripped unchanged when either tag is empty. The tags
        come from :class:`InferenceSettings` so they can be set per model (YAML).
        """
        open_tag = self.settings.reasoning_open_tag
        close_tag = self.settings.reasoning_close_tag
        if not open_tag or not close_tag:
            return text.strip()
        if close_tag in text:
            text = text.rsplit(close_tag, 1)[-1]
        block = re.compile(re.escape(open_tag) + ".*?" + re.escape(close_tag), re.DOTALL)
        return block.sub("", text).strip()

    # ------------------------------------------------------------------ #
    # Extension point                                                    #
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    async def infer(self, prompt: str, **params: Any) -> str:
        """Run inference for ``prompt`` and return the model's text output."""
        raise NotImplementedError
