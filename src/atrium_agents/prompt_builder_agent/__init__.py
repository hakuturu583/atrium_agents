"""PromptBuilderAgent — a model-free, independently versioned Atrium agent.

This package bundles one generation of the prompt-assembly service:

* ``agent.py`` — the host-side :class:`PromptBuilderAgent` (A2A only), which
  serves role-specific system prompts composed from
  :mod:`atrium_agents.prompt_profiles`.
* ``card.py``  — its A2A :class:`AgentCard`.

Unlike the inference agents it feeds, it holds no model and needs no GPU:
composing a prompt is pure, host-side string work. ``__version__`` is the single
source of truth for the agent version and its image tag
``local-registry/prompt_builder_agent:<__version__>``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from atrium_agents.prompt_builder_agent.agent import (
    DEFAULT_PROFILE,
    KIND_BUILD,
    KIND_LIST,
    STATUS_ERROR,
    STATUS_OK,
    PromptBuilderAgent,
)
from atrium_agents.prompt_builder_agent.card import build_agent_card

__all__ = [
    "PromptBuilderAgent",
    "build_agent_card",
    "KIND_BUILD",
    "KIND_LIST",
    "STATUS_OK",
    "STATUS_ERROR",
    "DEFAULT_PROFILE",
    "__version__",
]
