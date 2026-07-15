"""Atrium agents — the evolvable agent set (self-updated by the evolution loop).

These agents live in their own repository, deliberately separated from the two
things the self-evolution loop must **never** rewrite:

* the fixed **control plane** — ``atrium.core`` / ``atrium.sandbox`` /
  ``atrium.protocol`` (registry, factory, Morpher, base agent, A2A protocol);
* the fixed **evolution machinery** — the builder, task and code-workspace
  agents that drive and execute the loop.

Keeping the evolvable workers here means a coding agent can clone and rewrite
*this* repo freely without ever touching the trusted host code. They depend on
``atrium`` (a pinned dependency) for the shared runtime — :class:`BaseAgent`, the
A2A protocol, sandbox types and the agent factory.
"""

from __future__ import annotations

from atrium.core.factory import register_agent_type

from atrium_agents.inference_agent import InferenceAgent, InferenceSettings
from atrium_agents.prompt_memory import (
    PromptLayer,
    PromptMemory,
    default_prompt_memory,
    tools_layer,
)
from atrium_agents.tabby_llm_agent import TabbyAgentConfig, TabbyLLMAgent

__all__ = [
    "InferenceAgent",
    "InferenceSettings",
    "PromptLayer",
    "PromptMemory",
    "default_prompt_memory",
    "tools_layer",
    "TabbyAgentConfig",
    "TabbyLLMAgent",
]

# Register the evolvable concrete agents so they can be launched from a bare slug
# (create_agent_by_slug) once the registry has an active generation for them.
register_agent_type(TabbyLLMAgent)
