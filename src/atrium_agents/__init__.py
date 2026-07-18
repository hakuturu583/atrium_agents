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

from atrium_agents.inference_agent import InferenceAgent, InferenceSettings, Role
from atrium_agents.interface_agent import (
    InterfaceAgent,
    SlackInterfaceAgent,
    Turn,
)
from atrium_agents.prompt_memory import (
    PromptLayer,
    PromptMemory,
    default_prompt_memory,
    tools_layer,
)
from atrium_agents.prompt_profiles import (
    builtin_profiles,
    coder_profile,
    flow_reviewer_profile,
    planner_profile,
    reviewer_profile,
)
from atrium_agents.role import (
    PlannerRole,
    ReviewerRole,
    build_plan_prompt,
    build_review_prompt,
    coder_role,
    flow_reviewer_role,
    parse_plan_output,
    parse_verdict,
    planner_role,
    reviewer_role,
)
from atrium_agents.tabby_llm_agent import TabbyAgentConfig, TabbyLLMAgent

__all__ = [
    "InferenceAgent",
    "InferenceSettings",
    "Role",
    "InterfaceAgent",
    "SlackInterfaceAgent",
    "Turn",
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
    "PromptLayer",
    "PromptMemory",
    "default_prompt_memory",
    "tools_layer",
    "coder_profile",
    "reviewer_profile",
    "planner_profile",
    "flow_reviewer_profile",
    "builtin_profiles",
    "TabbyAgentConfig",
    "TabbyLLMAgent",
]

# Register the evolvable concrete agents so they can be launched from a bare slug
# (create_agent_by_slug) once the registry has an active generation for them.
# A coder / reviewer / planner is a TabbyLLMAgent + a role (not a distinct agent
# type), so there is no separate coder/reviewer/planner slug to register.
register_agent_type(TabbyLLMAgent)
register_agent_type(SlackInterfaceAgent)
