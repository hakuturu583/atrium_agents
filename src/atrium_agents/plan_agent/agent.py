"""``PlanAgent`` — a TabbyLLMAgent carrying the planner role, with its own slug.

See the package docstring. There is no new backend and no custom ``handle_task``:
the inherited :meth:`InferenceAgent.handle_task` frames the request through the
injected :func:`~atrium_agents.role.planner_role` (request → planning prompt →
``infer`` → ``plan_result``). This class exists only to give the planner a
registrable ``AGENT_SLUG`` the control plane can target, while reusing the tabby
backend and its WAN/GPU-isolated inference envelope unchanged.
"""

from __future__ import annotations

from typing import Optional

from atrium_agents.inference_agent import InferenceSettings, Role
from atrium_agents.role import planner_role
from atrium_agents.tabby_llm_agent.agent import TabbyAgentConfig, TabbyLLMAgent
from atrium.core.types import SandboxConfig, VersionTag

__all__ = ["PlanAgent"]


class PlanAgent(TabbyLLMAgent):
    """The planning specialist: request → generated ``flow.py`` + params.

    Defaults its role to :func:`planner_role` (so a bare construction is already a
    planner) and otherwise behaves exactly like a :class:`TabbyLLMAgent` — same
    engine, same isolation. The tuned coding-agent generation settings (4096 output
    tokens, temperature 0.2) are already the right register for emitting a whole
    ``flow.py``, so no settings override is needed.
    """

    #: Image slug → ``local-registry/plan_agent:<version>``.
    AGENT_SLUG = "plan_agent"

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        config: Optional[TabbyAgentConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        settings: Optional[InferenceSettings] = None,
        role: Optional[Role] = None,
    ) -> None:
        from atrium_agents.plan_agent import __version__

        super().__init__(
            agent_id,
            version or __version__,
            config=config,
            sandbox_config=sandbox_config,
            settings=settings,
            role=role if role is not None else planner_role(),
        )
