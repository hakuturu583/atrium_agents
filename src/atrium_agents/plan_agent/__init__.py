"""``PlanAgent`` — the LLM specialist that turns a request into a runnable job.

A plan agent reads a human request and drafts the job's Prefect ``flow.py`` +
params (see :func:`~atrium_agents.role.planner_role`). It is the *same* inference
engine as the coder/reviewer — a :class:`~atrium_agents.tabby_llm_agent.TabbyLLMAgent`
handed the ``planner`` role — wrapped in a thin subclass only so it has a
registrable slug (``plan_agent``) the control plane can target.

``__version__`` is the single source of truth for the agent version and its image
tag ``local-registry/plan_agent:<__version__>``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from atrium_agents.plan_agent.agent import PlanAgent

__all__ = ["PlanAgent", "__version__"]
