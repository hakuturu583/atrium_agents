"""A2A AgentCard for the PromptBuilderAgent."""

from __future__ import annotations

from collections.abc import Sequence

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

__all__ = ["build_agent_card"]


def build_agent_card(
    version: str,
    *,
    name: str = "prompt_builder_agent",
    profiles: Sequence[str] = ("coder", "reviewer"),
) -> AgentCard:
    """Build the A2A card advertising the prompt-assembly capability.

    The available role ``profiles`` are listed in the skill tags so callers can
    discover which prompts this builder serves.
    """
    skill = AgentSkill(
        id="prompt_build",
        name="Prompt assembly",
        description=(
            "Compose LLM-agnostic, role-specific system prompts (coder, reviewer, "
            "…) from layered profiles and return them over A2A."
        ),
        tags=["prompt", "prompt-builder", "llm-agnostic", *[f"profile:{p}" for p in profiles]],
    )

    return AgentCard(
        name=name,
        description=(
            "Atrium PromptBuilderAgent: a model-free A2A service that assembles "
            "role-specific system prompts, so coder and reviewer agents draw "
            "their prompts from one backend-independent source."
        ),
        version=str(version),
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        default_input_modes=["application/json", "text/plain"],
        default_output_modes=["text/plain", "application/json"],
    )
