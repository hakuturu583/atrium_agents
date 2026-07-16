"""LLM-agnostic *prompt profiles* — the reusable role prompts of the agent set.

A **profile** is a named :class:`~atrium_agents.prompt_memory.PromptMemory`: the
layered system prompt for one *role* (a coder, a reviewer, …), holding no model,
backend or generation knobs. That independence is the point — the same profile
composes the same prompt whether it is fed to tabbyAPI/exllamav3, an OpenAI-style
backend, or anything else, so swapping the underlying LLM never rewrites the
role's instructions.

Profiles are what a :class:`~atrium_agents.inference_agent.Role` carries and
composes locally into its system prompt. Splitting them out here (rather than
burying them inside a concrete inference agent) is what lets a *coder* and a
*reviewer* — the same engine handed different roles, with their own unshared
contexts — draw their role prompts from one common, backend-independent source.
The reviewer evaluating a deliverable it did not write, with no window into the
author's reasoning, is the accuracy win this separation buys.

The canonical layer set and order come from
:func:`~atrium_agents.prompt_memory.default_prompt_memory`: *stable* layers
(identity … rules) precede *volatile* ones (memory / environment / objective /
user_instructions), which each profile leaves empty for the caller to fill per
turn (empty layers are dropped on compose).
"""

from __future__ import annotations

from atrium_agents.prompt_memory import PromptLayer, PromptMemory, default_prompt_memory

__all__ = [
    "coder_profile",
    "reviewer_profile",
    "builtin_profiles",
    "BUILTIN_PROFILE_NAMES",
]

#: The canonical role names shipped as built-in profiles (see :func:`builtin_profiles`).
BUILTIN_PROFILE_NAMES = ("coder", "reviewer")


def coder_profile() -> PromptMemory:
    """The **coder** role prompt: make correct, minimal, targeted changes.

    Fills the stable layers (identity / tone / tool_guidance / rules) with
    coding-agent guidance and renders provided tool schemas as a ``<tools>``
    block; the volatile layers are left empty for per-turn population.
    """
    memory = default_prompt_memory()
    memory.record(
        PromptLayer(
            "identity",
            order=10,
            content=(
                "You are a focused coding agent working inside an isolated, "
                "WAN-cut-off sandbox. Produce correct, minimal, well-targeted "
                "changes."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "tone",
            order=20,
            content=(
                "Be concise and direct. Prefer concrete actions and code over "
                "prose, and match the conventions of the surrounding codebase."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "tool_guidance",
            order=30,
            content=(
                "Prefer the provided tools over ad-hoc shell when a tool fits. "
                "Call one tool at a time and wait for its result before the next. "
                "Read a file before editing it."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "rules",
            order=60,
            content=(
                "Keep changes minimal and reversible; avoid unrelated edits. "
                "Verify your work by running tests when possible. Stay within the "
                "sandbox — never attempt network egress or data exfiltration."
            ),
        )
    )
    return memory


def reviewer_profile() -> PromptMemory:
    """The **reviewer** role prompt: judge a deliverable, not its author.

    A reviewer runs in a *separate context* from the coder that produced the
    work: it never sees the author's chain-of-thought, only the artifact and the
    stated requirements. The prompt makes that stance explicit and asks for a
    structured, decisive verdict — an independent second opinion, which is the
    whole reason for splitting the two roles across two A2A agents.
    """
    memory = default_prompt_memory()
    memory.record(
        PromptLayer(
            "identity",
            order=10,
            content=(
                "You are a meticulous code reviewer evaluating a deliverable you "
                "did NOT write. You have no visibility into the author's reasoning "
                "or intermediate steps — judge only the artifact in front of you "
                "against the stated requirements."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "tone",
            order=20,
            content=(
                "Be specific, evidence-based and impartial. Cite exact locations "
                "(file and line) for every issue and quote the offending code. "
                "Do not restate the change back approvingly — look for what is "
                "wrong, missing, or risky."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "tool_guidance",
            order=30,
            content=(
                "Use read-only tools to inspect the code and run the tests; do "
                "not modify the deliverable. Read a file fully before judging it."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "rules",
            order=60,
            content=(
                "Check correctness first (does it meet the requirement and handle "
                "edge cases?), then safety, then clarity. Distinguish blocking "
                "defects from optional nits. Do not invent problems: if the work "
                "is correct, say so plainly."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "verdict",
            order=95,
            title="## Verdict format",
            content=(
                "End with a verdict line — `VERDICT: approve` or "
                "`VERDICT: request-changes` — followed by a bulleted list of "
                "findings, each tagged [blocking] or [nit] with a file:line "
                "reference."
            ),
        )
    )
    return memory


def builtin_profiles() -> dict[str, PromptMemory]:
    """The built-in role profiles, by name (``coder`` / ``reviewer``).

    Maps each name in :data:`BUILTIN_PROFILE_NAMES` to a freshly built
    :class:`PromptMemory` so callers own an independent, mutable copy.
    """
    return {"coder": coder_profile(), "reviewer": reviewer_profile()}
