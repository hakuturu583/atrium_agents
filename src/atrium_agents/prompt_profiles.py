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
    "planner_profile",
    "flow_reviewer_profile",
    "builtin_profiles",
    "BUILTIN_PROFILE_NAMES",
]

#: The canonical role names shipped as built-in profiles (see :func:`builtin_profiles`).
BUILTIN_PROFILE_NAMES = ("coder", "reviewer", "planner", "flow_reviewer")


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


def planner_profile() -> PromptMemory:
    """The **planner** role prompt: turn a request into a runnable job.

    A planner reads a human request (JSON) and emits **two artifacts that together
    form a job**: a self-contained Prefect ``flow.py`` and a ``params`` JSON. The
    ``flow.py`` is *not* the work — it is an agent-dispatch orchestration: each task
    assigns a piece of work to a subagent (a role-bearing inference agent) via the
    preinstalled trusted primitive ``atrium_dispatch(agent, instruction, payload)``,
    and the DAG edges are the dependencies. The prompt fixes a strict fenced-block
    output contract so the reply parses deterministically (mirroring the reviewer's
    ``VERDICT:`` contract).
    """
    memory = default_prompt_memory()
    memory.record(
        PromptLayer(
            "identity",
            order=10,
            content=(
                "You are a planning specialist. Given a human request (JSON), you "
                "produce a runnable job: a self-contained Prefect flow (`flow.py`) "
                "plus a `params` JSON. The flow ORCHESTRATES work — it does not do "
                "the work itself. Each task assigns a piece of work to a subagent "
                "and waits on its result; the task graph is the plan."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "tone",
            order=20,
            content=(
                "Be precise and minimal. Prefer the smallest DAG that satisfies the "
                "request. Assign each task to exactly one subagent from the roster "
                "you are given; never invent an agent that is not listed."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "rules",
            order=60,
            content=(
                "Dispatch to subagents ONLY through the preinstalled primitive "
                "`from atrium_dispatch import atrium_dispatch` — call "
                "`atrium_dispatch(agent, instruction, payload)` which returns "
                "`{status, text, data}`. Do not open sockets, import networking "
                "libraries, or attempt any egress: the runner is WAN-isolated and "
                "anything else is blocked. Use only the standard library, `prefect`, "
                "and `atrium_dispatch`. The flow must be self-contained and run to "
                "completion in-process (no Prefect server/worker). Read inputs from "
                "`params.json` next to the flow."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "output_format",
            order=95,
            title="## Output format",
            content=(
                "Emit EXACTLY two fenced code blocks and nothing that must be "
                "parsed outside them:\n"
                "1. a ```python block containing the entire `flow.py` — it MUST "
                "define a single `@flow`-decorated entrypoint named `main` AND call "
                "it at module level behind an "
                "`if __name__ == \"__main__\": main()` guard, so `python flow.py` "
                "actually runs it;\n"
                "2. a ```json block containing the `params` object (use `{}` if "
                "there are none).\n"
                "Any prose belongs outside the blocks. If you cannot produce a "
                "valid flow, emit no python block — do not emit a partial one."
            ),
        )
    )
    return memory


def flow_reviewer_profile() -> PromptMemory:
    """The **flow reviewer** role prompt: judge a generated flow *before* it runs.

    A specialization of the reviewer for the plan path: the deliverable is a
    generated Prefect ``flow.py`` (an agent-dispatch orchestration) that is about to
    be executed in a WAN-isolated runner. The review is therefore pre-execution and
    safety-first — it checks the flow only orchestrates (dispatches via
    ``atrium_dispatch``, defines a single ``main``, attempts no egress), not just
    that it is plausible code. Emits the same ``VERDICT:`` line the reviewer gate
    parses, so it plugs into the workboard review path unchanged.
    """
    memory = default_prompt_memory()
    memory.record(
        PromptLayer(
            "identity",
            order=10,
            content=(
                "You are a safety reviewer for a generated Prefect flow that is "
                "about to run in a WAN-isolated sandbox. You did NOT write it; judge "
                "only the flow source in front of you. The flow should ORCHESTRATE "
                "work (dispatch to subagents), not do risky work itself."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "tone",
            order=20,
            content=(
                "Be specific and decisive, and quote the offending line for every "
                "issue. Favor rejecting anything unsafe or ambiguous over approving."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "rules",
            order=60,
            content=(
                "Reject the flow unless ALL hold: it defines a single `main` "
                "entrypoint; it reaches other agents ONLY through "
                "`atrium_dispatch(agent, instruction, payload)`; it opens no sockets "
                "and imports no networking/egress libraries; it runs no shell, "
                "subprocess, or filesystem writes outside the workspace; it has no "
                "obvious unbounded loop or resource bomb. Approve a flow that only "
                "orchestrates dispatch calls and processes their results."
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
                "findings, each tagged [blocking] or [nit] with a line reference."
            ),
        )
    )
    return memory


def builtin_profiles() -> dict[str, PromptMemory]:
    """The built-in role profiles, by name (coder / reviewer / planner / flow_reviewer).

    Maps each name in :data:`BUILTIN_PROFILE_NAMES` to a freshly built
    :class:`PromptMemory` so callers own an independent, mutable copy.
    """
    return {
        "coder": coder_profile(),
        "reviewer": reviewer_profile(),
        "planner": planner_profile(),
        "flow_reviewer": flow_reviewer_profile(),
    }
