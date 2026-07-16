"""Unit tests for the LLM-agnostic role prompt *profiles*.

Pure / GPU-free: they exercise the ``coder`` / ``reviewer`` profiles and their
composition without a sandbox or a model.
"""

from __future__ import annotations

from atrium_agents.prompt_profiles import (
    BUILTIN_PROFILE_NAMES,
    builtin_profiles,
    coder_profile,
    reviewer_profile,
)

SAMPLE_TOOL = {
    "type": "function",
    "function": {"name": "read_file", "description": "Read a file", "parameters": {}},
}


def test_builtin_profiles_cover_the_declared_names():
    profiles = builtin_profiles()
    assert set(profiles) == set(BUILTIN_PROFILE_NAMES) == {"coder", "reviewer"}


def test_builtin_profiles_are_independent_copies():
    # Each call yields fresh PromptMemory objects (mutating one never leaks).
    a = builtin_profiles()
    b = builtin_profiles()
    a["coder"].remove("identity")
    assert "identity" in b["coder"].layers


def test_coder_profile_composes_identity_and_tools():
    composed = coder_profile().compose({"tools": [SAMPLE_TOOL]})
    assert "coding agent" in composed
    assert "<tools>" in composed  # tool schemas rendered into the prompt
    assert "read_file" in composed


def test_reviewer_profile_is_evaluation_oriented_and_distinct():
    reviewer = reviewer_profile().compose()
    coder = coder_profile().compose()
    assert reviewer != coder
    # Reviewer judges a deliverable it did not author.
    assert "did NOT write" in reviewer
    assert "VERDICT" in reviewer
    # ...and carries none of the coder's authoring identity.
    assert "focused coding agent" not in reviewer


def test_reviewer_profile_drops_empty_volatile_layers():
    # objective/memory/environment are empty by default -> skipped on compose.
    composed = reviewer_profile().compose()
    # No stray blank sections: composing twice with no context is stable.
    assert composed == reviewer_profile().compose()
    assert composed.strip() == composed
