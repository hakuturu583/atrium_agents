# atrium_agents

The **evolvable** Atrium agent set — the self-updating agents, kept in their own
repository so the self-evolution loop can rewrite them without ever touching the
trusted host code (the `atrium` control plane and evolution machinery). They
depend on `atrium` (a pinned dependency) for the shared runtime: `BaseAgent`, the
A2A protocol, sandbox types and the agent factory.

## The agents

| Agent | Base | Single responsibility |
| --- | --- | --- |
| `TabbyLLMAgent` | `InferenceAgent` | Run an LLM. WAN-isolated, GPU-only inference over A2A (tabbyAPI / exllamav3). Knows nothing about prompt assembly. |
| `PromptBuilderAgent` | `BaseAgent` | Assemble role prompts. Model-free A2A service. Knows nothing about models. |

The two are joined by a thin seam — a
[`PromptSource`](src/atrium_agents/prompt_source.py), injected into the inference
agent — so neither depends on the other's internals.

## Prompts as a service: coder ⟂ reviewer

The prompt an LLM receives is an *assembly of reusable, ordered sections*
(identity, tone, tool guidance, rules, project memory, the current objective …).
That assembly is the layered
[`PromptMemory`](src/atrium_agents/prompt_memory.py) engine, and the concrete
role prompts live as **profiles** in
[`prompt_profiles.py`](src/atrium_agents/prompt_profiles.py) — deliberately
**LLM-agnostic**: the same `coder` / `reviewer` profile composes the same prompt
whichever backend runs behind it, so swapping the model never rewrites the role's
instructions.

`PromptBuilderAgent` serves those profiles from **its own A2A endpoint**. It
holds no model and needs no GPU — composing a prompt is pure, host-side string
work. Making it a separate agent lets a **coder** and a **reviewer** run as two
distinct A2A agents with **unshared contexts**, each drawing its role prompt from
one common source:

```text
                     ┌──────────────────────┐
                     │  PromptBuilderAgent   │  profiles: coder, reviewer
                     └──────────┬───────────┘
             build:coder │      │ build:reviewer   (A2A, via PromptSource)
                    ┌─────▼──┐   └──▼────────┐
                    │ coder  │      │reviewer│      separate contexts
                    └───┬────┘      └───┬────┘
                        └──── one shared tabby backend ────┘
```

The reviewer evaluates a deliverable it **did not write**, with no window into
the coder's reasoning or intermediate turns — an independent second opinion,
which is the accuracy win the split is for.

### The seam: `PromptSource`

An inference agent has **no built-in role**. It is handed a `PromptSource` and
asks it for the system prompt each turn — the source it receives *is* its role:

* `RemotePromptSource(target, profile)` — fetch the prompt from a
  `PromptBuilderAgent` over **A2A** (the default path).
* `LocalPromptSource(builder, profile)` — call a co-located builder directly (no
  network; handy for single-process wiring and tests).

### Wiring it up

```python
from atrium_agents.prompt_builder_agent import PromptBuilderAgent
from atrium_agents.prompt_source import RemotePromptSource
from atrium_agents.tabby_llm_agent.agent import TabbyLLMAgent

# 1. A model-free prompt service (serves the built-in coder/reviewer profiles).
builder = PromptBuilderAgent("prompt-builder")
pb = builder.a2a_endpoint()

# 2. Two inference clients fanned into ONE tabby backend. Their role is the
#    injected source; they keep independent A2A contexts.
coder = TabbyLLMAgent.connect(
    "coder", bridge_url, prompt_source=RemotePromptSource(pb, "coder"),
)
reviewer = TabbyLLMAgent.connect(
    "reviewer", bridge_url, prompt_source=RemotePromptSource(pb, "reviewer"),
)

patch  = await coder.infer("Implement the feature described in TASK.md")
review = await reviewer.infer(f"Review this patch against TASK.md:\n{patch}")
```

An agent with no `prompt_source` is a role-agnostic backend: it sends no system
prompt unless the caller passes one explicitly.

### A2A contract

`PromptBuilderAgent` speaks a two-verb contract over A2A (a structured data part):

* **build** — `{"type": "build", "profile": name, "context": {...}, "tools": [...],
  "include": [...], "exclude": [...]}` → a text reply with the composed system
  prompt (`metadata.status == "ok"`).
* **list_profiles** — `{"type": "list_profiles"}` → a data reply
  `{"profiles": [...]}`.

Profiles can also be defined in YAML and layered over (or replace) the built-ins
via `PromptBuilderAgent.from_yaml(...)`.

## Development

```bash
uv run pytest tests --ignore=tests/integration   # unit tests (no GPU / no sandbox)
```

For local control-plane development, override the pinned `atrium` dependency with
the sibling checkout (see the comment in `pyproject.toml`).
