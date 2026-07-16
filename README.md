# atrium_agents

The **evolvable** Atrium agent set — the self-updating agents, kept in their own
repository so the self-evolution loop can rewrite them without ever touching the
trusted host code (the `atrium` control plane and evolution machinery). They
depend on `atrium` (a pinned dependency) for the shared runtime: `BaseAgent`, the
A2A protocol, sandbox types and the agent factory.

## The agent + its role

There is one inference agent — `TabbyLLMAgent` (WAN-isolated, GPU-only inference
over A2A, tabbyAPI / exllamav3) — and a **`Role`** that says what it is *for*:

| Piece | Responsibility |
| --- | --- |
| `InferenceAgent` / `TabbyLLMAgent` | Run an LLM. Role-agnostic engine. |
| `Role` (on the `InferenceAgent` layer) | A prompt **profile** + how the agent frames I/O (request → prompt, model text → reply). |

`Role` lives on the `InferenceAgent` layer — not on any concrete backend — so a
future backend (another `InferenceAgent` subclass over a different serving stack)
reuses the exact same role machinery.

## Coder ⟂ reviewer: one engine, two roles

A coder and a reviewer are the **same** `TabbyLLMAgent` client, fanned into one
GPU-resident model, differing only by the `Role` handed to them:

```text
                 ┌──────────── one tabby backend (shared model) ───────────┐
                 │                                                          │
        role=coder_role()                                       role=reviewer_role()
        ┌────────▼────────┐                                     ┌──────────▼─────────┐
        │  coder client   │  code / tool-calls                  │  reviewer client   │  VERDICT
        └─────────────────┘                                     └────────────────────┘
                 └──────── concurrent requests → continuous batching ───────┘
```

Because they are two clients of *one* backend, their concurrent inference
requests ride tabbyAPI's **continuous batching** and share the KV cache — make
them separate backends and that is lost. They keep **unshared contexts**, so the
reviewer judges a deliverable it never authored, with no window into the coder's
reasoning — the review-accuracy win.

A **profile** is a layered [`PromptMemory`](src/atrium_agents/prompt_memory.py)
(see [`prompt_profiles.py`](src/atrium_agents/prompt_profiles.py)) — deliberately
**LLM-agnostic**, so swapping the model never rewrites a role. The role composes
its profile **locally** (pure string assembly — no prompt service, no extra
network hop); the profiles are still shared, backend-independent data.

### Roles

* `coder_role()` — the `coder` profile with passthrough framing. It also folds a
  rework `review_feedback` payload into the prompt, so a coder re-dispatched by
  the workboard review gate sees the reviewer's feedback.
* `reviewer_role()` → `ReviewerRole` — reads a `review_request` (task +
  deliverable) into a review prompt and parses the model's `VERDICT: approve /
  request-changes` into a **workboard verdict**, which is what couples it to
  `atrium.orchestration` (the review gate). Fail-closed: an ambiguous review is
  never an approval.

### Wiring it up

```python
from atrium_agents.role import coder_role, reviewer_role
from atrium_agents.tabby_llm_agent.agent import TabbyLLMAgent

# Two clients fanned into ONE backend, differing only by role.
coder    = TabbyLLMAgent.connect("coder",    bridge_url, role=coder_role())
reviewer = TabbyLLMAgent.connect("reviewer", bridge_url, role=reviewer_role())

patch  = await coder.infer("Implement the feature described in TASK.md")
review = await reviewer.infer(...)   # or dispatched by the workboard review gate
```

The reviewer is the endpoint the control-plane **review gate**
(`atrium.orchestration.review`) dispatches each node's deliverable to; its verdict
becomes the node's Prefect state. An agent with no role is a role-agnostic
backend: it sends no system prompt unless the caller passes one.

## Development

```bash
uv run pytest tests --ignore=tests/integration   # unit tests (no GPU / no sandbox)
```

For local control-plane development, override the pinned `atrium` dependency with
the sibling checkout (see the comment in `pyproject.toml`).
