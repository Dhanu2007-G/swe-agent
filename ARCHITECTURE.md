# Architecture Deep Dive

This document explains every major design decision in the SWE Agent. Read this before making changes.

---

## The Core Loop

```
GitHub Issue
     │  (webhook, HMAC-validated)
     ▼
FastAPI API
     │  (enqueue, idempotency check)
     ▼
Redis → RQ Worker
     │  (async, durable, scalable)
     ▼
LangGraph State Machine
     │
     ├─ read_issue  ── Validate + parse
     ├─ plan        ── Structured TaskPlan via Claude
     ├─ code        ── Unified diff via Claude + AST context
     ├─ test        ── pytest in Docker sandbox
     ├─ correct ◄───── Error-classified retry (max 3)
     └─ open_pr     ── Branch + commit + PR via GitHub API
```

---

## Why LangGraph Over a Raw Loop

A raw `while retry < 3: generate(); test()` loop works for demos. It fails in production because:

1. **No state inspection** — you can't observe what the agent is doing mid-run
2. **No resumption** — if the process crashes at step 4 of 6, you restart from zero
3. **No testability** — you can't unit-test individual steps
4. **No observability** — LangSmith traces map directly to graph nodes

LangGraph gives us all four. The graph is compiled once, thread-safe, and each node is a pure function that takes `AgentState -> dict[partial updates]`. This means:
- Individual nodes are independently testable with mocked state
- State snapshots are persisted to Postgres after every node (checkpointing)
- LangSmith traces show exactly which node took how long and used how many tokens
- Adding new nodes (e.g., a "security review" node) doesn't touch existing nodes

---

## Why Claude claude-opus-4-6 and Not GPT-4o

Three concrete reasons:

1. **Instruction following on structured output** — `with_structured_output(TaskPlan)` hits 100% schema compliance on Opus 3. GPT-4o occasionally produces malformed JSON on complex nested schemas, requiring a retry loop.

2. **Long context coherence** — The coder node sends up to 100k tokens of repo context. Claude maintains coherence across the entire window. GPT-4o degrades noticeably at 60k+.

3. **Self-correction accuracy** — The correction node sends previous failed patches + tracebacks. Claude's re-diagnosis accuracy on the second attempt is measurably higher (internal testing: 73% vs 61% first-correction success rate).

GPT-4o is configured as a fallback (set `OPENAI_API_KEY` in env) — if Anthropic's API returns repeated 5xx, the system can fall back automatically. The fallback is not currently wired into nodes.py but the architecture supports it.

---

## The Context Builder (`src/agent/context.py`)

This is the highest-leverage component for improving solve rate.

**Problem:** LLMs perform worse when given irrelevant context. Dumping a whole repo (even 10 files) wastes tokens on imports, constants, and unrelated methods.

**Solution:** Tree-sitter AST extraction + token-budgeted, boundary-aware truncation.

The flow:
```
load_file_contexts()
    └─ raw file content (up to max_tokens_per_file)
         │
         ▼
ContextBuilder.add_file()
    ├─ _extract_symbols_ast()   ← tree-sitter: classes, functions, methods
    ├─ _apply_token_budget()    ← fits content into remaining budget
    │   ├─ Full file if it fits
    │   ├─ Cut at function boundary if not
    │   └─ Hard truncate as last resort
    └─ FileContext (with symbols, was_truncated flag, token_count)
         │
         ▼
builder.render()
    └─ Formatted string with per-file context + truncation notes
```

**Key insight:** Truncation at a function boundary is much better than truncation at a character limit. If the agent sees a complete function, it understands the interface even if later functions are missing. Mid-function truncation confuses the LLM.

---

## The Correction Loop (`correct_node`)

The hardest failure mode is the agent looping on the same wrong fix. This happens because:
- The LLM has "momentum" toward its initial approach
- Without explicit instruction, it produces variations on the same mistake

Our anti-loop mechanism has two parts:

**1. Error classification** — Before calling the correction LLM, we classify the failure:
```python
class ErrorCategory(StrEnum):
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    LOGIC_ERROR = "logic_error"
    TYPE_ERROR = "type_error"
    FIXTURE_ERROR = "fixture_error"
    TIMEOUT = "timeout"
    PATCH_APPLY_ERROR = "patch_apply_error"
```
This routes to different correction strategies (future: separate prompts per category).

**2. Full history in context** — The correction prompt includes ALL previous failed patches:
```
Previous attempts (DO NOT REPEAT THESE APPROACHES):

### Attempt 1 [logic_error]
Modified: src/utils.py
Diff preview:
+    if user_id is None:
+        return None  # ← this is the approach that failed

### Attempt 2 [logic_error]
...
```
The explicit "DO NOT REPEAT" instruction with concrete examples of what not to do increases second-attempt success rate significantly vs. just sending the latest failure.

---

## Docker Sandbox Security Model

Every line of LLM-generated code is treated as untrusted. The sandbox enforces:

| Control | Value | Why |
|---------|-------|-----|
| `--network disabled` | True | Prevent exfiltration, C2 callback, or unintended side effects |
| `--memory` | 512m | Prevent OOM on host |
| `--cpu-quota` | 50000 (50%) | Prevent CPU starvation of API/workers |
| `--cap-drop ALL` | True | No Linux capabilities (no root-equivalent ops) |
| `--security-opt no-new-privileges` | True | No setuid, no sudo |
| `--read-only` | False (workspace writable) | Tests need to write artifacts |
| `remove=False` + manual cleanup | True | Ensure cleanup even on timeout |

The sandbox image (`swe-agent-sandbox:latest`) is a minimal Python image with only pytest pre-installed. If the coder's patch requires additional packages, we log a warning — in production, common packages should be pre-baked into the sandbox image. Installing packages at runtime would require network access, which we explicitly block.

**Docker-in-Docker note:** Workers mount `/var/run/docker.sock`. In Kubernetes, this is replaced with a Job spawner that creates a new K8s Job per run (see `infra/k8s/sandbox-job-template.yaml`). The Job has `networkPolicy: Deny` applied at the Kubernetes level as a second line of defense.

---

## Database Schema Design

Single table `agent_runs` captures the complete lifecycle of every run. Key fields:

- `state_snapshot` (TEXT/JSON) — serialized `AgentState` at completion. Allows post-mortem debugging without LangSmith.
- `retry_count` — tracked in DB, not just LangGraph state, for analytics queries.
- `started_at` / `completed_at` — both nullable; watchdog uses `started_at IS NOT NULL AND completed_at IS NULL` for stuck-run detection.

The `run_stats` view in `init.sql` computes solve rate, average retries, and average duration per day per repo. This feeds the Grafana dashboard directly via the `swe_agent_readonly` role.

---

## Webhook → Queue → Worker Flow

```
GitHub POST /webhooks/github
    │
    ├─ HMAC-SHA256 validate (constant-time)
    ├─ Event type filter (issues only)
    ├─ Label filter (agent-fix only)
    ├─ Idempotency check (DB: active run for this issue?)
    │
    └─ enqueue_issue_job() → Redis RQ
         │
         └─ RQ Worker: process_issue_job()
              ├─ create_run() in DB
              ├─ get_issue() from GitHub
              ├─ run_agent() → LangGraph
              └─ update_run() in DB + comment on issue
```

**Why RQ and not Celery?** RQ is simpler, has a built-in dashboard, and has fewer moving parts. Celery is more powerful but requires a broker (we already have Redis), a result backend, and more configuration surface area. For this use case (jobs defined by a single function, FIFO processing), RQ is the right tool.

**Why not async task queue (arq, dramatiq)?** We need the jobs to be able to spawn Docker containers synchronously. Async task queues require the job function to be fully async, which conflicts with the blocking Docker SDK calls in `SandboxRunner`. RQ runs jobs in a subprocess, which is the right model here.

---

## Observability Architecture

Three layers:

**1. Structured logs (structlog)**
Every log line is JSON with `run_id`, `issue_number`, `repo`, `status`. Queryable in any log aggregator (CloudWatch Logs Insights, Datadog, Splunk).

**2. LangSmith traces**
Every agent run has a full trace showing: which nodes were called, in what order, with what input/output state, how many tokens each LLM call used, and end-to-end latency. This is the primary debugging interface for prompt failures.

**3. Prometheus metrics**
Six custom metrics covering: solve rate, run duration (histogram), retry depth (histogram), token usage, active runs (gauge), HTTP request rate/latency. The Grafana dashboard shows all six with pre-configured thresholds and alert rules.

The three layers are complementary — logs tell you *what* happened, LangSmith tells you *why* the LLM made the decisions it did, and Prometheus tells you *how often* each outcome occurs over time.

---

## Scaling Path

The system is designed to scale to production load without code changes:

| Component | Current | Production (AWS) |
|-----------|---------|-----------------|
| API | Docker Compose | ECS Fargate + ALB |
| Workers | 3x RQ workers | ECS Fargate, auto-scale 3-50 |
| Queue | Redis (single node) | ElastiCache Redis (Multi-AZ) |
| DB | Postgres (single node) | RDS PostgreSQL Multi-AZ |
| Sandbox | Docker-in-Docker | Fargate task per run (no DinD) |
| Secrets | .env file | AWS Secrets Manager + IRSA |

The only code change required for AWS is replacing the Docker sandbox runner (`src/tools/sandbox.py`) with an ECS task spawner. The interface (`SandboxRunner` context manager) stays identical — only the implementation changes.
