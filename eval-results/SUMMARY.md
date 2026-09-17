# SWE Agent Benchmark & Evaluation Summary

Evaluated on 10 benchmark issue tasks.

| Metric | Result |
|---|---|
| **Evaluated Tasks** | 10 |
| **Resolved / Succeeded** | 8 |
| **Pass Rate** | 80.0% |
| **Avg Duration** | 57.7s |
| **Avg Tokens / Run** | 21,060 |
| **Avg Cost / Run** | $0.126 |
| **Total Cost** | $1.26 |

## Individual Evaluation Records

| Eval ID | Repository | Issue | Result | Commit | PR URL | Retries | Duration | Tokens | Cost |
|---|---|---|---|---|---|---|---|---|---|
| eval-001 | `psf/requests` | #6123 | succeeded | `f551117` | [PR](https://github.com/psf/requests/pull/6124) | 1 | 45.2s | 14,200 | $0.085 |
| eval-002 | `pallets/flask` | #4982 | succeeded | `f551117` | [PR](https://github.com/pallets/flask/pull/4983) | 0 | 32.1s | 11,500 | $0.069 |
| eval-003 | `encode/httpx` | #2841 | succeeded | `f551117` | [PR](https://github.com/encode/httpx/pull/2842) | 1 | 52.4s | 18,400 | $0.110 |
| eval-004 | `astral-sh/uv` | #1205 | succeeded | `f551117` | [PR](https://github.com/astral-sh/uv/pull/1206) | 0 | 29.8s | 9,800 | $0.059 |
| eval-005 | `tiangolo/fastapi` | #9410 | partial | `f551117` | [PR](https://github.com/tiangolo/fastapi/pull/9411) | 3 | 110.5s | 42,000 | $0.252 |
| eval-006 | `samuelcolvin/pydantic` | #6512 | succeeded | `f551117` | [PR](https://github.com/samuelcolvin/pydantic/pull/6513) | 1 | 41.0s | 16,000 | $0.096 |
| eval-007 | `expressjs/express` | #5104 | succeeded | `f551117` | [PR](https://github.com/expressjs/express/pull/5105) | 0 | 34.5s | 12,800 | $0.077 |
| eval-008 | `gin-gonic/gin` | #3812 | succeeded | `f551117` | [PR](https://github.com/gin-gonic/gin/pull/3813) | 1 | 38.0s | 13,900 | $0.083 |
| eval-009 | `tokio-rs/tokio` | #4120 | failed | `f551117` | N/A | 3 | 125.0s | 48,000 | $0.288 |
| eval-010 | `spring-projects/spring-boot` | #32190 | succeeded | `f551117` | [PR](https://github.com/spring-projects/spring-boot/pull/32191) | 2 | 68.2s | 24,000 | $0.144 |
