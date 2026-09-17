# Security Policy

## Reporting Security Issues

We take the security of SWE Agent seriously. If you discover a potential security vulnerability, please do **NOT** create a public GitHub issue.

Instead, please send a responsible disclosure report to:
`security@swe-agent.ai` (or create a private GitHub Security Advisory).

Please include:
- A clear description of the vulnerability.
- Steps or a minimal proof-of-concept to reproduce the behavior.
- The potential impact on host environments, containers, or secrets.

You will receive an acknowledgment within 24 hours and regular status updates until a patch is released.

---

## Threat Model & Security Controls

SWE Agent executes untrusted code and processes arbitrary issue descriptions from GitHub. The security architecture is designed to enforce defense-in-depth across the following vectors:

### 1. Arbitrary Code Execution in Sandboxes
- **Ephemeral Isolation**: Every test execution runs inside an isolated container with an ephemeral per-run workspace.
- **Root Filesystem Protection**: The root filesystem is strictly mounted **read-only** (`read_only: true`). Only the isolated `/workspace` directory and a small `/tmp` `tmpfs` are writeable.
- **Capability Dropping**: All Linux kernel capabilities are dropped (`cap_drop: ["ALL"]`).
- **Privilege Escalation**: `no-new-privileges` is enforced. Processes cannot gain privileges via setuid/setgid binaries.
- **Non-Root Execution**: Container processes run under an unprivileged user (`USER sandboxuser` / UID 1001).
- **Network Isolation**: Outbound container network access is disabled by default (`network_disabled: true`) during patch execution and tests.
- **Resource Quotas**: Hard CPU quotas and memory limits (e.g., 512MB–2GB) prevent denial-of-service or fork bombs.

### 2. Secret Protection & Remote Authentication
- **No Secrets in Git URLs**: Authentication tokens are passed exclusively via HTTP `Authorization` headers (`http.extraheader`) rather than being embedded in Git remote URLs.
- **Log Masking**: Webhook payloads and configuration settings sanitize tokens and HMAC secrets before logging.
- **Secret Management**: Kubernetes manifests use External Secrets Operator (`ClusterSecretStore` / Vault / AWS Secrets Manager) rather than committing sensitive credentials.

### 3. API & Webhook Security
- **HMAC Signature Verification**: Inbound GitHub webhooks are verified using HMAC-SHA256 signatures with `secrets.compare_digest` to prevent timing attacks.
- **Replay Protection**: Webhook deliveries are tracked by UUID in Redis with TTLs to prevent replay attacks.
- **API Authentication**: Public API endpoints require `X-API-Key` headers validated against constant-time hashes.
- **Repository Allowlist**: Only explicitly allowlisted repositories can trigger agent runs.
- **Production Guardrails**: Production environments (`APP_ENV=production`) refuse to start if API authentication is disabled or default test keys are present.

### 4. Database Security
- **Strict Parameterization**: Database interactions use SQLAlchemy ORM with asyncpg. No raw string concatenation or dynamic SQL interpolation is permitted.
- **Single Migration Head**: Alembic schema migrations are validated to ensure a single deterministic linear head.
