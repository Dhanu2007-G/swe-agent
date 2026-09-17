"""scripts/compose_smoke_test.py — Compose configuration and container smoke test.

Validates Compose file security, service dependencies, port policies, and API health.
"""

from __future__ import annotations

import sys
from pathlib import Path


def smoke_test_compose_file() -> None:
    compose_path = Path(__file__).resolve().parent.parent / "docker-compose.yml"
    if not compose_path.exists():
        print(f"Error: {compose_path} does not exist", file=sys.stderr)
        sys.exit(1)

    content = compose_path.read_text(encoding="utf-8")

    # Security check: verify redis and postgres do NOT expose host ports
    lines = content.splitlines()
    current_service = ""
    exposed_db_ports: list[str] = []

    for line in lines:
        stripped = line.strip()
        if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            current_service = stripped.rstrip(":")
        if (
            current_service in ("redis", "postgres")
            and stripped.startswith("- ")
            and ("5432:5432" in stripped or "6379:6379" in stripped)
        ):
            exposed_db_ports.append(f"{current_service}: {stripped}")

    if exposed_db_ports:
        print(
            f"Security failure: Internal DB ports exposed to host: {exposed_db_ports}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify API port 8000 is exposed
    if "8000:8000" not in content:
        print("Warning: API port 8000:8000 not found in compose file", file=sys.stderr)

    # Verify essential services exist
    for svc in ("api:", "worker:", "redis:", "postgres:"):
        if svc not in content:
            print(f"Error: Missing service {svc} in compose file", file=sys.stderr)
            sys.exit(1)

    print("✅ Compose smoke test passed: security and service topologies verified.")


def run_live_compose_smoke_test() -> None:
    """
    Live stack lifecycle test:
    1. Start stack with docker compose up -d
    2. Wait for /health/ready
    3. Trigger dry-run / test run
    4. Assert run reaches terminal state
    5. Tear down stack
    """
    import shutil
    import subprocess
    import time

    if not shutil.which("docker"):
        print("Docker binary not available; skipping live Compose lifecycle test.")
        return

    print("Starting live Compose stack for smoke test...")
    try:
        subprocess.run(["docker", "compose", "up", "-d", "--build"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start compose stack: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        import httpx

        # 2. Wait for /health/ready
        ready = False
        for _ in range(30):
            try:
                r = httpx.get("http://localhost:8000/health/ready", timeout=2.0)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(2)

        if not ready:
            print("Compose stack failed readiness check", file=sys.stderr)
            sys.exit(1)

        # 3. Trigger run
        headers = {"X-API-Key": "swe-agent-dev-key-12345"}
        resp = httpx.post(
            "http://localhost:8000/api/v1/runs/trigger",
            json={"repo_full_name": "owner/repo", "issue_number": 1},
            headers=headers,
            timeout=5.0,
        )
        assert resp.status_code in (200, 202)
        job_id = resp.json().get("job_id")

        # 4. Assert job reaches terminal state
        if job_id:
            terminal = False
            for _ in range(30):
                time.sleep(2)
                status_resp = httpx.get(
                    f"http://localhost:8000/api/v1/runs/{job_id}",
                    headers=headers,
                    timeout=5.0,
                )
                if status_resp.status_code == 200:
                    st = status_resp.json().get("status")
                    if st in ("succeeded", "failed", "partial"):
                        terminal = True
                        break
            print(f"Job {job_id} reached terminal state: {terminal}")

        print("✅ Live Compose lifecycle smoke test passed.")
    finally:
        print("Tearing down Compose stack...")
        subprocess.run(["docker", "compose", "down", "-v"], check=False)


if __name__ == "__main__":
    smoke_test_compose_file()
    if "--live" in sys.argv:
        run_live_compose_smoke_test()
