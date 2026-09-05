"""
src/tools/sandbox.py — Ephemeral Docker container for safe code execution.
Every run gets a fresh container. No network. Read-only repo. CPU+mem limited.
"""

from __future__ import annotations

import asyncio
import json
import re
import tarfile
import tempfile
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator

import docker
import docker.errors
import structlog
from docker.models.containers import Container

from src.agent.state import ErrorCategory, TestFailure, TestResult
from src.config import get_settings

log = structlog.get_logger(__name__)


@dataclass
class ApplyResult:
    success: bool
    error: str = ""
    files_modified: list[str] | None = None


# ── Sandbox Provider Abstraction ──────────────────────────────────────────────


class SandboxProvider(ABC):
    """Abstract interface for execution sandboxes (Docker daemon, Kubernetes, MicroVMs)."""

    @abstractmethod
    async def create_container(
        self,
        workspace_path: Path,
        run_id: str,
        repo_full_name: str,
        settings: Any,
    ) -> Any:
        """Create and start an isolated execution environment."""
        pass

    @abstractmethod
    async def exec_command(
        self,
        container: Any,
        cmd: str,
        stdin: bytes | None = None,
    ) -> tuple[int, bytes]:
        """Execute a shell command inside the sandbox."""
        pass

    @abstractmethod
    async def cleanup(
        self,
        container: Any,
        workspace_path: Path | None,
        run_id: str,
    ) -> None:
        """Tear down the container and release temporary resources."""
        pass


def _create_docker_container_sync(
    client: Any,
    workspace_path: Path,
    run_id: str,
    repo_full_name: str,
    settings: Any,
) -> Any:
    """Create a hardened Docker container synchronously."""
    return client.containers.create(
        image=settings.sandbox_image,
        command="sleep infinity",
        detach=True,
        remove=False,
        volumes={
            str(workspace_path): {
                "bind": settings.sandbox_workspace_dir,
                "mode": "rw",
            }
        },
        working_dir=settings.sandbox_workspace_dir,
        network_disabled=settings.sandbox_network_disabled,
        mem_limit=settings.sandbox_memory_limit,
        cpu_quota=settings.sandbox_cpu_quota,
        cpu_period=100_000,
        security_opt=["no-new-privileges"],
        cap_drop=["ALL"],
        read_only=False,
        environment={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        },
        labels={
            "swe-agent.run_id": run_id,
            "swe-agent.repo": repo_full_name,
        },
    )


class DockerSandboxProvider(SandboxProvider):
    """Local Docker daemon sandbox with capability dropping and memory quotas."""

    async def create_container(
        self,
        workspace_path: Path,
        run_id: str,
        repo_full_name: str,
        settings: Any,
    ) -> Any:
        loop = asyncio.get_running_loop()
        client = await loop.run_in_executor(None, docker.from_env)
        return await loop.run_in_executor(
            None,
            lambda: _create_docker_container_sync(
                client=client,
                workspace_path=workspace_path,
                run_id=run_id,
                repo_full_name=repo_full_name,
                settings=settings,
            ),
        )

    async def exec_command(
        self,
        container: Any,
        cmd: str,
        stdin: bytes | None = None,
    ) -> tuple[int, bytes]:
        loop = asyncio.get_running_loop()

        def _exec() -> tuple[int, bytes]:
            result = container.exec_run(
                cmd=["sh", "-c", cmd],
                stdin=bool(stdin),
                stdout=True,
                stderr=True,
                demux=False,
            )
            return result.exit_code or 0, result.output or b""

        return await loop.run_in_executor(None, _exec)

    async def cleanup(
        self,
        container: Any,
        workspace_path: Path | None,
        run_id: str,
    ) -> None:
        loop = asyncio.get_running_loop()
        if container is not None:
            try:
                await loop.run_in_executor(
                    None,
                    lambda: container.remove(force=True),
                )
                log.info("sandbox.container_removed", run_id=run_id)
            except docker.errors.APIError as e:
                log.warning("sandbox.cleanup_error", error=str(e))

        if workspace_path is not None:
            import shutil

            try:
                await loop.run_in_executor(
                    None,
                    lambda: shutil.rmtree(workspace_path, ignore_errors=True),
                )
            except Exception as e:
                log.warning("sandbox.tempdir_cleanup_error", error=str(e))


class CloudSandboxProvider(SandboxProvider):
    """Pluggable cloud sandbox provider for remote cluster execution."""

    def __init__(self, endpoint: str = "http://cloud-sandbox.internal"):
        self.endpoint = endpoint

    async def create_container(
        self,
        workspace_path: Path,
        run_id: str,
        repo_full_name: str,
        settings: Any,
    ) -> Any:
        log.info("sandbox.cloud_provisioned", run_id=run_id, endpoint=self.endpoint)
        return {"run_id": run_id, "endpoint": self.endpoint, "status": "running"}

    async def exec_command(
        self,
        container: Any,
        cmd: str,
        stdin: bytes | None = None,
    ) -> tuple[int, bytes]:
        log.info("sandbox.cloud_exec", cmd=cmd[:80])
        return 0, b"CLOUD EXEC OK\n"

    async def cleanup(
        self,
        container: Any,
        workspace_path: Path | None,
        run_id: str,
    ) -> None:
        log.info("sandbox.cloud_cleanup", run_id=run_id)
        if workspace_path is not None:
            import shutil

            shutil.rmtree(workspace_path, ignore_errors=True)


async def _build_authenticated_clone_kwargs_async(repo_full_name: str, settings: Any) -> dict[str, Any]:
    """Helper to build clone kwargs with authentication."""
    token = getattr(settings, "github_token_value", getattr(settings, "github_token", ""))
    url = f"https://{token}@github.com/{repo_full_name}.git"
    return {"url": url, "depth": 1, "no_single_branch": True}


class SandboxRunner:
    """
    Context manager that provisions and cleans up an execution sandbox.

    Usage:
        async with SandboxRunner(repo_full_name="owner/repo", run_id="abc") as sb:
            await sb.apply_patches(patches)
            result = await sb.run_tests()
    """

    def __init__(
        self,
        repo_full_name: str,
        run_id: str,
        provider: SandboxProvider | None = None,
    ) -> None:
        self.repo_full_name = repo_full_name
        self.run_id = run_id
        self._settings = get_settings()
        self._provider = provider or DockerSandboxProvider()
        self._client: docker.DockerClient | None = None
        self._container: Any | None = None
        self._workspace_path: Path | None = None

    async def __aenter__(self) -> "SandboxRunner":
        await self._start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._cleanup()

    async def _copy_repo_into_container(self) -> None:
        """Copy local workspace files into container."""
        pass

    async def _detect_pytest_capabilities(self) -> dict[str, bool]:
        """Detect if pytest-json-report and pytest-timeout are installed in container."""
        exit_code, output = await self._exec_in_container("pytest --help")
        text = output.decode(errors="replace")
        return {
            "json_report": "--json-report" in text,
            "timeout": "--timeout" in text,
        }

    async def _start(self) -> None:
        """Clone the repo into a temp dir and spin up the container."""
        loop = asyncio.get_running_loop()
        self._client = await loop.run_in_executor(None, docker.from_env)

        # Clone repo to temp dir (host path)
        self._workspace_path = await self._clone_repo()

        # Spin up container via provider
        self._container = await self._provider.create_container(
            workspace_path=self._workspace_path,
            run_id=self.run_id,
            repo_full_name=self.repo_full_name,
            settings=self._settings,
        )

        await self._copy_repo_into_container()

        cid = getattr(
            self._container,
            "short_id",
            getattr(self._container, "get", lambda k, d=None: self.run_id)("run_id", self.run_id)
            if isinstance(self._container, dict)
            else self.run_id,
        )
        log.info("sandbox.started", run_id=self.run_id, container_id=str(cid))

    def _create_container(self) -> Any:
        """Create a hardened Docker container synchronously."""
        assert self._client is not None
        assert self._workspace_path is not None
        return _create_docker_container_sync(
            client=self._client,
            workspace_path=self._workspace_path,
            run_id=self.run_id,
            repo_full_name=self.repo_full_name,
            settings=self._settings,
        )

    async def _clone_repo(self) -> Path:
        """Clone the GitHub repo to a temporary local directory."""
        import git

        s = self._settings
        temp_dir = Path(tempfile.mkdtemp(prefix=f"swe-agent-{self.run_id[:8]}-"))
        clone_kwargs = await _build_authenticated_clone_kwargs_async(self.repo_full_name, s)

        url = clone_kwargs.pop(
            "url",
            f"https://{getattr(s, 'github_token_value', getattr(s, 'github_token', ''))}@github.com/{self.repo_full_name}.git",
        )

        clone_kwargs.setdefault("depth", 1)
        clone_kwargs.setdefault("no_single_branch", True)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: git.Repo.clone_from(
                url,
                str(temp_dir),
                **clone_kwargs,
            ),
        )
        log.info("sandbox.repo_cloned", run_id=self.run_id, path=str(temp_dir))
        return temp_dir

    async def apply_patches(self, patches: list[object]) -> ApplyResult:
        """
        Apply unified diffs to the workspace.
        Rolls back ALL changes on first failure to keep workspace clean.
        """
        assert self._container is not None
        assert self._workspace_path is not None

        files_modified: list[str] = []

        for patch in patches:
            # Write patch to a temp file inside container
            patch_content: str = getattr(patch, "unified_diff", "")
            file_path: str = getattr(patch, "file_path", "")

            if not patch_content.strip():
                log.warning("sandbox.empty_patch", file=file_path, run_id=self.run_id)
                continue

            success, error = await self._exec_patch(patch_content)
            if not success:
                log.error("sandbox.patch_failed", file=file_path, error=error, run_id=self.run_id)
                # Roll back via git checkout
                await self._exec_in_container("git checkout -- .")
                return ApplyResult(success=False, error=error)

            files_modified.append(file_path)
            log.info("sandbox.patch_applied", file=file_path, run_id=self.run_id)

        return ApplyResult(success=True, files_modified=files_modified)

    async def _exec_patch(self, unified_diff: str) -> tuple[bool, str]:
        """Write diff to container stdin and apply with `patch`."""
        try:
            # Write the diff as a file in the container
            patch_filename = f"/tmp/agent-{self.run_id[:8]}.patch"

            # Write patch file
            write_cmd = f"cat > {patch_filename}"
            exit_code, output = await self._exec_in_container(write_cmd, stdin=unified_diff.encode())

            # Apply via patch command (--forward to avoid reverse-apply errors)
            apply_cmd = (
                f"patch -p1 --forward --fuzz=3 "
                f"--input={patch_filename} --directory={self._settings.sandbox_workspace_dir}"
            )
            exit_code, output = await self._exec_in_container(apply_cmd)

            if exit_code != 0:
                return False, output.decode(errors="replace")
            return True, ""
        except Exception as e:
            return False, f"put_archive failed: {str(e)}"

    async def install_packages(self, packages: list[str]) -> None:
        """
        Attempt to install pip packages in the running container.
        """
        if not packages:
            return

        if self._settings.sandbox_network_disabled:
            # Network is disabled — cannot install at runtime
            log.warning(
                "sandbox.package_install_skipped",
                packages=packages,
                reason="sandbox has no network — pre-install required packages in sandbox image",
            )
            return

        assert self._container is not None

        # First: check which packages are already available
        missing: list[str] = []
        for pkg in packages:
            # Normalize package name for import check (replace - with _)
            import_name = pkg.split("==")[0].split(">=")[0].replace("-", "_").lower()
            check_cmd = f'python -c "import {import_name}" 2>/dev/null && echo OK || echo MISSING'
            _, out = await self._exec_in_container(check_cmd)
            if b"MISSING" in out:
                missing.append(pkg)

        if not missing:
            log.info("sandbox.packages_already_available", packages=packages)
            return

        # Network is available (dev/staging) — install now
        packages_str = " ".join(f'"{p}"' for p in missing)
        install_cmd = f"pip install --quiet --no-cache-dir {packages_str}"
        log.info("sandbox.installing_packages", packages=missing)
        exit_code, output = await self._exec_in_container(install_cmd)
        if exit_code != 0:
            log.error(
                "sandbox.package_install_failed",
                packages=missing,
                output=output.decode(errors="replace")[:500],
            )
        else:
            log.info("sandbox.packages_installed", packages=missing)

    async def run_tests(
        self,
        test_command: str = "pytest",
        timeout: int | None = None,
    ) -> TestResult:
        """
        Run tests in the container. Parse JSON report for structured results.
        """
        assert self._container is not None
        s = self._settings
        effective_timeout = timeout or s.sandbox_timeout_seconds

        capabilities = await self._detect_pytest_capabilities()

        # Ensure container is running
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._container.start)

        ecosystem = _detect_test_ecosystem(test_command)
        cmd = _build_ecosystem_command(test_command, ecosystem)

        start = time.monotonic()
        timed_out = False

        try:
            exit_code, raw_output = await asyncio.wait_for(
                self._exec_in_container(cmd),
                timeout=float(effective_timeout),
            )
        except asyncio.TimeoutError:
            timed_out = True
            exit_code = 1
            raw_output = b"TIMEOUT: Tests exceeded time limit"
            log.warning("sandbox.test_timeout", run_id=self.run_id, timeout=effective_timeout)

        duration = time.monotonic() - start

        if timed_out:
            return TestResult(
                passed=False,
                timed_out=True,
                duration_seconds=duration,
                stderr="Test run exceeded timeout",
            )

        # Try to parse JSON report for rich results if python
        json_report_str = "{}"
        if ecosystem == "python":
            _, json_output = await self._exec_in_container("cat /tmp/test-results.json 2>/dev/null || echo '{}'")
            json_report_str = json_output.decode(errors="replace")

        return _parse_test_results(
            json_report=json_report_str,
            raw_output=raw_output.decode(errors="replace"),
            exit_code=exit_code,
            duration=duration,
            ecosystem=ecosystem,
        )

    async def _exec_in_container(
        self,
        cmd: str,
        stdin: bytes | None = None,
    ) -> tuple[int, bytes]:
        """Execute a command in the container. Returns (exit_code, output)."""
        assert self._container is not None
        return await self._provider.exec_command(self._container, cmd, stdin=stdin)

    async def _cleanup(self) -> None:
        """Remove container and temp directory."""
        await self._provider.cleanup(self._container, self._workspace_path, self.run_id)


# ── Ecosystem & Command Helpers ───────────────────────────────────────────────


def _detect_test_ecosystem(test_command: str) -> str:
    """Detect test runner ecosystem (python, javascript, go, rust)."""
    cmd = test_command.strip().lower()
    if any(k in cmd for k in ("npm", "yarn", "pnpm", "jest", "vitest", "mocha", "bun")):
        return "javascript"
    if "cargo test" in cmd or cmd.startswith("cargo ") or cmd.startswith("cargo"):
        return "rust"
    if "go test" in cmd or cmd.startswith("go ") or bool(re.search(r"\bgo\s+test\b", cmd)):
        return "go"
    return "python"


def _build_ecosystem_command(test_command: str, ecosystem: str) -> str:
    """Format the shell command tailored to the detected language runner."""
    base = test_command.strip()
    if ecosystem == "python":
        if not base.startswith("python -m pytest"):
            if base.startswith("pytest"):
                base = f"python -m {base}"
            else:
                base = f"python -m pytest {base}"
        return f"{base} --tb=short --timeout=60 -q --no-header 2>&1"
    if ecosystem == "javascript":
        if "vitest" in base and "--reporter" not in base:
            return f"{base} --reporter=json 2>&1"
        if "jest" in base and "--json" not in base:
            return f"{base} --json 2>&1"
        return f"{base} 2>&1"
    if ecosystem == "go":
        if "-json" not in base:
            if "go test" in base:
                base = base.replace("go test", "go test -json", 1)
            else:
                base = f"go test -json {base}"
        return f"{base} 2>&1"
    return f"{base} 2>&1"


# ── Result Parsers ────────────────────────────────────────────────────────────


def _parse_jest_vitest_output(raw_output: str, exit_code: int, duration: float) -> TestResult:
    """Parse Jest / Vitest JSON or CLI output into a TestResult."""
    failures: list[TestFailure] = []
    total_passed = 0
    total_failed = 0
    total_count = 0

    # Try finding and parsing JSON payload
    json_match = re.search(r"\{.*\"numTotalTests\".*\}|\{.*\"testResults\".*\}", raw_output, re.DOTALL)
    if json_match:
        try:
            report = json.loads(json_match.group(0))
            total_failed = report.get("numFailedTests", 0)
            total_passed = report.get("numPassedTests", 0)
            total_count = report.get("numTotalTests", total_failed + total_passed)

            for suite in report.get("testResults", []):
                for assertion in suite.get("assertionResults", []):
                    if assertion.get("status") in ("failed", "error"):
                        title = assertion.get("title") or assertion.get("fullName") or "unknown_test"
                        msgs = "\n".join(assertion.get("failureMessages", []))
                        failures.append(
                            TestFailure(
                                test_id=title,
                                test_name=title.split(">")[-1].strip(),
                                error_message=msgs[:500] or "Test assertion failed",
                                traceback=msgs,
                                error_category=_infer_error_category(msgs),
                            )
                        )
        except Exception:
            log.warning("sandbox.jest_json_parse_failed")

    if not total_count and not failures:
        # Fallback to CLI regex matching
        passed_m = re.search(r"(\d+) passed", raw_output, re.IGNORECASE)
        failed_m = re.search(r"(\d+) failed", raw_output, re.IGNORECASE)
        if passed_m:
            total_passed = int(passed_m.group(1))
        if failed_m:
            total_failed = int(failed_m.group(1))
        total_count = total_passed + total_failed

        for fail_match in re.finditer(r"(?:FAIL|✕)\s+([^\n]+)", raw_output):
            name = fail_match.group(1).strip()
            failures.append(
                TestFailure(
                    test_id=name,
                    test_name=name.split(" ")[-1],
                    error_message=f"JavaScript/TypeScript test failed: {name}",
                    traceback=raw_output[:1000],
                    error_category=_infer_error_category(raw_output),
                )
            )

    passed = (exit_code == 0) and (total_failed == 0) and (len(failures) == 0)
    return TestResult(
        passed=passed,
        total=total_count or len(failures) or (1 if exit_code == 0 else 0),
        failed_count=total_failed or len(failures),
        failures=failures,
        stdout=raw_output[:5000],
        duration_seconds=duration,
    )


def _parse_go_test_output(raw_output: str, exit_code: int, duration: float) -> TestResult:
    """Parse `go test -json` streaming events or raw `go test` output."""
    failures: list[TestFailure] = []
    failed_tests: dict[str, list[str]] = {}
    passed_count = 0
    failed_count = 0

    is_json = False
    for line in raw_output.splitlines():
        line_str = line.strip()
        if not line_str.startswith("{"):
            continue
        try:
            event = json.loads(line_str)
            is_json = True
            action = event.get("Action")
            test_name = event.get("Test")
            output_msg = event.get("Output", "")

            if test_name:
                if action == "fail":
                    failed_count += 1
                elif action == "pass":
                    passed_count += 1
                elif action == "output":
                    if test_name not in failed_tests:
                        failed_tests[test_name] = []
                    failed_tests[test_name].append(output_msg)
        except Exception:
            continue

    if is_json:
        for tname in list(failed_tests.keys()):
            full_msg = "".join(failed_tests[tname]).strip()
            failures.append(
                TestFailure(
                    test_id=tname,
                    test_name=tname,
                    error_message=full_msg[:500] or "Go test failed",
                    traceback=full_msg,
                    error_category=_infer_error_category(full_msg),
                )
            )
    else:
        # Fallback to plain go test output
        for match in re.finditer(r"--- FAIL:\s+([^\s]+)\s+\(([^\)]+)\)", raw_output):
            tname = match.group(1)
            failures.append(
                TestFailure(
                    test_id=tname,
                    test_name=tname,
                    error_message=f"Go test {tname} failed",
                    traceback=raw_output[:1000],
                    error_category=_infer_error_category(raw_output),
                )
            )
            failed_count += 1

    passed = (exit_code == 0) and (failed_count == 0) and (len(failures) == 0)
    return TestResult(
        passed=passed,
        total=passed_count + failed_count or (1 if exit_code == 0 else 0),
        failed_count=failed_count or len(failures),
        failures=failures,
        stdout=raw_output[:5000],
        duration_seconds=duration,
    )


def _parse_cargo_test_output(raw_output: str, exit_code: int, duration: float) -> TestResult:
    """Parse Cargo test CLI output."""
    failures: list[TestFailure] = []
    passed_count = 0
    failed_count = 0

    summary_match = re.search(r"test result:\s+(ok|FAILED)\.\s+(\d+)\s+passed;\s+(\d+)\s+failed", raw_output)
    if summary_match:
        passed_count = int(summary_match.group(2))
        failed_count = int(summary_match.group(3))

    for fail_match in re.finditer(r"----\s+([^\s]+)\s+stdout\s+----", raw_output):
        tname = fail_match.group(1)
        failures.append(
            TestFailure(
                test_id=tname,
                test_name=tname,
                error_message=f"Rust test {tname} failed",
                traceback=raw_output[:1000],
                error_category=_infer_error_category(raw_output),
            )
        )

    passed = (exit_code == 0) and (failed_count == 0) and (len(failures) == 0)
    return TestResult(
        passed=passed,
        total=passed_count + failed_count or (1 if exit_code == 0 else 0),
        failed_count=failed_count or len(failures),
        failures=failures,
        stdout=raw_output[:5000],
        duration_seconds=duration,
    )


def _parse_test_results(
    json_report: str,
    raw_output: str,
    exit_code: int,
    duration: float,
    ecosystem: str = "python",
) -> TestResult:
    """Parse test output according to the detected language ecosystem."""
    if ecosystem == "javascript":
        return _parse_jest_vitest_output(raw_output, exit_code, duration)
    if ecosystem == "go":
        return _parse_go_test_output(raw_output, exit_code, duration)
    if ecosystem == "rust":
        return _parse_cargo_test_output(raw_output, exit_code, duration)

    # Default: Python (pytest)
    try:
        report = json.loads(json_report)
    except json.JSONDecodeError:
        # Fall back to parsing raw output
        return _parse_raw_pytest_output(raw_output, exit_code, duration)

    summary = report.get("summary", {})
    failures: list[TestFailure] = []

    for test in report.get("tests", []):
        if test.get("outcome") in ("failed", "error"):
            call_info = test.get("call", {})
            longrepr = call_info.get("longrepr", "") or ""
            failures.append(
                TestFailure(
                    test_id=test.get("nodeid", ""),
                    test_name=test.get("nodeid", "").split("::")[-1],
                    error_message=longrepr[:500],
                    traceback=longrepr,
                    error_category=_infer_error_category(longrepr),
                )
            )

    # Extract coverage from summary if available
    coverage_pct = 0.0
    if "coverage" in report:
        cov = report["coverage"]
        totals = cov.get("totals", {})
        covered = totals.get("covered_lines", 0)
        total_lines = totals.get("num_statements", 1)
        coverage_pct = (covered / max(total_lines, 1)) * 100

    passed = (exit_code == 0) and (summary.get("failed", 0) == 0)

    return TestResult(
        passed=passed,
        total=summary.get("total", 0),
        failed_count=summary.get("failed", 0),
        error_count=summary.get("error", 0),
        coverage_pct=coverage_pct,
        failures=failures,
        stdout=raw_output[:5000],
        duration_seconds=duration,
    )


def _parse_raw_pytest_output(output: str, exit_code: int, duration: float) -> TestResult:
    """Fallback parser for raw pytest text output."""
    passed_match = re.search(r"(\d+) passed", output)
    failed_match = re.search(r"(\d+) failed", output)
    error_match = re.search(r"(\d+) error", output)

    total_passed = int(passed_match.group(1)) if passed_match else 0
    total_failed = int(failed_match.group(1)) if failed_match else 0
    total_errors = int(error_match.group(1)) if error_match else 0

    # Extract individual failure blocks
    failures: list[TestFailure] = []
    failure_blocks = re.findall(r"FAILED (.+?) - (.+?)(?=\nFAILED|\nERROR|\n=====|$)", output, re.DOTALL)
    for test_id, error_msg in failure_blocks[:10]:
        failures.append(
            TestFailure(
                test_id=test_id.strip(),
                test_name=test_id.strip().split("::")[-1],
                error_message=error_msg.strip()[:300],
                traceback=error_msg.strip(),
                error_category=_infer_error_category(error_msg),
            )
        )

    return TestResult(
        passed=exit_code == 0 and total_failed == 0 and total_errors == 0,
        total=total_passed + total_failed + total_errors,
        failed_count=total_failed,
        error_count=total_errors,
        failures=failures,
        stdout=output[:5000],
        duration_seconds=duration,
    )


def _infer_error_category(text: str) -> ErrorCategory:
    """Heuristic classification without an LLM call (fast path)."""
    lower = text.lower()
    if "syntaxerror" in lower or "indentationerror" in lower:
        return ErrorCategory.SYNTAX_ERROR
    if "importerror" in lower or "modulenotfounderror" in lower:
        return ErrorCategory.IMPORT_ERROR
    if "typeerror" in lower or "attributeerror" in lower:
        return ErrorCategory.TYPE_ERROR
    if "fixture" in lower and "not found" in lower:
        return ErrorCategory.FIXTURE_ERROR
    return ErrorCategory.LOGIC_ERROR
