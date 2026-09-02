from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import docker.errors
import pytest

from src.agent.state import FilePatch


def make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        sandbox_image="swe-agent-sandbox:latest",
        sandbox_workspace_dir="/workspace",
        sandbox_network_disabled=True,
        sandbox_memory_limit="512m",
        sandbox_cpu_quota=50000,
        sandbox_timeout_seconds=120,
    )


@contextmanager
def case_dir(prefix: str) -> Path:
    root = (
        Path(__file__).resolve().parents[2]
        / "test_out"
        / "sandbox_cases"
        / f"{prefix}-{uuid4().hex}"
    )
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class TestSandboxRunnerLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager_starts_and_cleans_up(self) -> None:
        from src.tools.sandbox import SandboxRunner

        with patch("src.tools.sandbox.get_settings", return_value=make_settings()):
            runner = SandboxRunner("owner/repo", "run-123")

        with (
            patch.object(runner, "_start", new=AsyncMock()) as mock_start,
            patch.object(runner, "_cleanup", new=AsyncMock()) as mock_cleanup,
        ):
            async with runner as active:
                assert active is runner

        mock_start.assert_awaited_once()
        mock_cleanup.assert_awaited_once()

    def test_create_container_uses_hardened_settings(self) -> None:
        from src.tools.sandbox import SandboxRunner

        client = MagicMock()
        container = MagicMock()
        client.containers.create.return_value = container

        with patch("src.tools.sandbox.get_settings", return_value=make_settings()):
            runner = SandboxRunner("owner/repo", "run-123")

        runner._client = client
        runner._workspace_path = Path("workspace")

        created = runner._create_container()

        assert created is container
        client.containers.create.assert_called_once()
        kwargs = client.containers.create.call_args.kwargs
        assert kwargs["network_disabled"] is True
        assert kwargs["cap_drop"] == ["ALL"]
        assert kwargs["labels"]["swe-agent.run_id"] == "run-123"

    @pytest.mark.asyncio
    async def test_start_initializes_client_workspace_and_container(self) -> None:
        from src.tools.sandbox import SandboxRunner

        loop = MagicMock()

        async def run_in_executor(_: object, fn: object) -> object:
            return fn()

        loop.run_in_executor = AsyncMock(side_effect=run_in_executor)
        container = MagicMock()
        container.short_id = "abc123"

        mock_client = MagicMock()
        with (
            patch("src.tools.sandbox.get_settings", return_value=make_settings()),
            patch("src.tools.sandbox.asyncio.get_running_loop", return_value=loop),
            patch("src.tools.sandbox.docker.from_env", return_value=mock_client),
            patch.object(
                SandboxRunner,
                "_clone_repo",
                new=AsyncMock(return_value=Path("workspace")),
            ),
            patch.object(SandboxRunner, "_copy_repo_into_container", new=AsyncMock()),
            patch("src.tools.sandbox.log.info") as log_info,
        ):
            runner = SandboxRunner("owner/repo", "run-123")
            runner._provider.create_container = AsyncMock(return_value=container)
            await runner._start()

        assert runner._client == mock_client
        assert runner._workspace_path == Path("workspace")
        assert runner._container is container
        log_info.assert_called_once_with(
            "sandbox.started",
            run_id="run-123",
            container_id="abc123",
        )


class TestSandboxRunnerExecution:
    @pytest.mark.asyncio
    async def test_clone_repo_uses_authenticated_clone_kwargs(self) -> None:
        from src.tools.sandbox import SandboxRunner

        loop = MagicMock()

        async def run_in_executor(_: object, fn: object) -> None:
            fn()

        loop.run_in_executor = AsyncMock(side_effect=run_in_executor)

        with (
            case_dir("clone") as root,
            patch("src.tools.sandbox.get_settings", return_value=make_settings()),
            patch("src.tools.sandbox.tempfile.mkdtemp", return_value=str(root)),
            patch(
                "src.tools.sandbox._build_authenticated_clone_kwargs_async",
                new_callable=AsyncMock,
                return_value={
                    "url": "https://github.com/owner/repo.git",
                    "multi_options": ["-c", "header"],
                },
            ),
            patch("src.tools.sandbox.asyncio.get_running_loop", return_value=loop),
            patch("git.Repo.clone_from") as mock_clone,
        ):
            runner = SandboxRunner("owner/repo", "run-123")
            path = await runner._clone_repo()

        assert path == root
        mock_clone.assert_called_once_with(
            "https://github.com/owner/repo.git",
            str(root),
            depth=1,
            no_single_branch=True,
            multi_options=["-c", "header"],
        )

    @pytest.mark.asyncio
    async def test_apply_patches_skips_empty_and_tracks_modified_files(self) -> None:
        from src.tools.sandbox import SandboxRunner

        with patch("src.tools.sandbox.get_settings", return_value=make_settings()):
            runner = SandboxRunner("owner/repo", "run-123")

        runner._container = MagicMock()
        runner._workspace_path = Path("workspace")

        patches = [
            FilePatch(
                file_path="src/a.py",
                unified_diff="",
                change_type="modify",
                description="empty",
            ),
            FilePatch(
                file_path="src/b.py",
                unified_diff="@@ -1 +1 @@\n-print('bad')\n+print('good')\n",
                change_type="modify",
                description="real",
            ),
        ]

        with patch.object(
            runner,
            "_exec_patch",
            new=AsyncMock(return_value=(True, "")),
        ):
            result = await runner.apply_patches(patches)

        assert result.success is True
        assert result.files_modified == ["src/b.py"]

    @pytest.mark.asyncio
    async def test_apply_patches_rolls_back_on_first_failure(self) -> None:
        from src.tools.sandbox import SandboxRunner

        with patch("src.tools.sandbox.get_settings", return_value=make_settings()):
            runner = SandboxRunner("owner/repo", "run-123")

        runner._container = MagicMock()
        runner._workspace_path = Path("workspace")

        patch_file = FilePatch(
            file_path="src/a.py",
            unified_diff="@@ -1 +1 @@\n-print('bad')\n+print('good')\n",
            change_type="modify",
            description="broken",
        )

        with (
            patch.object(
                runner,
                "_exec_patch",
                new=AsyncMock(return_value=(False, "patch failed")),
            ),
            patch.object(
                runner,
                "_exec_in_container",
                new=AsyncMock(return_value=(0, b"")),
            ) as mock_exec,
        ):
            result = await runner.apply_patches([patch_file])

        assert result.success is False
        assert result.error == "patch failed"
        mock_exec.assert_awaited_once_with("git checkout -- .")

    @pytest.mark.asyncio
    async def test_exec_patch_reports_put_archive_failures(self) -> None:
        from src.tools.sandbox import SandboxRunner

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(side_effect=RuntimeError("copy failed"))

        with patch("src.tools.sandbox.get_settings", return_value=make_settings()):
            runner = SandboxRunner("owner/repo", "run-123")

        runner._container = MagicMock()
        with patch("src.tools.sandbox.asyncio.get_running_loop", return_value=loop):
            success, error = await runner._exec_patch("diff --git a/x b/x")

        assert success is False
        assert "put_archive failed" in error

    @pytest.mark.asyncio
    async def test_exec_patch_reports_patch_command_failures(self) -> None:
        from src.tools.sandbox import SandboxRunner

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(return_value=None)

        with patch("src.tools.sandbox.get_settings", return_value=make_settings()):
            runner = SandboxRunner("owner/repo", "run-123")

        runner._container = MagicMock()

        with (
            patch("src.tools.sandbox.asyncio.get_running_loop", return_value=loop),
            patch.object(
                runner,
                "_exec_in_container",
                new=AsyncMock(return_value=(1, b"patch failed")),
            ),
        ):
            success, error = await runner._exec_patch("diff --git a/x b/x")

        assert success is False
        assert error == "patch failed"

    @pytest.mark.asyncio
    async def test_exec_patch_returns_success_when_patch_applies(self) -> None:
        from src.tools.sandbox import SandboxRunner

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(return_value=None)

        with patch("src.tools.sandbox.get_settings", return_value=make_settings()):
            runner = SandboxRunner("owner/repo", "run-123")

        runner._container = MagicMock()

        with (
            patch("src.tools.sandbox.asyncio.get_running_loop", return_value=loop),
            patch.object(
                runner,
                "_exec_in_container",
                new=AsyncMock(return_value=(0, b"applied")),
            ),
        ):
            success, error = await runner._exec_patch("diff --git a/x b/x")

        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_install_packages_logs_skipped_install(self) -> None:
        from src.tools.sandbox import SandboxRunner

        with (
            patch("src.tools.sandbox.get_settings", return_value=make_settings()),
            patch("src.tools.sandbox.log.warning") as log_warning,
        ):
            runner = SandboxRunner("owner/repo", "run-123")
            await runner.install_packages(["httpx", "orjson"])

        log_warning.assert_called_once_with(
            "sandbox.package_install_skipped",
            packages=["httpx", "orjson"],
            reason="sandbox has no network — pre-install required packages in sandbox image",
        )

    @pytest.mark.asyncio
    async def test_run_tests_returns_timeout_result(self) -> None:
        from src.tools.sandbox import SandboxRunner

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(return_value=None)

        async def timeout_wait_for(
            coro: object,
            timeout: float | None = None,
        ) -> object:
            coro.close()
            raise TimeoutError

        with patch("src.tools.sandbox.get_settings", return_value=make_settings()):
            runner = SandboxRunner("owner/repo", "run-123")

        runner._container = MagicMock()

        with (
            patch("src.tools.sandbox.asyncio.get_running_loop", return_value=loop),
            patch("src.tools.sandbox.asyncio.wait_for", side_effect=timeout_wait_for),
            patch.object(
                runner,
                "_detect_pytest_capabilities",
                new=AsyncMock(return_value={"json_report": True, "timeout": True}),
            ),
        ):
            result = await runner.run_tests("pytest tests/unit", timeout=1)

        assert result.passed is False
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_run_tests_parses_json_report_when_available(self) -> None:
        from src.tools.sandbox import SandboxRunner

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(return_value=None)

        with (
            patch("src.tools.sandbox.get_settings", return_value=make_settings()),
            patch("src.tools.sandbox.asyncio.get_running_loop", return_value=loop),
        ):
            runner = SandboxRunner("owner/repo", "run-123")

        runner._container = MagicMock()

        exec_mock = AsyncMock(
            side_effect=[
                (0, b"3 passed in 0.5s"),
                (
                    0,
                    b'{"summary":{"total":3,"passed":3,"failed":0},"tests":[]}',
                ),
            ]
        )

        with (
            patch.object(
                runner,
                "_detect_pytest_capabilities",
                new=AsyncMock(return_value={"json_report": False, "timeout": False}),
            ),
            patch.object(runner, "_exec_in_container", new=exec_mock),
        ):
            result = await runner.run_tests("python -m pytest tests/unit -q")

        assert result.passed is True
        first_command = exec_mock.await_args_list[0].args[0]
        assert first_command.startswith("python -m pytest tests/unit -q")

    @pytest.mark.asyncio
    async def test_exec_in_container_uses_exec_run(self) -> None:
        from src.tools.sandbox import SandboxRunner

        loop = MagicMock()

        async def run_in_executor(_: object, fn: object) -> object:
            return fn()

        loop.run_in_executor = AsyncMock(side_effect=run_in_executor)

        with (
            patch("src.tools.sandbox.get_settings", return_value=make_settings()),
            patch("src.tools.sandbox.asyncio.get_running_loop", return_value=loop),
        ):
            runner = SandboxRunner("owner/repo", "run-123")

        runner._container = MagicMock()
        runner._container.exec_run.return_value = SimpleNamespace(
            exit_code=None,
            output=None,
        )

        exit_code, output = await runner._exec_in_container("echo hello")

        assert exit_code == 0
        assert output == b""

    @pytest.mark.asyncio
    async def test_detect_pytest_capabilities_parses_help_text(self) -> None:
        from src.tools.sandbox import SandboxRunner

        with patch("src.tools.sandbox.get_settings", return_value=make_settings()):
            runner = SandboxRunner("owner/repo", "run-123")

        with patch.object(
            runner,
            "_exec_in_container",
            new=AsyncMock(
                return_value=(0, b"--json-report\n--timeout=SECONDS\n"),
            ),
        ):
            capabilities = await runner._detect_pytest_capabilities()

        assert capabilities == {"json_report": True, "timeout": True}

    @pytest.mark.asyncio
    async def test_cleanup_removes_container_and_workspace(self) -> None:
        from src.tools.sandbox import SandboxRunner

        loop = MagicMock()

        async def run_in_executor(_: object, fn: object) -> None:
            fn()

        loop.run_in_executor = AsyncMock(side_effect=run_in_executor)

        with case_dir("cleanup") as root:
            with (
                patch("src.tools.sandbox.get_settings", return_value=make_settings()),
                patch("src.tools.sandbox.asyncio.get_running_loop", return_value=loop),
            ):
                runner = SandboxRunner("owner/repo", "run-123")

            runner._container = MagicMock()
            runner._workspace_path = root

            await runner._cleanup()

            runner._container.remove.assert_called_once_with(force=True)
            assert not root.exists()

    @pytest.mark.asyncio
    async def test_cleanup_logs_api_and_workspace_errors(self) -> None:
        from src.tools.sandbox import SandboxRunner

        loop = MagicMock()

        async def run_in_executor(_: object, fn: object) -> None:
            fn()

        loop.run_in_executor = AsyncMock(side_effect=run_in_executor)
        container_error = docker.errors.APIError("cannot remove")

        with (
            case_dir("cleanup-errors") as root,
            patch("src.tools.sandbox.get_settings", return_value=make_settings()),
            patch("src.tools.sandbox.asyncio.get_running_loop", return_value=loop),
            patch("src.tools.sandbox.log") as mock_log,
            patch("shutil.rmtree", side_effect=RuntimeError("workspace locked")),
        ):
            runner = SandboxRunner("owner/repo", "run-123")
            runner._container = MagicMock()
            runner._container.remove.side_effect = container_error
            runner._workspace_path = root

            await runner._cleanup()

        assert mock_log.warning.call_count == 2
