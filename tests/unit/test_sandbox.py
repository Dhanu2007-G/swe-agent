"""
tests/unit/test_sandbox.py — Unit tests for Docker sandbox result parsing.
No real Docker required — tests the pure parsing logic.
"""
from __future__ import annotations

import json
import pytest

from src.tools.sandbox import _infer_error_category, _parse_raw_pytest_output, _parse_test_results
from src.agent.state import ErrorCategory


class TestErrorCategoryInference:
    def test_syntax_error(self) -> None:
        assert _infer_error_category("SyntaxError: invalid syntax") == ErrorCategory.SYNTAX_ERROR

    def test_import_error(self) -> None:
        assert _infer_error_category("ModuleNotFoundError: No module named 'foo'") == ErrorCategory.IMPORT_ERROR

    def test_type_error(self) -> None:
        assert _infer_error_category("TypeError: unsupported operand type(s)") == ErrorCategory.TYPE_ERROR

    def test_attribute_error(self) -> None:
        assert _infer_error_category("AttributeError: 'NoneType' object has no attribute") == ErrorCategory.TYPE_ERROR

    def test_fixture_error(self) -> None:
        assert _infer_error_category("fixture 'db_session' not found") == ErrorCategory.FIXTURE_ERROR

    def test_unknown_fallback(self) -> None:
        assert _infer_error_category("some random error message") == ErrorCategory.LOGIC_ERROR


class TestParsePytestOutput:
    def test_all_passing(self) -> None:
        output = "12 passed, 0 failed in 2.34s"
        result = _parse_raw_pytest_output(output, exit_code=0, duration=2.34)
        assert result.passed is True
        assert result.total == 12
        assert result.failed_count == 0

    def test_some_failing(self) -> None:
        output = "10 passed, 2 failed in 3.1s\nFAILED tests/test_foo.py::test_bar - AssertionError: expected 1"
        result = _parse_raw_pytest_output(output, exit_code=1, duration=3.1)
        assert result.passed is False
        assert result.failed_count == 2

    def test_nonzero_exit_means_failed(self) -> None:
        output = ""
        result = _parse_raw_pytest_output(output, exit_code=1, duration=0.1)
        assert result.passed is False


class TestParseTestResultsJson:
    SAMPLE_REPORT = {
        "summary": {"total": 5, "passed": 4, "failed": 1},
        "tests": [
            {
                "nodeid": "tests/test_foo.py::test_ok",
                "outcome": "passed",
                "call": {},
            },
            {
                "nodeid": "tests/test_foo.py::test_broken",
                "outcome": "failed",
                "call": {
                    "longrepr": "AssertionError: assert 1 == 2\n  where 1 = foo()"
                },
            },
        ],
    }

    def test_parses_failures(self) -> None:
        import json
        result = _parse_test_results(
            json_report=json.dumps(self.SAMPLE_REPORT),
            raw_output="",
            exit_code=1,
            duration=1.5,
        )
        assert result.passed is False
        assert result.failed_count == 1
        assert len(result.failures) == 1
        assert result.failures[0].test_name == "test_broken"

    def test_invalid_json_falls_back(self) -> None:
        result = _parse_test_results(
            json_report="not valid json {{{",
            raw_output="5 passed in 1.2s",
            exit_code=0,
            duration=1.2,
        )
        assert result.passed is True

    def test_parses_coverage_from_json(self) -> None:
        import json
        report = {
            "summary": {"total": 5, "passed": 5, "failed": 0},
            "tests": [],
            "coverage": {"totals": {"covered_lines": 80, "num_statements": 100}},
        }
        result = _parse_test_results(
            json_report=json.dumps(report),
            raw_output="",
            exit_code=0,
            duration=1.0,
        )
        assert result.passed is True
        assert result.coverage_pct == 80.0


class TestSandboxRunnerHelpers:
    @pytest.mark.asyncio
    async def test_build_authenticated_clone_kwargs_async(self) -> None:
        from src.tools.sandbox import _build_authenticated_clone_kwargs_async
        from types import SimpleNamespace

        settings = SimpleNamespace(github_token_value="ghp_secret")
        kwargs = await _build_authenticated_clone_kwargs_async("owner/repo", settings)
        assert kwargs["url"] == "https://ghp_secret@github.com/owner/repo.git"
        assert kwargs["no_single_branch"] is True

    @pytest.mark.asyncio
    async def test_install_packages_workflow(self) -> None:
        from src.tools.sandbox import SandboxRunner
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock, patch

        settings = SimpleNamespace(
            sandbox_network_disabled=False,
            sandbox_timeout_seconds=60,
        )

        with patch("src.tools.sandbox.get_settings", return_value=settings):
            sb = SandboxRunner("owner/repo", "run-1")
            sb._container = MagicMock()

            # Empty packages returns early
            await sb.install_packages([])

            # Packages already available
            sb._exec_in_container = AsyncMock(return_value=(0, b"OK\n"))
            await sb.install_packages(["pytest>=8.0"])
            assert sb._exec_in_container.await_count == 1

            # Missing package installed successfully
            sb._exec_in_container = AsyncMock(side_effect=[(0, b"MISSING\n"), (0, b"Successfully installed")])
            await sb.install_packages(["numpy"])
            assert sb._exec_in_container.await_count == 2

            # Missing package install fails
            sb._exec_in_container = AsyncMock(side_effect=[(0, b"MISSING\n"), (1, b"Error installing")])
            await sb.install_packages(["broken-pkg"])
            assert sb._exec_in_container.await_count == 2

    @pytest.mark.asyncio
    async def test_copy_repo_into_container(self) -> None:
        from src.tools.sandbox import SandboxRunner

        sb = SandboxRunner("owner/repo", "run-1")
        await sb._copy_repo_into_container()

    @pytest.mark.asyncio
    async def test_run_tests_command_building(self) -> None:
        from src.tools.sandbox import SandboxRunner
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock, patch

        settings = SimpleNamespace(
            sandbox_timeout_seconds=60,
            sandbox_image="swe-sandbox:latest",
        )
        with patch("src.tools.sandbox.get_settings", return_value=settings):
            sb = SandboxRunner("owner/repo", "run-1")
            sb._container = MagicMock()
            sb._exec_in_container = AsyncMock(return_value=(0, b"1 passed in 0.1s"))
            sb._detect_pytest_capabilities = AsyncMock(return_value={})

            res = await sb.run_tests("tests/test_feature.py")
            assert res.passed is True
            # Verify python -m pytest was prepended
            cmd_run = sb._exec_in_container.call_args_list[0][0][0]
            assert "python -m pytest tests/test_feature.py" in cmd_run

    def test_detect_test_ecosystem_and_build_command(self) -> None:
        from src.tools.sandbox import _detect_test_ecosystem, _build_ecosystem_command

        # Detection
        assert _detect_test_ecosystem("npm test") == "javascript"
        assert _detect_test_ecosystem("yarn vitest run") == "javascript"
        assert _detect_test_ecosystem("pnpm jest") == "javascript"
        assert _detect_test_ecosystem("go test ./...") == "go"
        assert _detect_test_ecosystem("cargo test --lib") == "rust"
        assert _detect_test_ecosystem("pytest tests/") == "python"

        # Command builders
        assert "--reporter=json" in _build_ecosystem_command("vitest run", "javascript")
        assert "--json" in _build_ecosystem_command("jest", "javascript")
        assert "npm test 2>&1" in _build_ecosystem_command("npm test", "javascript")
        assert "go test -json ./..." in _build_ecosystem_command("go test ./...", "go")
        assert "go test -json" in _build_ecosystem_command("./pkg/...", "go")
        assert "cargo test --lib 2>&1" in _build_ecosystem_command("cargo test --lib", "rust")
        assert "python -m pytest tests/" in _build_ecosystem_command("pytest tests/", "python")

    def test_parse_jest_vitest_output(self) -> None:
        from src.tools.sandbox import _parse_jest_vitest_output, _parse_test_results

        # 1. Structured JSON (Vitest/Jest)
        json_payload = json.dumps({
            "numTotalTests": 2,
            "numPassedTests": 1,
            "numFailedTests": 1,
            "testResults": [{
                "assertionResults": [
                    {"status": "passed", "title": "test one"},
                    {"status": "failed", "title": "test two", "failureMessages": ["TypeError: undefined is not a function"]},
                ]
            }]
        })
        res_json = _parse_jest_vitest_output(f"Output before\n{json_payload}\nOutput after", exit_code=1, duration=1.2)
        assert res_json.passed is False
        assert res_json.total == 2
        assert res_json.failed_count == 1
        assert len(res_json.failures) == 1
        assert res_json.failures[0].test_id == "test two"

        # 2. CLI Text Fallback (Regex)
        raw_cli = "FAIL src/auth.test.ts\n✕ should validate token\nTests: 1 failed, 2 passed, 3 total\n"
        res_cli = _parse_jest_vitest_output(raw_cli, exit_code=1, duration=0.8)
        assert res_cli.passed is False
        assert res_cli.total == 3
        assert res_cli.failed_count == 1
        assert len(res_cli.failures) >= 1

        # 3. Malformed JSON matching regex
        res_malformed = _parse_jest_vitest_output('{"numTotalTests": corrupt json}', exit_code=0, duration=0.5)
        assert res_malformed.passed is True

        # 4. Dispatched through _parse_test_results
        res_dispatch = _parse_test_results(
            json_report="{}",
            raw_output=json_payload,
            exit_code=1,
            duration=1.0,
            ecosystem="javascript",
        )
        assert res_dispatch.passed is False

    def test_parse_go_test_output(self) -> None:
        from src.tools.sandbox import _parse_go_test_output, _parse_test_results

        # 1. Streaming JSON events with one malformed JSON line
        events = [
            '{"corrupt json line',
            json.dumps({"Action": "run", "Test": "TestAdd"}),
            json.dumps({"Action": "pass", "Test": "TestAdd"}),
            json.dumps({"Action": "run", "Test": "TestSub"}),
            json.dumps({"Action": "output", "Test": "TestSub", "Output": "sub_test.go:12: expected 2 got 3\n"}),
            json.dumps({"Action": "fail", "Test": "TestSub"}),
        ]
        raw_events = "\n".join(events)
        res_json = _parse_go_test_output(raw_events, exit_code=1, duration=0.5)
        assert res_json.passed is False
        assert res_json.total == 2
        assert res_json.failed_count == 1
        assert len(res_json.failures) == 1
        assert res_json.failures[0].test_id == "TestSub"
        assert "expected 2 got 3" in res_json.failures[0].error_message

        # 2. CLI Text Fallback
        raw_cli = "=== RUN   TestDivide\n--- FAIL: TestDivide (0.00s)\n    math_test.go:40: divide by zero\nFAIL\n"
        res_cli = _parse_go_test_output(raw_cli, exit_code=1, duration=0.2)
        assert res_cli.passed is False
        assert res_cli.failed_count == 1
        assert res_cli.failures[0].test_id == "TestDivide"

        # 3. Dispatched
        res_dispatch = _parse_test_results(
            json_report="{}",
            raw_output=raw_events,
            exit_code=1,
            duration=0.5,
            ecosystem="go",
        )
        assert res_dispatch.passed is False

    def test_parse_cargo_test_output(self) -> None:
        from src.tools.sandbox import _parse_cargo_test_output, _parse_test_results

        # 1. Cargo test failure
        raw_cargo_fail = (
            "running 3 tests\n"
            "test tests::test_add ... ok\n"
            "test tests::test_sub ... FAILED\n"
            "----\n"
            "---- tests::test_sub stdout ----\n"
            "thread 'tests::test_sub' panicked at 'assertion failed: `(left == right)`'\n"
            "test result: FAILED. 2 passed; 1 failed; 0 ignored;\n"
        )
        res_fail = _parse_cargo_test_output(raw_cargo_fail, exit_code=101, duration=2.1)
        assert res_fail.passed is False
        assert res_fail.total == 3
        assert res_fail.failed_count == 1
        assert len(res_fail.failures) == 1
        assert res_fail.failures[0].test_id == "tests::test_sub"

        # 2. Cargo test success
        raw_cargo_ok = (
            "running 2 tests\n"
            "test tests::test_one ... ok\n"
            "test tests::test_two ... ok\n"
            "test result: ok. 2 passed; 0 failed; 0 ignored;\n"
        )
        res_ok = _parse_cargo_test_output(raw_cargo_ok, exit_code=0, duration=1.0)
        assert res_ok.passed is True
        assert res_ok.total == 2
        assert res_ok.failed_count == 0

        # 3. Dispatched
        res_dispatch = _parse_test_results(
            json_report="{}",
            raw_output=raw_cargo_fail,
            exit_code=101,
            duration=2.1,
            ecosystem="rust",
        )
        assert res_dispatch.passed is False

    @pytest.mark.asyncio
    async def test_cloud_sandbox_provider(self, tmp_path: Path) -> None:
        from src.tools.sandbox import CloudSandboxProvider, SandboxRunner
        from types import SimpleNamespace

        provider = CloudSandboxProvider(endpoint="http://remote-cluster:8080")
        settings = SimpleNamespace(sandbox_timeout_seconds=30)
        
        container = await provider.create_container(tmp_path, "run-xyz", "owner/repo", settings)
        assert container["status"] == "running"

        code, out = await provider.exec_command(container, "echo test")
        assert code == 0
        assert b"CLOUD EXEC OK" in out

        await provider.cleanup(container, tmp_path, "run-xyz")
        assert not tmp_path.exists()

    @pytest.mark.asyncio
    async def test_docker_sandbox_provider_exec_and_cleanup(self, tmp_path: Path) -> None:
        from src.tools.sandbox import DockerSandboxProvider
        from unittest.mock import MagicMock
        import docker.errors

        provider = DockerSandboxProvider()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=b"DOCKER OK")

        code, out = await provider.exec_command(mock_container, "ls")
        assert code == 0
        assert out == b"DOCKER OK"

        # Test cleanup with Docker API error and tempdir cleanup
        mock_container.remove.side_effect = docker.errors.APIError("API down")
        await provider.cleanup(mock_container, tmp_path, "run-123")
        assert not tmp_path.exists()

    @pytest.mark.asyncio
    async def test_docker_sandbox_provider_create_container(self, tmp_path: Path) -> None:
        from src.tools.sandbox import DockerSandboxProvider
        from unittest.mock import MagicMock, patch
        from types import SimpleNamespace

        provider = DockerSandboxProvider()
        settings = SimpleNamespace(
            sandbox_image="swe-sandbox:latest",
            sandbox_workspace_dir="/workspace",
            sandbox_network_disabled=True,
            sandbox_memory_limit="1g",
            sandbox_cpu_quota=50000,
        )

        mock_client = MagicMock()
        mock_client.containers.create.return_value = MagicMock(short_id="dock123")

        with patch("docker.from_env", return_value=mock_client):
            container = await provider.create_container(tmp_path, "run-dock", "owner/repo", settings)
            assert container.short_id == "dock123"
            mock_client.containers.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_sandbox_provider_abstract_methods(self) -> None:
        from src.tools.sandbox import SandboxProvider
        from pathlib import Path

        class DummyProvider(SandboxProvider):
            async def create_container(self, *args, **kwargs):
                return await super().create_container(*args, **kwargs)

            async def exec_command(self, *args, **kwargs):
                return await super().exec_command(*args, **kwargs)

            async def cleanup(self, *args, **kwargs):
                return await super().cleanup(*args, **kwargs)

        dummy = DummyProvider()
        assert await dummy.create_container(Path("/tmp"), "run", "repo", None) is None
        assert await dummy.exec_command(None, "cmd") is None
        assert await dummy.cleanup(None, None, "run") is None
