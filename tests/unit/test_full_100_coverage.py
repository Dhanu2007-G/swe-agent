"""Tests targeting 100% statement coverage across all remaining branches."""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from src.agent.context import (
    _extract_generic_symbols_treesitter,
    _extract_polyglot_symbols_regex,
    _extract_symbols_ast,
    _find_brace_block_end,
)
from src.agent.nodes import read_review_node
from src.observability.tracing import record_run_metrics
from src.tools.github import GitHubClient
from src.tools.sandbox import KubernetesSandboxProvider
from src.tools.search import find_relevant_files
from src.worker.processor import (
    _persist_refinement_failure,
    _run_refinement_agent_async,
    process_issue_job,
    process_review_refinement_job,
)
from src.worker.queue import enqueue_review_refinement_job

if TYPE_CHECKING:
    from src.agent.state import AgentState

# ─────────────────────────────────────────────────────────────────────────────
# 1. src/agent/context.py (lines 362, 528, 553)
# ─────────────────────────────────────────────────────────────────────────────


def test_context_find_closing_brace_no_closing() -> None:
    lines = ["func foo() {", "    x := 1", "    y := 2"]
    # No closing brace -> line 528 min(start + 20, len(lines) - 1)
    end = _find_brace_block_end(lines, 0)
    assert end == len(lines) - 1


def test_context_extract_polyglot_symbols_regex_rust() -> None:
    rust_code = """
    pub struct MyStruct {
        field: i32,
    }
    pub enum MyEnum {
        A, B
    }
    pub trait MyTrait {
        fn test();
    }
    pub async fn run_task() {
    }
    """
    symbols = _extract_polyglot_symbols_regex(rust_code, "rust")
    names = [s.name for s in symbols]
    assert "MyStruct" in names
    assert "MyEnum" in names
    assert "MyTrait" in names
    assert "run_task" in names


def test_context_extract_generic_symbols_treesitter_and_grammar_dispatch() -> None:
    mock_parser = MagicMock()
    mock_tree = MagicMock()
    mock_node = MagicMock()
    mock_node.type = "class_declaration"
    mock_node.start_point = (0, 0)
    mock_node.end_point = (5, 0)

    mock_name_child = MagicMock()
    mock_name_child.text.decode.return_value = "Calculator"
    mock_node.child_by_field_name.return_value = mock_name_child
    mock_node.children = []
    mock_tree.root_node.children = [mock_node]
    mock_parser.parse.return_value = mock_tree

    symbols = _extract_generic_symbols_treesitter("class Calculator {}", mock_parser, "typescript")
    assert len(symbols) == 1
    assert symbols[0].name == "Calculator"

    # Line 362: language in _GRAMMAR_MODULES dispatches to treesitter
    with patch("src.agent.context._get_treesitter_parser", return_value=mock_parser):
        res = _extract_symbols_ast("class Calculator {}", "typescript")
        assert len(res) == 1
        assert res[0].name == "Calculator"


# ─────────────────────────────────────────────────────────────────────────────
# 2. src/agent/nodes.py (lines 621-622, 640-641, 644)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_review_node_exception_handling_and_fallback_issue() -> None:
    mock_github = MagicMock()
    mock_github.__aenter__.return_value = mock_github
    mock_github.__aexit__.return_value = None

    # Fail get_pull_request -> lines 621-622
    mock_github.get_pull_request = AsyncMock(side_effect=RuntimeError("GitHub PR API down"))
    # Fail get_review_comment -> lines 640-641
    mock_github.get_review_comment = AsyncMock(side_effect=RuntimeError("GitHub comment API down"))

    state: AgentState = {
        "run_id": "test-run",
        "repo_full_name": "owner/repo",
        "pr_number": 42,
        "review_comment_id": 999,
        "review_feedback": "Please fix formatting",
        "job_type": "review_refinement",
    }

    with patch("src.tools.github.GitHubClient", return_value=mock_github):
        result = await read_review_node(state)

    # Line 644: issue was None, so fallback GithubIssue is created
    assert result["issue"] is not None
    assert result["issue"].issue_number == 42
    assert "Review refinement for PR #42" in result["issue"].title


# ─────────────────────────────────────────────────────────────────────────────
# 3. src/api/webhook.py (line 142)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_webhook_review_comment_returns_job_id() -> None:
    from httpx import ASGITransport, AsyncClient

    from src.api.main import create_app

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "action": "created",
            "repository": {"full_name": "owner/repo"},
            "pull_request": {"number": 10},
            "comment": {"id": 12345, "body": "nit: rename var"},
        }

        # Mock enqueue_review_refinement_job to return a job_id -> triggers line 142
        with (
            patch(
                "src.worker.queue.enqueue_review_refinement_job",
                new_callable=AsyncMock,
                return_value="refine-job-12345",
            ),
            patch(
                "src.tools.github.GitHubClient.validate_webhook_signature",
                return_value=True,
            ),
        ):
            resp = await client.post(
                "/webhooks/github",
                json=payload,
                headers={
                    "X-GitHub-Event": "pull_request_review_comment",
                    "X-Hub-Signature-256": "sha256=mock",
                    "X-GitHub-Delivery": "delivery-123",
                },
            )
            assert resp.status_code == 202
            data = resp.json()
            assert data.get("job_id") == "refine-job-12345"


# ─────────────────────────────────────────────────────────────────────────────
# 4. src/observability/tracing.py (line 225)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tracing_record_agent_run_metrics_cost_usd() -> None:
    final_state = {
        "status": "completed",
        "total_tokens_used": 1500,
        "cumulative_cost_usd": 0.045,  # > 0 triggers line 225
        "pull_request": {"pr_url": "https://github.com/o/r/pull/1"},
        "is_fork": False,
    }
    await record_run_metrics(final_state)


# ─────────────────────────────────────────────────────────────────────────────
# 5. src/tools/github.py (lines 252-253, 272-273, 294-295)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_github_fork_retry_and_branch_collision(tmp_path: Path) -> None:
    client = GitHubClient()
    mock_gh = MagicMock()
    client._gh = mock_gh

    mock_repo = MagicMock()
    mock_repo.default_branch = "main"
    mock_repo.permissions.push = False
    mock_repo.get_branch.return_value.commit.sha = "sha123"

    mock_user = MagicMock()
    mock_user.login = "myuser"
    mock_gh.get_user.return_value = mock_user

    mock_fork = MagicMock()
    mock_fork.full_name = "myuser/repo"
    mock_fork.name = "repo"
    mock_user.create_fork.return_value = mock_fork

    # 1. Fork retry exception (lines 252-253)
    call_count = 0

    def get_repo_side_effect(name: str) -> MagicMock:
        nonlocal call_count
        if "myuser" in name:
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Fork not ready yet")
            return mock_fork
        return mock_repo

    mock_gh.get_repo.side_effect = get_repo_side_effect

    # 2. Branch collision on fork (lines 272-273)
    from github import GithubException

    ref_call_count = 0

    def create_ref_side_effect(ref: str, sha: str) -> None:
        nonlocal ref_call_count
        ref_call_count += 1
        if ref_call_count == 1:
            raise GithubException(422, {"message": "Reference already exists"}, headers={})
        return None

    mock_fork.create_git_ref.side_effect = create_ref_side_effect

    mock_created_pr = MagicMock()
    mock_created_pr.number = 101
    mock_created_pr.html_url = "https://github.com/upstream/repo/pull/101"
    mock_created_pr.title = "Fix bug"
    mock_created_pr.body = "Fix details"
    mock_repo.create_pull.return_value = mock_created_pr

    mock_git_repo = MagicMock()

    with (
        patch("git.Repo.clone_from", return_value=mock_git_repo),
        patch("src.tools.github._build_authenticated_clone_kwargs", return_value={}),
        patch("time.sleep"),
    ):
        pr = await client.create_pull_request(
            repo_full_name="upstream/repo",
            branch_name="fix-feature",
            title="Fix bug",
            body="Fix details",
            patches=[],
        )
        assert pr.pr_number == 101
        assert pr.is_fork is True


@pytest.mark.asyncio
async def test_github_existing_pr_checkout_fallback(tmp_path: Path) -> None:
    client = GitHubClient()
    mock_gh = MagicMock()
    client._gh = mock_gh

    mock_repo = MagicMock()
    mock_repo.default_branch = "main"
    mock_repo.permissions.push = True
    mock_repo.get_branch.return_value.commit.sha = "sha123"

    mock_existing_pr = MagicMock()
    mock_existing_pr.number = 15
    mock_existing_pr.html_url = "https://github.com/upstream/repo/pull/15"
    mock_existing_pr.title = "Existing PR"
    mock_existing_pr.body = "Existing description"
    mock_existing_pr.head.ref = "fix-feature"
    mock_existing_pr.base.ref = "main"
    mock_existing_pr.head.sha = "sha123"
    mock_repo.get_pull.return_value = mock_existing_pr
    mock_gh.get_repo.return_value = mock_repo

    mock_git_repo = MagicMock()
    # Checkout fails on first call -> triggers lines 294-295
    mock_git_repo.git.checkout.side_effect = [
        RuntimeError("branch does not exist locally"),
        None,
    ]

    with (
        patch("git.Repo.clone_from", return_value=mock_git_repo),
        patch("src.tools.github._build_authenticated_clone_kwargs", return_value={}),
        patch("time.sleep"),
    ):
        pr = await client.create_pull_request(
            repo_full_name="upstream/repo",
            branch_name="fix-feature",
            title="Fix bug",
            body="Fix details",
            patches=[],
            existing_pr_number=15,
        )
        assert pr.pr_number == 15


# ─────────────────────────────────────────────────────────────────────────────
# 6. src/tools/sandbox.py (lines 315-326, 355-358, 380-389)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_k8s_sandbox_provider_full_client_branches(tmp_path: Path) -> None:
    mock_k8s_client = MagicMock()
    provider = KubernetesSandboxProvider(namespace="test-ns", k8s_client=mock_k8s_client)
    mock_settings = MagicMock()
    mock_settings.is_production = False

    # 1. create_container success (lines 315-323)
    container = await provider.create_container(
        tmp_path, "test-run-k8s", "owner/repo", mock_settings
    )
    assert container["status"] == "running"

    # 2. create_container error branch (lines 324-326)
    mock_k8s_client.create_namespaced_pod.side_effect = RuntimeError("K8s API down")
    with pytest.raises(RuntimeError, match="K8s API down"):
        await provider.create_container(tmp_path, "test-run-fail", "owner/repo", mock_settings)

    mock_k8s_client.create_namespaced_pod.side_effect = None

    # 3. exec_command with awaitable and sync response (lines 355-358)
    async def async_exec(*args, **kwargs) -> tuple[int, bytes]:
        return 0, b"async output"

    mock_k8s_client.exec_command = async_exec
    code, out = await provider.exec_command(container, "ls")
    assert code == 0
    assert out == b"async output"

    mock_k8s_client.exec_command = MagicMock(return_value=(1, b"sync output"))
    code, out = await provider.exec_command(container, "pwd")
    assert code == 1
    assert out == b"sync output"

    # 4. cleanup success (lines 380-387)
    ws = tmp_path / "ws"
    ws.mkdir()
    await provider.cleanup(container, ws, "test-run-k8s")
    assert not ws.exists()

    # 5. cleanup error branch (lines 388-389)
    mock_k8s_client.delete_namespaced_pod.side_effect = RuntimeError("Delete pod failed")
    await provider.cleanup(container, None, "test-run-k8s")


# ─────────────────────────────────────────────────────────────────────────────
# 7. src/tools/search.py (line 207)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_skips_non_code_extensions(tmp_path: Path) -> None:
    # Create a non-code file (.txt) -> triggers line 207 continue
    (tmp_path / "notes.txt").write_text("just some text notes")
    (tmp_path / "main.py").write_text("def hello(): pass")

    with patch(
        "src.tools._repo_cache.get_local_repo_path",
        new_callable=AsyncMock,
        return_value=tmp_path,
    ):
        results = await find_relevant_files("mock/repo", "hello", exclude=[])
        assert any("main.py" in r for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# 8. src/worker/processor.py (lines 111-118, 278-286, 343, 372-373, 425-426)
# ─────────────────────────────────────────────────────────────────────────────


def test_processor_checkpointing_fallback_in_process_job() -> None:
    # Trigger lines 111-118: checkpointed_run raises exception in process_issue_job
    mock_repo = MagicMock()
    mock_repo.mark_run_running = AsyncMock()
    mock_repo.update_run = AsyncMock()

    with (
        patch(
            "src.worker.processor.checkpointed_run",
            side_effect=RuntimeError("Postgres DB down"),
        ),
        patch(
            "src.worker.processor.run_agent",
            new_callable=AsyncMock,
            return_value={"status": "completed"},
        ),
        patch(
            "src.worker.processor.RunRepository",
            return_value=mock_repo,
        ),
        patch(
            "src.worker.processor.get_redis_connection",
            new_callable=AsyncMock,
        ),
        patch(
            "src.worker.processor.release_active_job_lock",
            new_callable=AsyncMock,
        ),
        patch("src.worker.processor.GitHubClient"),
    ):
        result = process_issue_job(
            repo_full_name="owner/repo",
            issue_number=1,
            run_id="test-job-fallback",
        )
        assert result["status"] == "completed"


def test_processor_refinement_fatal_error_handling() -> None:
    # Trigger lines 278-286: process_review_refinement_job fatal error
    with (
        patch(
            "src.worker.processor._run_refinement_agent_async",
            side_effect=RuntimeError("Fatal refinement error"),
        ),
        patch(
            "src.worker.processor._persist_refinement_failure",
            new_callable=AsyncMock,
        ),
        pytest.raises(RuntimeError, match="Fatal refinement error"),
    ):
        process_review_refinement_job(
            repo_full_name="owner/repo",
            pr_number=5,
            comment_id=123,
            run_id="refine-fail-job",
            review_feedback="feedback",
        )


@pytest.mark.asyncio
async def test_run_refinement_agent_async_checkpoint_and_json_error() -> None:
    # Trigger line 343 (checkpointed_run succeeds) and lines 372-373 (json dumps fail)
    class Unserializable:
        pass

    unserializable_state = {
        "status": "completed",
        "pull_request": {"pr_url": "https://github.com/o/r/pull/1"},
        "retry_count": 0,
        "bad_obj": Unserializable(),
    }

    mock_checkpointer = MagicMock()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_checkpointed_run(run_id: str):
        yield mock_checkpointer

    mock_repo = MagicMock()
    mock_repo.mark_run_running = AsyncMock()
    mock_repo.update_run = AsyncMock()

    with (
        patch("src.worker.processor.checkpointed_run", side_effect=mock_checkpointed_run),
        patch(
            "src.worker.processor.run_agent",
            new_callable=AsyncMock,
            return_value=unserializable_state,
        ),
        patch("src.worker.processor.RunRepository", return_value=mock_repo),
        patch("src.worker.processor.get_redis_connection", new_callable=AsyncMock),
        patch("src.worker.processor.release_active_job_lock", new_callable=AsyncMock),
        patch("json.dumps", side_effect=TypeError("Cannot serialize")),
    ):
        res = await _run_refinement_agent_async(
            repo_full_name="owner/repo",
            pr_number=10,
            comment_id=1,
            run_id="refine-run-10",
            review_feedback="Fix typos",
        )
        assert res["status"] == "completed"


@pytest.mark.asyncio
async def test_persist_refinement_failure_swallows_exception() -> None:
    # Trigger lines 425-426: except Exception: pass in _persist_refinement_failure
    with (
        patch(
            "src.worker.processor.RunRepository",
            side_effect=RuntimeError("DB disconnected"),
        ),
        patch("src.worker.processor.release_active_job_lock", new_callable=AsyncMock),
    ):
        # Should not raise exception
        await _persist_refinement_failure("run-id", "o/r", 1, "some err")


# ─────────────────────────────────────────────────────────────────────────────
# 9. src/worker/queue.py (lines 198-199, 217-220, 252-256)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_review_refinement_job_lock_held() -> None:
    # Trigger lines 198-199: Lock not acquired
    mock_redis = AsyncMock()
    mock_redis.set.return_value = False
    mock_redis.get.return_value = "existing-job-id"

    with patch("src.worker.queue.get_redis_connection", return_value=mock_redis):
        jid = await enqueue_review_refinement_job("owner/repo", 5, 10, "feedback")
        assert jid == "existing-job-id"


@pytest.mark.asyncio
async def test_enqueue_review_refinement_job_integrity_error_no_active_run() -> None:
    # Trigger lines 217-220: IntegrityError without active run
    mock_redis = AsyncMock()
    mock_redis.set.return_value = True

    mock_repo = MagicMock()
    mock_repo.create_run = AsyncMock(side_effect=IntegrityError("stmt", "params", "orig"))
    mock_repo.get_active_run = AsyncMock(return_value=None)
    mock_repo.update_run = AsyncMock()

    with (
        patch("src.worker.queue.get_redis_connection", return_value=mock_redis),
        patch("src.worker.queue.RunRepository", return_value=mock_repo),
        pytest.raises(IntegrityError),
    ):
        await enqueue_review_refinement_job("owner/repo", 5, 10, "feedback")


@pytest.mark.asyncio
async def test_enqueue_review_refinement_job_rq_enqueue_error() -> None:
    # Trigger lines 252-256: loop.run_in_executor raises
    mock_redis = AsyncMock()
    mock_redis.set.return_value = True

    mock_repo = MagicMock()
    mock_repo.create_run = AsyncMock()
    mock_repo.update_run = AsyncMock()

    with (
        patch("src.worker.queue.get_redis_connection", return_value=mock_redis),
        patch("src.worker.queue.RunRepository", return_value=mock_repo),
        patch("asyncio.get_running_loop") as mock_loop,
    ):
        mock_loop.return_value.run_in_executor = AsyncMock(
            side_effect=RuntimeError("Redis queue enqueue failed")
        )
        with pytest.raises(RuntimeError, match="Redis queue enqueue failed"):
            await enqueue_review_refinement_job("owner/repo", 5, 10, "feedback")
