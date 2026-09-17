"""
tests/unit/test_phase_coverage.py — Tests covering polyglot, refinement, and auth features.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
class TestApiSecurityAuth:
    async def test_verify_api_key_auth_disabled(self) -> None:
        from src.api.security import verify_api_key

        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.api_auth_enabled = False
            result = await verify_api_key(None)
            assert result is None

    async def test_verify_api_key_missing(self) -> None:
        from src.api.security import verify_api_key

        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.api_auth_enabled = True
            with pytest.raises(HTTPException) as exc:
                await verify_api_key(None)
            assert exc.value.status_code == 401

    async def test_verify_api_key_invalid(self) -> None:
        from src.api.security import verify_api_key

        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.api_auth_enabled = True
            mock_settings.return_value.api_keys = ["secret-key-123"]
            with pytest.raises(HTTPException) as exc:
                await verify_api_key("wrong-key")
            assert exc.value.status_code == 403

    async def test_verify_api_key_valid(self) -> None:
        from src.api.security import verify_api_key

        with patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.api_auth_enabled = True
            mock_settings.return_value.api_keys = ["secret-key-123"]
            result = await verify_api_key("secret-key-123")
            assert result == "secret-key-123"


@pytest.mark.asyncio
class TestApiRoutesProtection:
    async def test_trigger_run_disallowed_repo(self) -> None:
        from src.api.routes import TriggerRequest, trigger_run

        req = TriggerRequest(repo_full_name="unauthorized/repo", issue_number=1)
        with patch("src.api.routes.get_settings") as mock_settings:
            mock_settings.return_value.allowed_repositories = ["authorized/repo"]
            with pytest.raises(HTTPException) as exc:
                await trigger_run(req)
            assert exc.value.status_code == 403


@pytest.mark.asyncio
class TestWebhookSecurityAndRefinement:
    async def test_webhook_missing_event_header(self) -> None:
        from fastapi import Request

        from src.api.webhook import github_webhook

        req = MagicMock(spec=Request)
        req.body = AsyncMock(return_value=b'{"action":"opened"}')

        with patch("src.tools.github.GitHubClient.validate_webhook_signature", return_value=True):
            with pytest.raises(HTTPException) as exc:
                await github_webhook(
                    request=req,
                    x_hub_signature_256="sha256=fake",
                    x_github_event=None,
                )
            assert exc.value.status_code == 400

    async def test_webhook_disallowed_repo(self) -> None:
        import json

        from fastapi import Request

        from src.api.webhook import github_webhook

        payload = json.dumps(
            {
                "action": "opened",
                "repository": {"full_name": "disallowed/repo"},
                "issue": {"number": 5},
            }
        ).encode()

        req = MagicMock(spec=Request)
        req.body = AsyncMock(return_value=payload)

        with (
            patch("src.tools.github.GitHubClient.validate_webhook_signature", return_value=True),
            patch("src.api.webhook.get_settings") as mock_settings,
        ):
            mock_settings.return_value.allowed_repositories = ["allowed/repo"]
            resp = await github_webhook(
                request=req,
                x_hub_signature_256="sha256=fake",
                x_github_event="issues",
            )
            assert resp.status_code == 200
            assert b"repository not allowed" in resp.body


class TestConfigValidators:
    def test_parse_api_keys_and_allowed_repos_str(self) -> None:
        from src.config import Settings

        s = Settings(
            anthropic_api_key="sk-ant-test",
            github_token="ghp_test",
            api_keys="key1, key2 ,key3",
            allowed_repositories="org/repo1, org/repo2",
        )
        assert s.api_keys == ["key1", "key2", "key3"]
        assert s.allowed_repositories == ["org/repo1", "org/repo2"]

    @pytest.mark.asyncio
    async def test_production_auth_startup_check(self) -> None:
        from src.api.main import lifespan
        from src.config import Settings

        s = Settings(
            app_env="production",
            api_auth_enabled=False,
            anthropic_api_key="sk-ant-test",
            github_token="ghp_test",
        )
        with (
            patch("src.api.main.get_settings", return_value=s),
            pytest.raises(
                RuntimeError, match="Production environment requires API_AUTH_ENABLED=true"
            ),
        ):
            async with lifespan(MagicMock()):
                pass


@pytest.mark.asyncio
class TestDatabaseRefinementUpdates:
    async def test_update_run_refinement_fields(self) -> None:
        from src.db.repository import RunRepository

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()

        async def mock_get_session():
            yield mock_session

        with patch("src.db.repository.get_session", mock_get_session):
            repo = RunRepository()
            await repo.update_run(
                run_id="run-refine-123",
                status="succeeded",
                pr_number=99,
                job_type="review_refinement",
                review_comment_id=12345,
                parent_run_id="parent-run-001",
            )
            assert mock_session.execute.called


@pytest.mark.asyncio
class TestReadReviewNode:
    async def test_read_review_node_success(self) -> None:
        from src.agent.nodes import read_review_node

        state = {
            "run_id": "run-rev-1",
            "repo_full_name": "owner/repo",
            "pr_number": 42,
            "review_comment_id": 999,
            "branch_name": "feature-branch",
            "review_feedback": "Initial feedback",
        }

        mock_gh = AsyncMock()
        mock_gh.get_pull_request = AsyncMock(
            return_value={
                "title": "Fix something",
                "body": "PR description",
                "head_branch": "fix-branch",
                "html_url": "https://github.com/owner/repo/pull/42",
            }
        )
        mock_gh.get_review_comment = AsyncMock(
            return_value={
                "path": "src/main.py",
                "body": "Please add typing",
                "diff_hunk": "@@ -1 +1 @@",
                "line": 10,
            }
        )

        with patch("src.tools.github.GitHubClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_gh
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client

            result = await read_review_node(state)  # type: ignore[arg-type]
            assert result["branch_name"] == "fix-branch"
            assert "Please add typing" in result["review_feedback"]
            assert result["issue"].issue_number == 42


class TestKubernetesSandboxRejection:
    def test_get_sandbox_provider_k8s_and_docker_rejection(self) -> None:
        from src.tools.sandbox import (
            KubernetesSandboxProvider,
            get_sandbox_provider,
        )

        # K8s provider selection
        s_k8s = SimpleNamespace(sandbox_provider="kubernetes", k8s_namespace="test-ns")
        p = get_sandbox_provider(s_k8s)
        assert isinstance(p, KubernetesSandboxProvider)
        assert p.namespace == "test-ns"

        # Docker in K8s rejection in production
        s_docker_prod = SimpleNamespace(
            sandbox_provider="docker",
            is_production=True,
            allow_docker_in_k8s=False,
        )
        with (
            patch.dict("os.environ", {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}),
            pytest.raises(
                RuntimeError, match="Docker sandbox provider cannot be run inside Kubernetes"
            ),
        ):
            get_sandbox_provider(s_docker_prod)

    @pytest.mark.asyncio
    async def test_k8s_exec_and_disabled_in_prod(self) -> None:
        from src.tools.sandbox import KubernetesSandboxProvider

        provider = KubernetesSandboxProvider(namespace="custom-ns")
        code, out = await provider.exec_command({"pod_name": "test-pod"}, "ls -la")
        assert code == 0
        assert b"K8S EXEC OK" in out

        # Test rejection if disabled in production
        s_disabled = SimpleNamespace(is_production=True, k8s_sandbox_enabled=False)
        with pytest.raises(
            RuntimeError, match="Kubernetes sandbox provider is disabled in production"
        ):
            await provider.create_container(MagicMock(), "run-1", "owner/repo", s_disabled)


class TestReviewRefinementWorkerProcessor:
    @pytest.mark.asyncio
    async def test_enqueue_review_refinement_job(self) -> None:
        from src.worker.queue import enqueue_review_refinement_job

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.hset = AsyncMock()

        with (
            patch(
                "src.worker.queue.get_redis_connection",
                new_callable=AsyncMock,
                return_value=mock_redis,
            ),
            patch("src.worker.queue._enqueue_review_refinement_sync"),
            patch("src.worker.queue.RunRepository") as mock_repo_cls,
        ):
            mock_repo = AsyncMock()
            mock_repo_cls.return_value = mock_repo

            job_id = await enqueue_review_refinement_job(
                repo_full_name="owner/repo",
                pr_number=42,
                comment_id=101,
                review_feedback="Fix typos",
            )
            assert job_id.startswith("refine-")
            assert mock_repo.create_run.called

    def test_process_review_refinement_job_wrapper(self) -> None:
        from src.worker.processor import process_review_refinement_job

        settings = type("Settings", (), {"log_level": "INFO", "log_format": "json"})()
        with (
            patch("src.worker.processor.get_settings", return_value=settings),
            patch("src.worker.processor.configure_logging"),
            patch(
                "src.worker.processor._run_refinement_agent_async",
                new=AsyncMock(return_value={"status": "succeeded", "run_id": "run-ref-99"}),
            ),
        ):
            res = process_review_refinement_job(
                repo_full_name="owner/repo",
                pr_number=42,
                comment_id=101,
                run_id="run-ref-99",
                review_feedback="Fix typos",
            )
            assert res["status"] == "succeeded"


@pytest.mark.asyncio
class TestGitHubClientReviewMethods:
    async def test_get_pull_request_and_review_comment(self) -> None:
        from src.tools.github import GitHubClient

        client = GitHubClient()
        mock_gh = MagicMock()
        client._gh = mock_gh

        mock_pr = MagicMock()
        mock_pr.number = 42
        mock_pr.title = "Sample PR"
        mock_pr.body = "Body"
        mock_pr.state = "open"
        mock_pr.draft = False
        mock_pr.head.ref = "patch-branch"
        mock_pr.base.ref = "main"
        mock_pr.html_url = "https://github.com/owner/repo/pull/42"

        mock_comment = MagicMock()
        mock_comment.id = 555
        mock_comment.body = "Looks good"
        mock_comment.path = "app.py"
        mock_comment.diff_hunk = "@@ -1 +1 @@"
        mock_comment.line = 12

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_pr.get_review_comment.return_value = mock_comment
        mock_gh.get_repo.return_value = mock_repo

        pr_dict = await client.get_pull_request("owner/repo", 42)
        assert pr_dict["title"] == "Sample PR"
        assert pr_dict["head_branch"] == "patch-branch"

        comment_dict = await client.get_review_comment("owner/repo", 42, 555)
        assert comment_dict["body"] == "Looks good"
        assert comment_dict["path"] == "app.py"


class TestGraphAndTreesitterCoverage:
    def test_route_entry_review_refinement(self) -> None:
        from src.agent.graph import route_entry

        assert route_entry({"job_type": "review_refinement"}) == "read_review"
        assert route_entry({"job_type": "issue"}) == "read_issue"

    @pytest.mark.asyncio
    async def test_graph_node_wrappers(self) -> None:
        from src.agent.graph import _wrap_read_review, _wrap_refine

        with (
            patch("src.agent.nodes.read_review_node", new_callable=AsyncMock) as mock_rev,
            patch("src.agent.nodes.refine_from_review_node", new_callable=AsyncMock) as mock_refine,
        ):
            mock_rev.return_value = {"status": "ok"}
            mock_refine.return_value = {"status": "refined"}

            res_rev = await _wrap_read_review({"run_id": "r1"})  # type: ignore[arg-type]
            res_ref = await _wrap_refine({"run_id": "r1"})  # type: ignore[arg-type]
            assert res_rev == {"status": "ok"}
            assert res_ref == {"status": "refined"}

    def test_extract_generic_symbols_treesitter_mock(self) -> None:
        from src.agent.context import _extract_generic_symbols_treesitter

        mock_parser = MagicMock()
        mock_root = MagicMock()
        mock_child = MagicMock()
        mock_child.type = "class_declaration"
        name_node = MagicMock()
        name_node.text = b"OrderService"
        mock_child.child_by_field_name.return_value = name_node
        mock_child.start_point = (1, 0)
        mock_child.end_point = (10, 0)
        mock_child.children = []
        mock_root.children = [mock_child]
        mock_root.type = "program"

        mock_tree = MagicMock()
        mock_tree.root_node = mock_root
        mock_parser.parse.return_value = mock_tree

        symbols = _extract_generic_symbols_treesitter(
            "class OrderService {}", mock_parser, "typescript"
        )
        assert len(symbols) == 1
        assert symbols[0].name == "OrderService"
        assert symbols[0].kind == "class"


@pytest.mark.asyncio
class TestProcessorRefinementExecution:
    async def test_run_refinement_agent_async(self) -> None:
        from src.worker.processor import _run_refinement_agent_async

        mock_repo = AsyncMock()
        mock_redis = AsyncMock()

        with (
            patch("src.worker.processor.RunRepository", return_value=mock_repo),
            patch(
                "src.worker.processor.get_redis_connection",
                new_callable=AsyncMock,
                return_value=mock_redis,
            ),
            patch("src.worker.processor.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("src.worker.processor.release_active_job_lock", new_callable=AsyncMock),
        ):
            mock_agent.return_value = {
                "status": "succeeded",
                "pull_request": {"pr_url": "https://github.com/owner/repo/pull/42"},
                "retry_count": 0,
            }

            res = await _run_refinement_agent_async(
                repo_full_name="owner/repo",
                pr_number=42,
                comment_id=123,
                run_id="run-ref-test",
            )
            assert res["status"] == "succeeded"
            assert res["pr_url"] == "https://github.com/owner/repo/pull/42"

    async def test_persist_refinement_failure(self) -> None:
        from src.worker.processor import _persist_refinement_failure

        mock_repo = AsyncMock()
        mock_redis = AsyncMock()
        with (
            patch("src.worker.processor.RunRepository", return_value=mock_repo),
            patch(
                "src.worker.processor.get_redis_connection",
                new_callable=AsyncMock,
                return_value=mock_redis,
            ),
            patch(
                "src.worker.processor.release_active_job_lock", new_callable=AsyncMock
            ) as mock_release,
        ):
            await _persist_refinement_failure("run-fail-1", "owner/repo", 42, "Fatal error")
            assert mock_repo.update_run.called
            assert mock_release.called


class TestDeepCoverageAdditions:
    def test_record_run_metrics_with_cost(self) -> None:
        from src.observability.tracing import record_run_metrics

        final_state = {
            "status": "succeeded",
            "cumulative_cost_usd": 0.42,
            "total_tokens_used": 1000,
            "retry_count": 0,
            "pull_request": {"pr_url": "https://github.com/pr/1"},
            "is_fork": False,
        }
        record_run_metrics(final_state)

    def test_enqueue_review_refinement_sync(self) -> None:
        from src.worker.queue import _enqueue_review_refinement_sync

        mock_queue = MagicMock()
        mock_redis = MagicMock()

        with (
            patch("src.worker.queue.get_sync_redis", return_value=mock_redis),
            patch("src.worker.queue.Queue", return_value=mock_queue),
        ):
            _enqueue_review_refinement_sync(
                job_id="refine-job-1",
                repo_full_name="owner/repo",
                pr_number=42,
                comment_id=1,
                review_feedback="Fix issue",
                high_priority=True,
                timeout=1200,
            )
            assert mock_queue.enqueue.called

    @pytest.mark.asyncio
    async def test_enqueue_review_refinement_job_integrity_error(self) -> None:
        from sqlalchemy.exc import IntegrityError

        from src.worker.queue import enqueue_review_refinement_job

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)

        mock_repo = AsyncMock()
        mock_repo.create_run = AsyncMock(side_effect=IntegrityError("stmt", "params", Exception()))
        mock_repo.get_active_run = AsyncMock(return_value=SimpleNamespace(run_id="existing-run-99"))

        with (
            patch(
                "src.worker.queue.get_redis_connection",
                new_callable=AsyncMock,
                return_value=mock_redis,
            ),
            patch("src.worker.queue.RunRepository", return_value=mock_repo),
        ):
            active_id = await enqueue_review_refinement_job(
                repo_full_name="owner/repo",
                pr_number=42,
                comment_id=1,
                review_feedback="feedback",
            )
            assert active_id == "existing-run-99"

    @pytest.mark.asyncio
    async def test_create_pull_request_with_existing_pr(self) -> None:
        from src.agent.state import FilePatch
        from src.tools.github import GitHubClient

        client = GitHubClient()
        mock_gh = MagicMock()
        client._gh = mock_gh

        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_pr.number = 42
        mock_pr.head.ref = "existing-branch"
        mock_pr.base.ref = "main"
        mock_pr.head.sha = "abc123sha"
        mock_pr.html_url = "https://github.com/owner/repo/pull/42"
        mock_pr.title = "Existing PR"
        mock_pr.body = "Desc"
        mock_repo.get_pull.return_value = mock_pr
        mock_gh.get_repo.return_value = mock_repo

        mock_git_repo = MagicMock()
        mock_git_repo.is_dirty.return_value = True

        with (
            patch("git.Repo.clone_from", return_value=mock_git_repo),
            patch("src.tools.github._apply_patch_to_worktree"),
        ):
            patch_item = FilePatch(
                file_path="foo.py",
                action="modify",
                description="fix",
                unified_diff="--- a\n+++ b",
            )
            result_pr = await client.create_pull_request(
                repo_full_name="owner/repo",
                branch_name="ignore-this",
                base_branch=None,
                title="title",
                body="body",
                patches=[patch_item],
                issue_number=42,
                existing_pr_number=42,
            )
            assert result_pr.pr_number == 42
            assert result_pr.branch_name == "existing-branch"
            assert mock_pr.create_issue_comment.called
