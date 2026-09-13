"""
src/tools/github.py — Production GitHub client.
All operations wrapped with retry, rate-limit handling, and structured errors.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

import git
import structlog
from github import Github, GithubException
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from pathlib import Path

    from github.PullRequest import PullRequest as GHPullRequest

from src.agent.state import FilePatch, GithubIssue, PullRequest
from src.config import get_settings

log = structlog.get_logger(__name__)


class GitHubRateLimitError(Exception):
    pass


class GitHubNotFoundError(Exception):
    pass


def _build_authenticated_clone_kwargs(repo_full_name: str, settings: Any = None) -> dict[str, Any]:
    """Helper to build clone kwargs with authentication."""
    import base64

    s = settings or get_settings()
    token = getattr(s, "github_token_value", getattr(s, "github_token", ""))
    auth_bytes = f"x-access-token:{token}".encode()
    auth_b64 = base64.b64encode(auth_bytes).decode()
    url = f"https://github.com/{repo_full_name}.git"
    return {
        "url": url,
        "multi_options": ["-c", f"http.extraheader=AUTHORIZATION: basic {auth_b64}"],
    }


class GitHubClient:
    """
    Async-friendly GitHub client.
    Uses PyGitHub for REST + httpx for raw operations.
    All blocking PyGitHub calls are executed in threadpool.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._gh: Github | None = None

    async def __aenter__(self) -> GitHubClient:
        self._gh = Github(
            self._settings.github_token_value,
            retry=3,
            per_page=100,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._gh:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._gh.close)

    # ── Issue Operations ──────────────────────────────────────────────────────

    async def get_issue(self, repo_full_name: str, issue_number: int) -> GithubIssue:
        """Fetch and parse a GitHub issue into our domain model."""

        def _fetch() -> GithubIssue:
            assert self._gh is not None
            repo = self._gh.get_repo(repo_full_name)
            issue = repo.get_issue(issue_number)

            comments = [c.body for c in issue.get_comments() if c.body and len(c.body) > 10][:10]

            linked_prs: list[int] = []
            try:
                for event in issue.get_timeline():
                    if event.event == "cross-referenced" and event.source and event.source.issue:
                        linked_prs.append(event.source.issue.number)
            except Exception:
                pass

            return GithubIssue(
                issue_number=issue.number,
                repo_full_name=repo_full_name,
                title=issue.title,
                body=issue.body or "",
                labels=[label.name for label in issue.labels],
                comments=comments,
                linked_prs=linked_prs,
                assignees=[a.login for a in issue.assignees],
                created_at=issue.created_at,
                html_url=issue.html_url,
            )

        return cast("GithubIssue", await self._with_retry(_fetch))

    async def comment_on_issue(
        self,
        repo_full_name: str,
        issue_number: int,
        body: str,
    ) -> None:
        """Post a comment on an issue."""

        def _comment() -> None:
            assert self._gh is not None
            repo = self._gh.get_repo(repo_full_name)
            issue = repo.get_issue(issue_number)
            issue.create_comment(body)

        await self._with_retry(_comment)

    # ── Pull Request Operations ───────────────────────────────────────────────

    async def create_pull_request(
        self,
        repo_full_name: str,
        branch_name: str,
        base_branch: str | None = None,
        title: str = "",
        body: str = "",
        patches: list[FilePatch] | None = None,
        issue_number: int = 0,
        labels: list[str] | None = None,
        draft: bool = False,
    ) -> PullRequest:
        """
        Create branch, commit patches, and open a PR.
        Steps: clone → checkout branch → apply patches → commit → push → open PR.
        """
        settings = self._settings

        def _create() -> PullRequest:
            import tempfile
            from pathlib import Path

            assert self._gh is not None
            gh_repo = self._gh.get_repo(repo_full_name)

            # Determine base branch
            target_base = base_branch or gh_repo.default_branch

            # Get default branch SHA for branching
            default_sha = gh_repo.get_branch(target_base).commit.sha

            # Create remote branch via API
            try:
                gh_repo.create_git_ref(
                    ref=f"refs/heads/{branch_name}",
                    sha=default_sha,
                )
            except GithubException as e:
                if getattr(e, "status", None) != 422:  # 422 = branch already exists
                    raise

            # Clone locally to apply patches
            with tempfile.TemporaryDirectory() as tmpdir:
                clone_kwargs = _build_authenticated_clone_kwargs(repo_full_name, settings)
                clone_url = clone_kwargs.pop("url", f"https://github.com/{repo_full_name}.git")
                repo = git.Repo.clone_from(
                    clone_url, tmpdir, allow_unsafe_options=True, **clone_kwargs
                )
                repo.git.checkout("-b", branch_name)

                # Configure git identity
                repo.config_writer().set_value(
                    "user", "name", settings.github_bot_username
                ).release()
                repo.config_writer().set_value(
                    "user", "email", f"{settings.github_bot_username}@users.noreply.github.com"
                ).release()

                # Apply each patch
                for patch in patches or []:
                    _apply_patch_to_worktree(Path(tmpdir), patch)

                # Commit all changes
                repo.git.add(A=True)
                if repo.is_dirty():
                    repo.index.commit(
                        f"fix: automated fix for #{issue_number}\n\n"
                        f"Applied by swe-agent\n"
                        f"Issue: https://github.com/{repo_full_name}/issues/{issue_number}"
                    )

                # Push
                origin = repo.remote("origin")
                origin.push(refspec=f"{branch_name}:{branch_name}", force=False)

            # Open PR via API
            gh_pr: GHPullRequest = gh_repo.create_pull(
                title=title,
                body=body,
                head=branch_name,
                base=target_base,
                draft=draft,
            )

            # Add labels
            if labels:
                existing = [lbl.name for lbl in gh_repo.get_labels()]
                for label in labels:
                    if label not in existing:
                        with suppress(GithubException):
                            gh_repo.create_label(label, "0075ca")
                gh_pr.add_to_labels(*labels)

            # Link to issue via comment
            gh_pr.create_issue_comment(
                f"This PR was automatically generated to resolve #{issue_number}.\n"
                f"Closes #{issue_number}"
            )

            return PullRequest(
                pr_number=gh_pr.number,
                pr_url=gh_pr.html_url,
                branch_name=branch_name,
                is_draft=draft,
                title=title,
                body=body,
            )

        return cast("PullRequest", await self._with_retry(_create))

    @staticmethod
    def validate_webhook_signature(payload: bytes, signature_header: str) -> bool:
        """
        Validate GitHub webhook HMAC-SHA256 signature.
        Constant-time comparison to prevent timing attacks.
        The secret is sourced exclusively from environment/settings — never hardcoded.
        """
        if not signature_header or not signature_header.startswith("sha256="):
            return False

        received = signature_header.removeprefix("sha256=")
        keys_to_try: list[str] = []

        # Primary: explicit environment variable override
        env_secret = os.environ.get("GITHUB_WEBHOOK_SECRET")
        if env_secret:
            keys_to_try.append(env_secret)

        # Secondary: configured Pydantic settings (reads from .env / secrets)
        try:
            settings = get_settings()
            sec_val = getattr(settings, "github_webhook_secret_value", None)
            if sec_val is None:
                sec_val = getattr(settings, "github_webhook_secret", None)
            if sec_val and isinstance(sec_val, str) and sec_val not in keys_to_try:
                keys_to_try.append(sec_val)
        except Exception:
            pass

        if not keys_to_try:
            # No secret configured — reject all incoming webhooks for safety
            log.error(
                "webhook.signature_validation_failed",
                reason="no_secret_configured",
            )
            return False

        for key in keys_to_try:
            try:
                expected = hmac.new(
                    key=key.encode("utf-8"),
                    msg=payload,
                    digestmod=hashlib.sha256,
                ).hexdigest()
                if hmac.compare_digest(expected, received):
                    return True
            except Exception:
                continue

        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    async def _with_retry(fn_or_coro: Any) -> Any:
        """Wrap any coroutine or callable with exponential backoff for GitHub API errors."""
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(GithubException),
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        ):
            with attempt:
                if callable(fn_or_coro):
                    res = fn_or_coro()
                    if hasattr(res, "__await__"):
                        return await res
                    return res
                if hasattr(fn_or_coro, "__await__"):
                    return await fn_or_coro
                return fn_or_coro


# ── Patch Application Helper ──────────────────────────────────────────────────


def _apply_patch_to_worktree(root: Path, patch: FilePatch) -> None:
    """Apply a FilePatch to a local directory using the `patch` command with full-file fallback."""
    import subprocess

    target = root / patch.file_path

    if patch.change_type == "create":
        target.parent.mkdir(parents=True, exist_ok=True)
        if patch.full_content is not None:
            target.write_text(patch.full_content, encoding="utf-8")
        else:
            lines = [
                line[1:]
                for line in patch.unified_diff.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            ]
            target.write_text("\n".join(lines), encoding="utf-8")
        return

    if patch.change_type == "delete":
        if target.exists():
            target.unlink()
        return

    if patch.full_content is not None and not patch.unified_diff.strip():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patch.full_content, encoding="utf-8")
        return

    # Apply unified diff via patch command
    result = subprocess.run(
        ["patch", "-p1", "--forward", "--fuzz=3"],
        input=patch.unified_diff.encode(),
        capture_output=True,
        cwd=str(root),
        timeout=30,
    )

    if result.returncode != 0:
        if patch.full_content is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(patch.full_content, encoding="utf-8")
            return
        error = result.stderr.decode(errors="replace")
        raise RuntimeError(f"Patch failed for {patch.file_path}: {error}")
