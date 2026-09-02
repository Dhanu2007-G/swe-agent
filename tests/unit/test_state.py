from __future__ import annotations

from datetime import UTC, datetime

from src.agent.state import ErrorCategory, FileContext, GithubIssue, TestFailure, TestResult


class TestGithubIssue:
    def test_owner_and_repo_properties_split_full_name(self) -> None:
        issue = GithubIssue(
            issue_number=42,
            repo_full_name="owner/repo",
            title="Bug",
            body="Fix it",
            labels=["bug"],
            comments=[],
            linked_prs=[],
            assignees=[],
            created_at=datetime.now(UTC),
            html_url="https://github.com/owner/repo/issues/42",
        )

        assert issue.owner == "owner"
        assert issue.repo == "repo"

    def test_to_prompt_context_includes_limited_comments(self) -> None:
        issue = GithubIssue(
            issue_number=42,
            repo_full_name="owner/repo",
            title="Bug",
            body="Fix it",
            labels=["bug", "agent-fix"],
            comments=[
                "comment one",
                "comment two",
                "comment three",
                "comment four",
                "comment five",
                "comment six",
            ],
            linked_prs=[],
            assignees=[],
            created_at=datetime.now(UTC),
            html_url="https://github.com/owner/repo/issues/42",
        )

        context = issue.to_prompt_context()

        assert "## Issue #42: Bug" in context
        assert "**Repository:** owner/repo" in context
        assert "**Labels:** bug, agent-fix" in context
        assert "### Key Comments" in context
        assert "**Comment 5:** comment five" in context
        assert "comment six" not in context


class TestFileContext:
    def test_truncated_returns_self_when_within_limit(self) -> None:
        context = FileContext(
            path="src/app.py",
            content="print('ok')",
            language="python",
            size_bytes=11,
            token_count=5,
        )

        assert context.truncated(max_tokens=10) is context

    def test_truncated_returns_shortened_copy_when_over_limit(self) -> None:
        context = FileContext(
            path="src/app.py",
            content="abcdef" * 20,
            language="python",
            size_bytes=120,
            token_count=20,
        )

        truncated = context.truncated(max_tokens=10)

        assert truncated is not context
        assert truncated.token_count == 10
        assert truncated.content.endswith("\n... [TRUNCATED]")


class TestTestResult:
    def test_failure_summary_returns_default_when_no_failures(self) -> None:
        result = TestResult(passed=False, failures=[])

        assert result.failure_summary() == "No failure details captured."

    def test_failure_summary_includes_last_five_traceback_lines(self) -> None:
        result = TestResult(
            passed=False,
            failures=[
                TestFailure(
                    test_id="tests/test_app.py::test_bug",
                    test_name="test_bug",
                    error_message="AssertionError: boom",
                    traceback="line1\nline2\nline3\nline4\nline5\nline6",
                    error_category=ErrorCategory.LOGIC_ERROR,
                )
            ],
        )

        summary = result.failure_summary()

        assert "### test_bug [logic_error]" in summary
        assert "Error: AssertionError: boom" in summary
        assert "line1" not in summary
        assert "line6" in summary


class TestFilePatch:
    def test_file_patch_with_full_content(self) -> None:
        from src.agent.state import FilePatch

        patch = FilePatch(
            file_path="src/new_file.py",
            change_type="create",
            description="Add new file",
            full_content="def new_func(): return True\n",
        )
        assert patch.full_content == "def new_func(): return True\n"
        assert patch.change_type == "create"
        assert patch.unified_diff == ""
