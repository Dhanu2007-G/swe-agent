"""
src/agent/state.py — Typed state contracts for every node in the graph.
Using Pydantic models for validation + TypedDict for LangGraph compatibility.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ── Enumerations ──────────────────────────────────────────────────────────────

class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"  # PR opened as draft with failure notes


class ErrorCategory(StrEnum):
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    LOGIC_ERROR = "logic_error"
    TYPE_ERROR = "type_error"
    FIXTURE_ERROR = "fixture_error"
    TIMEOUT = "timeout"
    PATCH_APPLY_ERROR = "patch_apply_error"
    UNKNOWN = "unknown"


class TaskPriority(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Domain Models ─────────────────────────────────────────────────────────────

class GithubIssue(BaseModel):
    """Parsed GitHub issue with all context the agent needs."""
    issue_number: int
    repo_full_name: str  # "owner/repo"
    title: str
    body: str
    labels: list[str] = Field(default_factory=list)
    comments: list[str] = Field(default_factory=list)
    linked_prs: list[int] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)
    created_at: datetime
    html_url: str

    @property
    def owner(self) -> str:
        return self.repo_full_name.split("/")[0]

    @property
    def repo(self) -> str:
        return self.repo_full_name.split("/")[1]

    def to_prompt_context(self) -> str:
        """Serialize for LLM context — structured, not a raw dump."""
        parts = [
            f"## Issue #{self.issue_number}: {self.title}",
            f"**Repository:** {self.repo_full_name}",
            f"**Labels:** {', '.join(self.labels) or 'none'}",
            "",
            "### Description",
            self.body,
        ]
        if self.comments:
            parts += ["", "### Key Comments"]
            for i, comment in enumerate(self.comments[:5], 1):  # cap at 5
                parts.append(f"**Comment {i}:** {comment[:500]}")
        return "\n".join(parts)


class Task(BaseModel):
    """Atomic unit of work from the planner."""
    id: str
    description: str
    files_to_modify: list[str] = Field(default_factory=list)
    files_to_create: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_complexity: int = Field(default=3, ge=1, le=10)


class TaskPlan(BaseModel):
    """Structured output from the planner node."""
    summary: str
    tasks: list[Task]
    affected_test_files: list[str] = Field(default_factory=list)
    breaking_change_risk: bool = False
    requires_new_dependencies: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)


class FileContext(BaseModel):
    """A file's content loaded for the coder node."""
    path: str
    content: str
    language: str
    size_bytes: int
    token_count: int

    def truncated(self, max_tokens: int = 4000) -> "FileContext":
        """Return a truncated version if over token limit."""
        if self.token_count <= max_tokens:
            return self
        # Truncate to roughly max_tokens worth of content
        ratio = max_tokens / self.token_count
        cutoff = int(len(self.content) * ratio)
        return self.model_copy(update={
            "content": self.content[:cutoff] + "\n... [TRUNCATED]",
            "token_count": max_tokens,
        })


class FilePatch(BaseModel):
    """A single file change as a unified diff with optional full file replacement content."""
    file_path: str
    unified_diff: str = ""
    change_type: str = "modify"  # "modify" | "create" | "delete"
    description: str = ""
    full_content: str | None = None  # Complete file replacement fallback if diff fails


class CodePatch(BaseModel):
    """Structured output from the coder node."""
    patches: list[FilePatch]
    explanation: str
    test_command: str = "pytest"
    new_dependencies: list[str] = Field(default_factory=list)


class TestFailure(BaseModel):
    """A single test failure with full context."""
    __test__ = False
    test_id: str
    test_name: str
    error_message: str
    traceback: str
    error_category: ErrorCategory = ErrorCategory.UNKNOWN


class TestResult(BaseModel):
    """Structured output from running the test suite."""
    __test__ = False
    passed: bool
    total: int = 0
    failed_count: int = 0
    error_count: int = 0
    coverage_pct: float = 0.0
    failures: list[TestFailure] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False

    def failure_summary(self) -> str:
        """Compact summary for correction prompt."""
        if not self.failures:
            return "No failure details captured."
        lines = []
        for f in self.failures[:5]:  # top 5 failures max
            lines.append(f"### {f.test_name} [{f.error_category}]")
            lines.append(f"Error: {f.error_message}")
            if f.traceback:
                lines.append(f"Traceback (last 5 lines):")
                tb_lines = f.traceback.strip().split("\n")
                lines.extend(tb_lines[-5:])
            lines.append("")
        return "\n".join(lines)


class PullRequest(BaseModel):
    """Result of opening a PR."""
    pr_number: int
    pr_url: str
    branch_name: str
    is_draft: bool = False
    title: str
    body: str


class AttemptRecord(BaseModel):
    """Record of one patch attempt for the correction loop."""
    attempt_number: int
    patch: CodePatch
    test_result: TestResult
    error_category: ErrorCategory
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── LangGraph State ───────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    The single source of truth flowing through all graph nodes.
    Every field is optional (total=False) so nodes return partial updates.
    """
    # Input
    run_id: str
    issue: GithubIssue

    # Planning
    plan: TaskPlan

    # Coding
    current_task_index: int
    file_contexts: list[dict[str, Any]]  # serialized FileContext list
    code_patch: CodePatch

    # Testing
    test_result: TestResult

    # Correction loop
    retry_count: int
    attempt_history: list[dict[str, Any]]  # serialized AttemptRecord list
    last_error_category: str

    # Output
    pull_request: PullRequest
    status: str  # RunStatus value
    failure_reason: str

    # Metadata
    started_at: str
    completed_at: str
    total_tokens_used: int
