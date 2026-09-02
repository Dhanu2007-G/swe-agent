"""
tests/unit/test_prompts.py — Prompt quality and completeness tests.
Prompts are the contract between our system and the LLM.
Test them like any other interface.
"""
from __future__ import annotations

import pytest

from src.agent.prompts import (
    CODER_SYSTEM,
    CODER_USER,
    CORRECTOR_SYSTEM,
    CORRECTOR_USER,
    DRAFT_PR_BODY,
    ERROR_CLASSIFIER_SYSTEM,
    PLANNER_SYSTEM,
    PLANNER_USER,
    PR_BODY_USER,
)


class TestPlannerPrompt:
    def test_demands_exact_file_paths(self) -> None:
        assert "EXACTLY" in PLANNER_SYSTEM or "exact" in PLANNER_SYSTEM.lower()

    def test_caps_tasks_at_five(self) -> None:
        assert "5" in PLANNER_SYSTEM or "five" in PLANNER_SYSTEM.lower()

    def test_demands_json_output_only(self) -> None:
        """Must tell LLM to output ONLY JSON — no preamble."""
        assert "ONLY" in PLANNER_SYSTEM
        assert "JSON" in PLANNER_SYSTEM

    def test_user_template_has_all_slots(self) -> None:
        required = ["{issue_context}", "{repo_tree}", "{test_files}"]
        for slot in required:
            assert slot in PLANNER_USER, f"Missing slot: {slot}"

    def test_no_freeform_prose_instruction(self) -> None:
        """Planner should not invite LLM to 'explain' things."""
        bad_phrases = ["feel free", "you can also", "optionally"]
        for phrase in bad_phrases:
            assert phrase.lower() not in PLANNER_SYSTEM.lower(), (
                f"Permissive phrase found: '{phrase}'"
            )


class TestCoderPrompt:
    def test_demands_unified_diff_format(self) -> None:
        assert "unified diff" in CODER_SYSTEM.lower() or "git diff" in CODER_SYSTEM.lower()

    def test_forbids_full_file_output(self) -> None:
        assert "full file" in CODER_SYSTEM.lower() or "Never output full" in CODER_SYSTEM

    def test_demands_json_only(self) -> None:
        assert "ONLY" in CODER_SYSTEM
        assert "JSON" in CODER_SYSTEM

    def test_user_template_has_all_slots(self) -> None:
        required = [
            "{task_description}",
            "{acceptance_criteria}",
            "{file_contexts}",
            "{issue_summary}",
        ]
        for slot in required:
            assert slot in CODER_USER, f"Missing slot: {slot}"

    def test_mentions_import_ordering(self) -> None:
        """Coder must be told how to handle imports to avoid style divergence."""
        assert "import" in CODER_SYSTEM.lower()


class TestCorrectorPrompt:
    def test_demands_root_cause_analysis(self) -> None:
        assert "ROOT CAUSE" in CORRECTOR_SYSTEM or "root cause" in CORRECTOR_SYSTEM.lower()

    def test_forbids_repeating_previous_attempts(self) -> None:
        """Critical anti-loop instruction."""
        assert "DO NOT REPEAT" in CORRECTOR_USER or "DO NOT repeat" in CORRECTOR_USER

    def test_user_template_has_all_slots(self) -> None:
        required = [
            "{task_description}",
            "{applied_patch}",
            "{failure_summary}",
            "{previous_attempts}",
        ]
        for slot in required:
            assert slot in CORRECTOR_USER, f"Missing slot: {slot}"

    def test_includes_previous_attempts_slot(self) -> None:
        """This is the key anti-loop mechanism — must be in the prompt."""
        assert "{previous_attempts}" in CORRECTOR_USER

    def test_demands_json_only(self) -> None:
        assert "JSON" in CORRECTOR_SYSTEM


class TestErrorClassifierPrompt:
    EXPECTED_CATEGORIES = [
        "syntax_error",
        "import_error",
        "logic_error",
        "type_error",
        "fixture_error",
        "timeout",
        "patch_apply_error",
        "unknown",
    ]

    def test_all_categories_listed(self) -> None:
        for cat in self.EXPECTED_CATEGORIES:
            assert cat in ERROR_CLASSIFIER_SYSTEM, (
                f"Category '{cat}' missing from error classifier prompt"
            )

    def test_demands_single_word_output(self) -> None:
        """Classifier must produce a single category string, not an explanation."""
        assert "ONLY" in ERROR_CLASSIFIER_SYSTEM or "nothing else" in ERROR_CLASSIFIER_SYSTEM.lower()


class TestDraftPRBody:
    def test_has_all_template_slots(self) -> None:
        required = [
            "{max_retries}",
            "{issue_number}",
            "{changes_summary}",
            "{failure_summary}",
        ]
        for slot in required:
            assert slot in DRAFT_PR_BODY, f"Missing slot: {slot}"

    def test_marked_as_draft_clearly(self) -> None:
        """Human reviewers must immediately know this is a failed automated attempt."""
        assert "draft" in DRAFT_PR_BODY.lower() or "partial" in DRAFT_PR_BODY.lower()
        assert "human" in DRAFT_PR_BODY.lower()

    def test_explains_next_steps(self) -> None:
        assert "Next steps" in DRAFT_PR_BODY or "next step" in DRAFT_PR_BODY.lower()


class TestPRBodyTemplate:
    def test_has_all_template_slots(self) -> None:
        required = [
            "{issue_number}",
            "{issue_title}",
            "{issue_url}",
            "{changes_summary}",
            "{total_tests}",
            "{coverage_pct}",
        ]
        for slot in required:
            assert slot in PR_BODY_USER, f"Missing slot: {slot}"


class TestPromptLengths:
    """Prompts that are too long waste tokens. Prompts too short miss context."""

    def test_planner_system_is_substantial(self) -> None:
        assert len(PLANNER_SYSTEM) > 200, "Planner system prompt too short"

    def test_coder_system_is_substantial(self) -> None:
        assert len(CODER_SYSTEM) > 200, "Coder system prompt too short"

    def test_corrector_system_is_substantial(self) -> None:
        assert len(CORRECTOR_SYSTEM) > 100, "Corrector system prompt too short"

    def test_no_prompt_over_2000_chars_in_system(self) -> None:
        """System prompts > 2k chars consume context on every call — keep lean."""
        long_prompts = {
            "PLANNER_SYSTEM": PLANNER_SYSTEM,
            "CODER_SYSTEM": CODER_SYSTEM,
            "CORRECTOR_SYSTEM": CORRECTOR_SYSTEM,
        }
        for name, prompt in long_prompts.items():
            assert len(prompt) < 3000, (
                f"{name} is {len(prompt)} chars — trim it (target < 3000)"
            )
