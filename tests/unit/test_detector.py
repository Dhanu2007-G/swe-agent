from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from src.tools.detector import RepoEcosystem, detect_repo_ecosystem


def test_repo_ecosystem_to_dict() -> None:
    eco = RepoEcosystem(
        language="python",
        test_command="pytest",
        package_manager="pip",
        manifest_files=["pyproject.toml"],
    )
    d = eco.to_dict()
    assert d["language"] == "python"
    assert d["test_command"] == "pytest"
    assert d["package_manager"] == "pip"
    assert d["manifest_files"] == ["pyproject.toml"]


def test_detect_rust(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'test'\n")
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.language == "rust"
    assert eco.test_command == "cargo test"
    assert eco.package_manager == "cargo"
    assert "Cargo.toml" in eco.manifest_files


def test_detect_go(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/test\n")
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.language == "go"
    assert eco.test_command == "go test ./..."
    assert eco.package_manager == "go"
    assert "go.mod" in eco.manifest_files


def test_detect_js_npm(tmp_path: Path) -> None:
    pkg = {"name": "test-pkg", "scripts": {"test": "jest"}}
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.language == "javascript"
    assert eco.test_command == "npm test"
    assert eco.package_manager == "npm"


def test_detect_ts_pnpm(tmp_path: Path) -> None:
    pkg = {"name": "test-ts", "scripts": {"test": "vitest run"}}
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    (tmp_path / "tsconfig.json").write_text("{}")
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: 5.4")
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.language == "typescript"
    assert eco.test_command == "pnpm test"
    assert eco.package_manager == "pnpm"
    assert "tsconfig.json" in eco.manifest_files


def test_detect_js_yarn(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "mocha"}}))
    (tmp_path / "yarn.lock").write_text("")
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.package_manager == "yarn"
    assert eco.test_command == "yarn test"


def test_detect_js_bun(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(json.dumps({}))
    (tmp_path / "bun.lockb").write_text("")
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.package_manager == "bun"
    assert eco.test_command == "bun test"


def test_detect_js_bad_package_json(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{invalid json")
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.language == "javascript"
    assert eco.test_command == "npm test"


def test_detect_java_maven(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text("<project></project>")
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.language == "java"
    assert eco.test_command == "mvn test"
    assert eco.package_manager == "maven"


def test_detect_java_gradle(tmp_path: Path) -> None:
    (tmp_path / "build.gradle").write_text("plugins {}")
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.language == "java"
    assert eco.test_command == "./gradlew test"
    assert eco.package_manager == "gradle"


def test_detect_cpp(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.14)\n")
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.language == "cpp"
    assert eco.test_command == "ctest --output-on-failure"
    assert eco.package_manager == "cmake"
    assert "CMakeLists.txt" in eco.manifest_files


def test_detect_python_manifests(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'pytest'\n")
    (tmp_path / "requirements.txt").write_text("pytest\n")
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.language == "python"
    assert eco.test_command == "pytest"
    assert eco.package_manager == "pip"
    assert "pyproject.toml" in eco.manifest_files
    assert "requirements.txt" in eco.manifest_files


def test_detect_default_fallback(tmp_path: Path) -> None:
    eco = detect_repo_ecosystem(tmp_path)
    assert eco.language == "python"
    assert eco.test_command == "pytest"
    assert eco.package_manager == "pip"
    assert eco.manifest_files == []
