"""
src/tools/detector.py — Repository language and test ecosystem auto-detector.

Inspects repository root files (e.g., package.json, go.mod, Cargo.toml, pyproject.toml)
to determine the primary language, test runner command, and package manager without
requiring manual user configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class RepoEcosystem:
    """Detected language ecosystem and test configuration for a repository."""

    language: str
    test_command: str
    package_manager: str
    manifest_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "test_command": self.test_command,
            "package_manager": self.package_manager,
            "manifest_files": self.manifest_files,
        }


def detect_repo_ecosystem(repo_dir: Path | str) -> RepoEcosystem:
    """
    Detect the programming language ecosystem and default test command for a repository.

    Priority order:
      1. Rust (Cargo.toml)
      2. Go (go.mod)
      3. JavaScript / TypeScript (package.json, tsconfig.json, lockfiles)
      4. Java (pom.xml, build.gradle)
      5. C / C++ (CMakeLists.txt)
      6. Python (pyproject.toml, setup.py, requirements.txt)
      7. Default fallback: Python / pytest
    """
    root = Path(repo_dir)

    # 1. Rust
    cargo_toml = root / "Cargo.toml"
    if cargo_toml.exists():
        log.info("detector.ecosystem_found", ecosystem="rust", root=str(root))
        return RepoEcosystem(
            language="rust",
            test_command="cargo test",
            package_manager="cargo",
            manifest_files=["Cargo.toml"],
        )

    # 2. Go
    go_mod = root / "go.mod"
    if go_mod.exists():
        log.info("detector.ecosystem_found", ecosystem="go", root=str(root))
        return RepoEcosystem(
            language="go",
            test_command="go test ./...",
            package_manager="go",
            manifest_files=["go.mod"],
        )

    # 3. JavaScript / TypeScript
    package_json = root / "package.json"
    if package_json.exists():
        manifests = ["package.json"]
        is_ts = (root / "tsconfig.json").exists()
        lang = "typescript" if is_ts else "javascript"
        if is_ts:
            manifests.append("tsconfig.json")

        pkg_mgr = "npm"
        test_cmd = "npm test"

        if (root / "pnpm-lock.yaml").exists():
            pkg_mgr = "pnpm"
            test_cmd = "pnpm test"
            manifests.append("pnpm-lock.yaml")
        elif (root / "yarn.lock").exists():
            pkg_mgr = "yarn"
            test_cmd = "yarn test"
            manifests.append("yarn.lock")
        elif (root / "bun.lockb").exists() or (root / "bun.lock").exists():
            pkg_mgr = "bun"
            test_cmd = "bun test"

        try:
            with open(package_json, encoding="utf-8") as f:
                data = json.load(f)
                scripts = data.get("scripts", {})
                if "test" in scripts:
                    script_val = scripts["test"]
                    if "vitest" in script_val or "jest" in script_val:
                        test_cmd = f"{pkg_mgr} test"
        except Exception as e:
            log.warning("detector.package_json_parse_error", error=str(e))

        log.info("detector.ecosystem_found", ecosystem=lang, pkg_mgr=pkg_mgr, root=str(root))
        return RepoEcosystem(
            language=lang,
            test_command=test_cmd,
            package_manager=pkg_mgr,
            manifest_files=manifests,
        )

    # 4. Java
    pom_xml = root / "pom.xml"
    if pom_xml.exists():
        log.info("detector.ecosystem_found", ecosystem="java", root=str(root))
        return RepoEcosystem(
            language="java",
            test_command="mvn test",
            package_manager="maven",
            manifest_files=["pom.xml"],
        )

    build_gradle = root / "build.gradle"
    if build_gradle.exists():
        log.info("detector.ecosystem_found", ecosystem="java", root=str(root))
        return RepoEcosystem(
            language="java",
            test_command="./gradlew test",
            package_manager="gradle",
            manifest_files=["build.gradle"],
        )

    # 5. C / C++ (CMakeLists.txt)
    cmake_lists = root / "CMakeLists.txt"
    if cmake_lists.exists():
        log.info("detector.ecosystem_found", ecosystem="cpp", root=str(root))
        return RepoEcosystem(
            language="cpp",
            test_command="ctest --output-on-failure",
            package_manager="cmake",
            manifest_files=["CMakeLists.txt"],
        )

    # 6. Python
    py_manifests = []
    for f_name in ("pyproject.toml", "setup.py", "requirements.txt", "Pipfile"):
        if (root / f_name).exists():
            py_manifests.append(f_name)

    if py_manifests:
        log.info("detector.ecosystem_found", ecosystem="python", root=str(root))
        return RepoEcosystem(
            language="python",
            test_command="pytest",
            package_manager="pip",
            manifest_files=py_manifests,
        )

    # Default fallback
    log.info("detector.ecosystem_default", ecosystem="python", root=str(root))
    return RepoEcosystem(
        language="python",
        test_command="pytest",
        package_manager="pip",
        manifest_files=[],
    )
