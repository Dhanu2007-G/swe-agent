# ═══════════════════════════════════════════════════════════════════════════════
# SWE Agent — Developer Makefile
# Run `make help` to see all available commands.
# ═══════════════════════════════════════════════════════════════════════════════

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-unit test-integration evals \
        up down logs migrate sandbox-build clean migrate-validate

PYTHON ?= python3

# ── Setup ─────────────────────────────────────────────────────────────────────

install: ## Install all dependencies including dev extras
	pip install -e ".[dev]"
	pre-commit install

install-prod: ## Install production dependencies only
	pip install -e .

# ── Code Quality ──────────────────────────────────────────────────────────────

lint: ## Run ruff linter
	ruff check .

lint-fix: ## Run ruff linter and auto-fix
	ruff check --fix .

format: ## Run ruff formatter
	ruff format .

format-check: ## Check formatting without applying
	ruff format --check .

typecheck: ## Run mypy type checker
	mypy src/ --ignore-missing-imports

migrate-validate: ## Validate Alembic migrations graph has exactly one head
	$(PYTHON) scripts/verify_migrations.py

check: lint format-check typecheck migrate-validate ## Run all quality checks

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## Run all tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=95

test-unit: ## Run unit tests only (fast, no external dependencies)
	pytest tests/unit/ -v --cov=src --cov-report=term-missing

test-integration: ## Run integration tests (requires Redis, mocked LLM)
	pytest tests/integration/ -v

test-watch: ## Run unit tests in watch mode (requires pytest-watch)
	ptw tests/unit/ -- -v

evals: ## Run eval suite in dry-run mode (safe, no real API calls)
	$(PYTHON) tests/evals/eval_runner.py --dry-run

evals-real: ## Run eval suite against real APIs (costs money, opens real PRs)
	@echo "WARNING: This will make real API calls and open real GitHub PRs."
	@read -p "Are you sure? [y/N] " ans && [ $${ans:-N} = y ]
	$(PYTHON) tests/evals/eval_runner.py --no-dry-run

# ── Docker ────────────────────────────────────────────────────────────────────

up: ## Start full stack (API + workers + Redis + Postgres)
	docker compose up --build

up-dev: ## Start stack with dev extras (RQ dashboard, etc.)
	docker compose --profile dev up --build

up-observability: ## Start full stack including Prometheus + Grafana
	docker compose --profile observability up --build

down: ## Stop all services
	docker compose down

down-volumes: ## Stop all services and delete volumes (DESTRUCTIVE)
	docker compose down -v

logs: ## Follow logs for all services
	docker compose logs -f

logs-api: ## Follow API logs only
	docker compose logs -f api

logs-worker: ## Follow worker logs only
	docker compose logs -f worker

# ── Database ──────────────────────────────────────────────────────────────────

migrate: ## Run all pending Alembic migrations
	alembic upgrade head

migrate-check: ## Show pending migrations without applying
	alembic current
	alembic history

migrate-rollback: ## Rollback one migration
	alembic downgrade -1

migrate-new: ## Create a new migration (requires MSG variable)
	@test -n "$(MSG)" || (echo "Usage: make migrate-new MSG='describe the change'" && exit 1)
	alembic revision --autogenerate -m "$(MSG)"

# ── Sandbox ───────────────────────────────────────────────────────────────────

sandbox-build: ## Build the sandbox Docker image
	docker build --target sandbox -t swe-agent-sandbox:latest .

sandbox-check: ## Verify sandbox security settings
	$(PYTHON) -m src.cli sandbox-check

# ── CLI ───────────────────────────────────────────────────────────────────────

run-issue: ## Trigger a local run (requires REPO and ISSUE vars)
	@test -n "$(REPO)" || (echo "Usage: make run-issue REPO=owner/repo ISSUE=42" && exit 1)
	@test -n "$(ISSUE)" || (echo "Usage: make run-issue REPO=owner/repo ISSUE=42" && exit 1)
	$(PYTHON) -m src.cli run $(REPO) $(ISSUE)

run-dry: ## Dry run (no LLM calls) — requires REPO and ISSUE
	@test -n "$(REPO)" || (echo "Usage: make run-dry REPO=owner/repo ISSUE=42" && exit 1)
	@test -n "$(ISSUE)" || (echo "Usage: make run-dry REPO=owner/repo ISSUE=42" && exit 1)
	$(PYTHON) -m src.cli run $(REPO) $(ISSUE) --dry-run

show-config: ## Show current config (secrets redacted)
	$(PYTHON) -m src.cli config-show

list: ## List recent runs
	$(PYTHON) -m src.cli list-runs

demo: ## Launch the interactive demo UI at http://localhost:3000
	$(PYTHON) scripts/run_demo.py

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts, caches, and temp files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov/ dist/ build/ *.egg-info/

clean-docker: ## Remove stopped containers and dangling images
	docker container prune -f
	docker image prune -f

# ── Help ──────────────────────────────────────────────────────────────────────

help: ## Show this help
	@echo ""
	@echo "  SWE Agent — Developer Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
