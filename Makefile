.PHONY: help test lint types eval check index ask serve clean

PYTHON ?= python3
export PYTHONPATH := src

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-10s %s\n", $$1, $$2}'

test: ## run the test suite (no network, deterministic)
	$(PYTHON) -m pytest

lint: ## ruff
	$(PYTHON) -m ruff check .

types: ## mypy --strict
	$(PYTHON) -m mypy

eval: ## run the golden set for both strategies; fails on regression
	$(PYTHON) -m archivist.cli eval --strategy plain-rag agentic

check: lint types test eval ## everything CI runs

index: ## summarise the chunked corpus
	$(PYTHON) -m archivist.cli index

ask: ## ask a question: make ask Q="..."
	$(PYTHON) -m archivist.cli ask "$(Q)" -v

serve: ## run the HTTP service (requires the `service` extra)
	$(PYTHON) -m uvicorn --factory archivist.service.api:create_app --reload

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .archivist
