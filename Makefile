.PHONY: install test test-unit test-all lint format lean-build clean

UV := uv

install:
	$(UV) sync --extra dev

install-all:
	$(UV) sync --extra dev --extra llm --extra viz

test: test-unit

test-unit:
	$(UV) run pytest tests/unit -q

test-all:
	$(UV) run pytest tests -q

test-fast:
	$(UV) run pytest tests/unit -q -n auto

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

typecheck:
	$(UV) run mypy src/covcal

lean-build:
	cd lean && lake build

lean-cache:
	cd lean && lake exe cache get

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
