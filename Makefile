# Quarry-LDR build targets. Every target works on Linux, WSL2, and Windows
# (GNU make with cmd.exe or sh). Recipes are single plain commands on purpose:
# anything with logic lives in a Python script, not in shell.

ifeq ($(OS),Windows_NT)
BOOTSTRAP = powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
else
BOOTSTRAP = bash scripts/bootstrap.sh
endif

.PHONY: bootstrap test verify fmt lint searxng searxng-down smoke smoke-local audit fixtures

bootstrap:
	$(BOOTSTRAP)

test:
	uv run pytest

verify:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy
	uv run pytest

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run mypy

searxng:
	uv run quarry searxng up

searxng-down:
	uv run quarry searxng down

smoke:
	uv run python scripts/smoke.py

smoke-local:
	uv run python scripts/smoke.py --engine local

audit:
	uv run python scripts/pre_public_audit.py

fixtures:
	uv run python scripts/make_fixtures.py
