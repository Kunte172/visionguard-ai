.PHONY: setup data train api web lint test verify

setup:
	uv sync --extra dev
	cd apps/web && npm install

data:
	uv run visionguard generate-sample --output data/sample --seed 42

train: data
	uv run visionguard train --manifest data/sample/manifest.csv

api:
	uv run uvicorn visionguard_ai.api:app --reload --host 127.0.0.1 --port 8000

web:
	cd apps/web && npm run dev

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src

test:
	uv run pytest

verify: lint test
	cd apps/web && npm run build
