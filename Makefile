.PHONY: install dev db-up db-down migrate test lint format serve

install:
	poetry install

dev:
	poetry install --with dev

db-up:
	docker-compose up -d

db-down:
	docker-compose down

migrate:
	poetry run alembic upgrade head

test:
	poetry run pytest

lint:
	poetry run ruff check app cli tests
	poetry run black --check app cli tests

format:
	poetry run ruff check --fix app cli tests
	poetry run black app cli tests

serve:
	poetry run python -m cli serve
