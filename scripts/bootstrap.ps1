$ErrorActionPreference = "Stop"

Write-Host "Installing dependencies via Poetry..."
poetry install

Write-Host "Running quality gates..."
poetry run ruff check .
poetry run black .
poetry run mypy .
poetry run pytest
