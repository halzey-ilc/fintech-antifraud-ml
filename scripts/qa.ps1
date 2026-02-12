Param(
  [switch]$Fix
)

$ErrorActionPreference = "Stop"

function Ensure-Poetry {
  $poetry = Get-Command poetry -ErrorAction SilentlyContinue
  if (-not $poetry) {
    throw "Poetry not found. Install Poetry and ensure it's on PATH."
  }
}

Ensure-Poetry

Write-Host "==> Installing dependencies (poetry install)"
poetry install

if ($Fix) {
  Write-Host "==> Running Ruff (fix) + Black (format)"
  poetry run ruff check --fix .
  poetry run black .
} else {
  Write-Host "==> Running Ruff (check) + Black (check)"
  poetry run ruff check .
  poetry run black --check .
}

Write-Host "==> Running mypy"
poetry run mypy .

Write-Host "==> Running tests"
poetry run pytest

Write-Host "==> OK"
