Param(
  [string]$Out = "data/raw/transactions.csv",
  [int]$Rows = 5000,
  [int]$Seed = 42,
  [int]$Days = 30
)

$ErrorActionPreference = "Stop"

$projectRoot = (Get-Location).Path
$outPath = Join-Path $projectRoot $Out

Write-Host "Generating sample dataset:"
Write-Host "  Out : $outPath"
Write-Host "  Rows: $Rows"
Write-Host "  Seed: $Seed"
Write-Host "  Days: $Days"

$dir = Split-Path -Parent $outPath
if (-not (Test-Path $dir)) {
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

poetry run python scripts/generate_sample_data.py --out "$Out" --rows $Rows --seed $Seed --days $Days
