Param(
  [Parameter(Mandatory=$false)]
  [string]$Config = "configs/dev.yaml"
)

$ErrorActionPreference = "Stop"

if (-not $Config) {
  throw "Config path is empty"
}

Write-Host "Running train with config: $Config"

poetry run antifraud train --config "$Config"
