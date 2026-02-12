Param(
  [string]$Config = "configs/dev.yaml"
)

$ErrorActionPreference = "Stop"
poetry run antifraud eval --config $Config
