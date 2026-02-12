Param(
  [string]$Config = "configs/dev.yaml",
  [string]$HostAddress = "127.0.0.1",
  #[string]$Host = "127.0.0.1",

  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$env:ANTIFRAUD_CONFIG = $Config

poetry run uvicorn app.api.app:app --host $HostAddress --port $Port
#poetry run uvicorn app.api.app:app --host $Host --port $Port

