param([switch]$Offline)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$runArgs = @('-m', 'robot_data_lab', 'run-local')
if ($Offline) { $runArgs += '--offline' }
& '.\.venv\Scripts\python.exe' @runArgs
if ($LASTEXITCODE -ne 0) { throw 'pipeline failed; inspect artifacts/pipeline.json' }
