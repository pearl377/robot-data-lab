$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-local.txt
if ($LASTEXITCODE -ne 0) { throw 'dependency installation failed' }
& '.\.venv\Scripts\python.exe' -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw 'project installation failed' }
& '.\.venv\Scripts\python.exe' -m robot_data_lab doctor
