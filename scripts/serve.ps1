$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
& '.\.venv\Scripts\python.exe' -m robot_data_lab serve --port 8765
