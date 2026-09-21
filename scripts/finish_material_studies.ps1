param([int]$TrainingProcess)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($TrainingProcess -gt 0) { Wait-Process -Id $TrainingProcess -ErrorAction SilentlyContinue }
& .venv/Scripts/python.exe -u scripts/publish_material_studies.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
