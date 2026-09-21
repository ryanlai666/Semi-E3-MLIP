$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path $PWD '.venv/Scripts/python.exe'
$env:PYTHONPATH = $PWD.Path
$first = Get-Content reports/benchmark/documentation.json -Raw | ConvertFrom-Json
if (-not $first.complete) { throw 'Finish the first benchmark and timing study before using the GPU.' }
& $python -u scripts/material_studies.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
