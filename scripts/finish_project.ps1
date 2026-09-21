param([int]$FocusedProcess = 0, [int]$AimdProcess = 0, [int]$PilotsProcess = 0, [int]$VisualProcess = 0)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path $PWD '.venv/Scripts/python.exe'
$env:PYTHONPATH = $PWD.Path
if ($FocusedProcess -gt 0) { Wait-Process -Id $FocusedProcess -ErrorAction SilentlyContinue }
if (-not (Test-Path reports/focused/md.json)) { throw 'Focused study has not finished; inspect focused_continue.err.log.' }
if ($PilotsProcess -gt 0) { Wait-Process -Id $PilotsProcess -ErrorAction SilentlyContinue }
$pilotsDone = (Test-Path reports/benchmark/pilot_completion.json) -and (Get-Content reports/benchmark/pilot_completion.json -Raw | ConvertFrom-Json).complete
if (-not $pilotsDone) {
    & $python -u scripts/resume_pilots.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($AimdProcess -gt 0) { Wait-Process -Id $AimdProcess -ErrorAction SilentlyContinue }
if ($VisualProcess -gt 0) { Wait-Process -Id $VisualProcess -ErrorAction SilentlyContinue }
$visualDone = (Test-Path reports/visualizations/complete.json) -and (Get-Content reports/visualizations/complete.json -Raw | ConvertFrom-Json).complete
if (-not $visualDone) {
    & $python -u scripts/validate_and_render.py --checkpoint runs/device/expanded_tensor/best.pt --data data/device/test.jsonl --output reports/visualizations --device cuda
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $python -u scripts/benchmark_inference.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -u scripts/paper_figures.py runs/device/expanded_tensor/best.pt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -u scripts/build_project_report.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
