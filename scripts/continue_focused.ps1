$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path $PWD '.venv/Scripts/python.exe'
$env:PYTHONPATH = $PWD.Path
$stages = @(
    @('scripts/run_focused.py', '--stage', 'screen'),
    @('scripts/run_focused.py', '--stage', 'confirm'),
    @('scripts/finish_focused.py', '--md')
)
foreach ($stage in $stages) {
    & $python -u @stage
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
