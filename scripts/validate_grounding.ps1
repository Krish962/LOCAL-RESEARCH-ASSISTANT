param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$python = Join-Path (Get-Location) '.venv311/Scripts/python.exe'
$env:PYTHONUTF8='1'
$env:HF_HUB_OFFLINE='1'
foreach ($file in @($python,'evaluation/grounding_questions.jsonl','evaluation/grounding_confirmation.jsonl','docs/baseline/pre_grounding/inventory.json')) {
    if (-not (Test-Path -LiteralPath $file)) { throw "Required file missing: $file" }
}
if ($CheckOnly) { Write-Output 'Grounding validation entry points and frozen questions are present.'; return }
$active = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.ExecutablePath -like ((Get-Location).Path+'\*') -and $_.CommandLine -match '-m\s+evaluation\.(run_grounding_validation|grounding_worker|run_answers|run_ragas)'
})
if ($active.Count) { throw 'A project evaluation is already running. Wait for it before starting another.' }
& (Join-Path $PSScriptRoot 'start_ollama.ps1')
& $python -X utf8 -m scripts.prepare_grounding
if ($LASTEXITCODE -ne 0) { throw 'Grounding model preparation failed.' }
& $python -X utf8 -m pytest -q --tb=short
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& $python -X utf8 -u -m evaluation.run_grounding_regressions
if ($LASTEXITCODE -ne 0) { throw 'Previously observed failure regression failed.' }
if (Test-Path -LiteralPath 'results/grounding-validation-34c5d8c856b9/baseline_answers.jsonl') {
    & $python -X utf8 -u -m evaluation.run_grounding_validation --arm grounded --reuse-baseline results/grounding-validation-34c5d8c856b9
} else {
    & $python -X utf8 -u -m evaluation.run_grounding_validation
}
if ($LASTEXITCODE -ne 0) { throw 'New grounding validation failed.' }
& $python -X utf8 -m evaluation.summarize_grounding --questions evaluation/grounding_questions.jsonl
if ($LASTEXITCODE -ne 0) { throw 'Grounding comparison aggregation failed.' }
if (Test-Path -LiteralPath 'results/grounding-validation-0c1235147e8a/baseline_answers.jsonl') {
    & $python -X utf8 -u -m evaluation.run_grounding_validation --questions evaluation/grounding_confirmation.jsonl --arm grounded --reuse-baseline results/grounding-validation-0c1235147e8a
} else {
    & $python -X utf8 -u -m evaluation.run_grounding_validation --questions evaluation/grounding_confirmation.jsonl
}
if ($LASTEXITCODE -ne 0) { throw 'Fresh confirmation failed.' }
& $python -X utf8 -m evaluation.summarize_grounding --questions evaluation/grounding_confirmation.jsonl
if ($LASTEXITCODE -ne 0) { throw 'Confirmation aggregation failed.' }
& $python -X utf8 -u -m scripts.runtime_pilot --output results/grounding-v3-runtime-pilot.json
if ($LASTEXITCODE -ne 0) { throw 'Grounding runtime pilot failed.' }
Write-Output 'Completed comparisons are in results/grounding-validation-*/summary.json. Existing answer rows are reused for matching code/config/question hashes.'
& $python -X utf8 -m evaluation.check_grounding_quality
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Coverage/known-negative checks did not pass. See results/summary/grounding_acceptance.json. Runtime success alone is not project readiness.'
    exit 2
}
Write-Output 'Coverage and known-negative checks passed. Review answer correctness against the cited sources before declaring submission readiness.'
