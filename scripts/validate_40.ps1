param([switch]$CheckOnly,[int]$RagasLimit=0,[int]$JudgeSeconds=1800)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python=Join-Path $projectRoot '.venv311/Scripts/python.exe'
$questions='evaluation/final_validation_40.jsonl'
$env:PYTHONUTF8='1'
$env:HF_HUB_OFFLINE='1'
$env:RAGAS_DO_NOT_TRACK='true'
if ($RagasLimit -lt 0 -or $JudgeSeconds -lt 1) { throw 'RagasLimit must be nonnegative and JudgeSeconds positive.' }
foreach ($file in @($python,$questions,'docs/baseline/pre_grounding/inventory.json')) {
    if (-not (Test-Path -LiteralPath $file)) { throw "Required file missing: $file" }
}
& $python -X utf8 -m evaluation.build_validation40 --check
if ($LASTEXITCODE -ne 0) { throw 'Frozen dataset/source checks failed.' }
if ($CheckOnly) { Write-Output '40-question validation preflight passed. No model inference was run.'; return }
$active=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.ExecutablePath -like ($projectRoot+'\*') -and $_.CommandLine -match '-m\s+evaluation\.(run_grounding_validation|grounding_worker|run_answers|run_ragas)'
})
if ($active.Count) { throw 'A project evaluation is already running. Wait for it before starting another.' }
& $python -X utf8 -m pytest -q --tb=short
if ($LASTEXITCODE -ne 0) { throw 'Code/data tests failed.' }
& $python -X utf8 -m scripts.prepare_grounding
if ($LASTEXITCODE -ne 0) { throw 'Grounding cache verification failed.' }
& (Join-Path $PSScriptRoot 'start_ollama.ps1')
Write-Output 'Evaluating the same 40 questions on the preserved baseline and current pipeline (80 case/arm pairs). Saved matching rows are reused.'
& $python -X utf8 -u -m evaluation.run_grounding_validation --questions $questions --run-file results/summary/validation40_run.json
if ($LASTEXITCODE -ne 0) { throw '40-question evaluation failed; saved rows remain resumable.' }
$run=(Get-Content results/summary/validation40_run.json -Raw | ConvertFrom-Json).run
& $python -X utf8 -m evaluation.summarize_grounding --run $run --questions $questions
if ($LASTEXITCODE -ne 0) { throw 'Metric aggregation failed.' }
& $python -X utf8 -m evaluation.check_grounding_quality --questions $questions --output results/summary/validation40_acceptance.json
$qualityCode=$LASTEXITCODE
if ($RagasLimit -gt 0) {
    $judge=Join-Path $projectRoot '.venv-eval311/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $judge)) { throw 'Optional RAGAS environment is missing.' }
    & $judge -X utf8 -u -m evaluation.run_ragas --answers (Join-Path $run 'grounded_answers.jsonl') --questions $questions --limit $RagasLimit --max-seconds $JudgeSeconds
    if ($LASTEXITCODE -ne 0) { Write-Warning 'Supplementary RAGAS failed; preserve failure records and primary metrics.' }
}
Write-Output "Comparison CSV/JSON: $run"
Write-Output 'Acceptance summary: results/summary/validation40_acceptance.json'
Write-Output 'Coverage is not answer correctness. Source-review scores remain unscored until actual judgments are supplied.'
if ($qualityCode -ne 0) { Write-Warning 'Coverage/negative-control checks require review.'; exit $qualityCode }
Write-Output 'Coverage and known-negative controls passed. Review factual correctness against the cited sources.'
