param([int]$Cases = 10, [int]$JudgeSeconds = 1800, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if ($Cases -lt 1 -or $JudgeSeconds -lt 1) { throw 'Cases and JudgeSeconds must be positive.' }
$python = Join-Path (Get-Location) '.venv311/Scripts/python.exe'
$judge = Join-Path (Get-Location) '.venv-eval311/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:RAGAS_DO_NOT_TRACK = 'true'
$complete = @(Get-ChildItem results -Directory -Filter 'answers-test-*' | Where-Object {
    $manifest = Join-Path $_.FullName 'manifest.json'
    (Test-Path -LiteralPath $manifest) -and ((Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).status -eq 'complete')
} | Sort-Object LastWriteTime -Descending)
if (-not $complete.Count) { throw 'No completed held-out answer run found. Use continue_project.cmd first.' }
$answers = Join-Path $complete[0].FullName 'answers.jsonl'
foreach ($file in @($python,$judge,$answers,'evaluation/local_judge.py')) {
    if (-not (Test-Path -LiteralPath $file)) { throw "Required file missing: $file" }
}
if ($CheckOnly) { Write-Output "Ready to evaluate saved answers: $answers"; return }
$workers = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.ExecutablePath -like ((Get-Location).Path+'\*') -and $_.CommandLine -match '-m\s+evaluation\.(run_ragas|run_answers|grounding_worker|run_grounding_validation)'
})
if ($workers.Count) { throw 'A project inference/evaluation worker is already running. Wait for it before starting another.' }
& (Join-Path $PSScriptRoot 'start_ollama.ps1')
& $judge -X utf8 -m unittest evaluation.test_local_judge -v
if ($LASTEXITCODE -ne 0) { throw 'Local judge transport tests failed.' }
& $judge -X utf8 -u -m evaluation.run_ragas --answers $answers --limit 1 --max-seconds 480 --output results/ragas-native-allmetrics-pilot
if ($LASTEXITCODE -ne 0) { throw 'RAGAS compatibility pilot failed to run.' }
$pilot = Get-Content results/ragas-native-allmetrics-pilot/manifest.json -Raw | ConvertFrom-Json
if ($pilot.completed_metric_calls -ne 4) { throw 'Some pilot metrics failed; inspect results/ragas-native-allmetrics-pilot. Existing primary results are preserved.' }
& $judge -X utf8 -u -m evaluation.run_ragas --answers $answers --limit $Cases --max-seconds $JudgeSeconds
if ($LASTEXITCODE -ne 0) { throw 'RAGAS failed; inspect its saved failure records.' }
& $python -X utf8 -m evaluation.summarize
if ($LASTEXITCODE -ne 0) { throw 'Primary metric aggregation failed.' }
Write-Output 'Existing answers, retrieval runs and baselines were reused. RAGAS scores are supplementary; check actual completed/failed counts and time-budget status in its manifest.'
Write-Output 'Next, review factual correctness and cited-claim support:'
Write-Output ('.\.venv311\Scripts\python.exe -X utf8 -m evaluation.review_answers --answers "' + $answers + '" --mode hybrid_rerank --reviewer "team"')
