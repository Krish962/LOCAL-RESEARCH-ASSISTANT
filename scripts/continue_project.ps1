param(
    [switch]$CheckOnly,
    [switch]$SkipRagas,
    [switch]$SkipNoRag,
    [switch]$SkipCleanEnvironment,
    [int]$JudgeSeconds = 1800,
    [string]$AnswerRun = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv311/Scripts/python.exe'
$judgePython = Join-Path $projectRoot '.venv-eval311/Scripts/python.exe'
$uv = Join-Path $projectRoot '.tools/bin/uv.exe'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:RAGAS_DO_NOT_TRACK = 'true'
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache/uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.runtime/python'
$env:UV_SYSTEM_CERTS = 'true'
$env:UV_CONCURRENT_DOWNLOADS = '2'
$env:UV_HTTP_TIMEOUT = '120'
$required = @($python,$uv,'evaluation/run_answers.py','evaluation/run_retrieval.py',
    'evaluation/run_filter_ablation.py','evaluation/run_norag.py','evaluation/run_ragas.py',
    'evaluation/summarize.py','evaluation/questions.jsonl','evaluation/qrels.jsonl',
    'scripts/verify_preserved_data.py','requirements.txt','data/manifests/course_rag_v1.json')
if (-not $SkipRagas) { $required += $judgePython }
foreach ($path in $required) { if (-not (Test-Path -LiteralPath $path)) { throw "Required file missing: $path" } }
if ($CheckOnly) {
    Write-Output 'All command-line entry points, environment executables, labels and active corpus files are present.'
    Write-Output 'This check does not claim experiments or source review are complete.'
    return
}
function Run-Python {
    param([string[]]$Arguments)
    & $python -X utf8 -u @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python step failed: $($Arguments -join ' ')" }
}
Write-Output 'Reports and presentations are excluded. All inference/evaluation stays local.'
& (Join-Path $PSScriptRoot 'start_ollama.ps1')

# Never duplicate a surviving benchmark process from an interrupted chat.
$workers = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.ExecutablePath -eq $python -and $_.CommandLine -match '-m\s+evaluation\.run_answers' -and $_.CommandLine -match '--split\s+test'
})
foreach ($worker in $workers) {
    Write-Output "Existing held-out benchmark PID $($worker.ProcessId) is running; waiting instead of restarting it."
    $lastLine = ''
    while (Get-Process -Id $worker.ProcessId -ErrorAction SilentlyContinue) {
        if (Test-Path -LiteralPath '.runtime/heldout-answers.log') {
            $line = Get-Content -LiteralPath '.runtime/heldout-answers.log' -Tail 1
            if ($line -ne $lastLine) { Write-Output $line; $lastLine = $line }
        }
        Start-Sleep -Seconds 10
    }
}

Write-Output '[1/8] Tests and dependency consistency'
Run-Python -Arguments @('-m','pytest','-q','--tb=short')
& $uv pip check --python $python
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency check failed' }
Run-Python -Arguments @('-m','scripts.verify_preserved_data')

Write-Output '[2/8] Resume the held-out answer comparison'
if ($AnswerRun) {
    Run-Python -Arguments @('-m','evaluation.run_answers','--split','test','--resume',$AnswerRun)
} else {
    Run-Python -Arguments @('-m','evaluation.run_answers','--split','test')
    # The Python runner chooses its versioned path from code/config/data hashes.
    # Never automatically resume an older backend just because it was run last.
    $manifest = Get-ChildItem -Path 'results/answers-test-*/manifest.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $AnswerRun = Split-Path -Parent $manifest.FullName
}

Write-Output '[3/8] Final stage-by-stage retrieval metrics and metadata-filter ablation'
Run-Python -Arguments @('-m','evaluation.run_retrieval','--output','results/retrieval-final')
Run-Python -Arguments @('-m','evaluation.run_filter_ablation')

Write-Output '[4/8] No-passage baseline'
if (-not $SkipNoRag) { Run-Python -Arguments @('-m','evaluation.run_norag') }

Write-Output '[5/8] Clean-environment verification'
if (-not $SkipCleanEnvironment) {
    $cleanPython = Join-Path $projectRoot '.venv-verify311/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $cleanPython)) {
        & $uv venv --python $python .venv-verify311
        if ($LASTEXITCODE -ne 0) { throw 'Clean environment creation failed' }
    }
    & $uv pip install --offline --python $cleanPython --require-hashes -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) {
        Write-Output 'Some locked packages are not cached; fetching the hash-verified packages for the clean environment.'
        & $uv pip install --python $cleanPython --require-hashes -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
        if ($LASTEXITCODE -ne 0) { throw 'Clean lock installation failed; inspect the dependency download error.' }
    }
    & $uv pip check --python $cleanPython
    if ($LASTEXITCODE -ne 0) { throw 'Clean dependency check failed' }
    & $cleanPython -X utf8 -u -m scripts.runtime_pilot --output results/clean-environment-pilot.json
    if ($LASTEXITCODE -ne 0) { throw 'Real clean-environment model smoke test failed' }
}

Write-Output '[6/8] Local supplementary RAGAS (bounded; no paid API)'
if (-not $SkipRagas) {
    & $uv pip check --python $judgePython
    if ($LASTEXITCODE -ne 0) { throw 'Evaluator dependency check failed' }
    & $judgePython -X utf8 -u -m evaluation.run_ragas --answers (Join-Path $AnswerRun 'answers.jsonl') --limit 1 --metrics faithfulness context_recall --max-seconds 300 --output results/ragas-native-adapter-pilot
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath 'results/ragas-native-adapter-pilot/manifest.json')) {
        $pilot = Get-Content -LiteralPath 'results/ragas-native-adapter-pilot/manifest.json' -Raw | ConvertFrom-Json
        if ($pilot.completed_metric_calls -eq 2) {
            & $judgePython -X utf8 -u -m evaluation.run_ragas --answers (Join-Path $AnswerRun 'answers.jsonl') --limit 10 --max-seconds $JudgeSeconds
            if ($LASTEXITCODE -ne 0) { Write-Warning 'RAGAS failed. Preserve failure records; primary metrics remain available.' }
        } else { Write-Warning 'Local judge adapter pilot did not complete both metrics. Supplementary scores remain unavailable; inspect its failure records.' }
    } else { Write-Warning 'RAGAS could not initialize; inspect evaluator output. No scores were invented.' }
}

Write-Output '[7/8] Aggregate actual results'
Run-Python -Arguments @('-m','evaluation.summarize')

Write-Output '[8/8] Ready for source review'
Write-Output "Saved answers: $AnswerRun"
Write-Output 'Metrics: results/summary/metrics.csv and metrics.json'
Write-Output 'Factual correctness and exact citation support are intentionally unscored until reviewed.'
Write-Output ('.\.venv311\Scripts\python.exe -X utf8 -m evaluation.review_answers --answers "' + (Join-Path $AnswerRun 'answers.jsonl') + '" --mode hybrid_rerank --reviewer "team"')
Write-Output 'After review, rerun: .\.venv311\Scripts\python.exe -m evaluation.summarize'
