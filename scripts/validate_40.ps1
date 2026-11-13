param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python=Join-Path $projectRoot '.venv311/Scripts/python.exe'
$env:PYTHONUTF8='1'
$env:HF_HUB_OFFLINE='1'
foreach($file in @($python,'evaluation/final_validation_40.jsonl','evaluation/baselines/generation/inventory.json')){
    if(-not(Test-Path -LiteralPath $file)){throw "Required file missing: $file"}
}
& $python -B -X utf8 -m evaluation.validate_data --check
if($LASTEXITCODE -ne 0){throw 'Frozen dataset/source checks failed.'}
if($CheckOnly){Write-Output 'Source-data verification passed; no inference was run.';return}
$active=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.ExecutablePath -like ($projectRoot+'\*') -and $_.CommandLine -match '-m\s+evaluation\.(run_grounding_validation|grounding_worker|run_ragas)'
})
if($active.Count){throw 'A project evaluation is already running.'}
& $python -B -X utf8 -m scripts.prepare_grounding
if($LASTEXITCODE -ne 0){throw 'Grounding-model verification failed.'}
& (Join-Path $PSScriptRoot 'start_ollama.ps1')
& $python -B -X utf8 -u -m evaluation.run_grounding_validation --run-file results/summary/validation40_run.json
if($LASTEXITCODE -ne 0){throw 'Evaluation failed; completed rows remain resumable.'}
$run=(Get-Content results/summary/validation40_run.json -Raw | ConvertFrom-Json).run
& $python -B -X utf8 -m evaluation.summarize --run $run
if($LASTEXITCODE -ne 0){throw 'Metric aggregation failed.'}
