param([switch]$IncludeEvaluation)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.runtime/python'
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache/uv'
$env:UV_CONCURRENT_DOWNLOADS = '2'
$env:UV_SYSTEM_CERTS = 'true'
$uvPath = Join-Path $projectRoot '.tools/bin/uv.exe'
if (-not (Test-Path -LiteralPath $uvPath)) {
    python -m pip install uv --target .tools --cache-dir .cache/pip
    if ($LASTEXITCODE -ne 0) { throw 'Could not install project-local uv' }
}
& $uvPath python install 3.11 --no-bin
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 download failed' }
if (-not (Test-Path -LiteralPath '.venv311/Scripts/python.exe')) {
    & $uvPath venv --python 3.11 .venv311
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
& $uvPath pip install --python .venv311/Scripts/python.exe --require-hashes -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed' }
if ($IncludeEvaluation) {
    & $uvPath pip install --python .venv311/Scripts/python.exe -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation dependency installation failed' }
}
& $uvPath pip check --python .venv311/Scripts/python.exe
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
Write-Output 'Environment ready. See README for local model/runtime acquisition and start commands.'
