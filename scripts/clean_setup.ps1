param([switch]$IncludeEvaluation,[switch]$IncludeExtraction)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8='1'
$env:UV_PYTHON_INSTALL_DIR=Join-Path $projectRoot '.runtime/python'
$env:UV_CACHE_DIR=Join-Path $projectRoot '.cache/uv'
$env:UV_CONCURRENT_DOWNLOADS='2'
$env:UV_SYSTEM_CERTS='true'
$uvPath=Join-Path $projectRoot '.tools/bin/uv.exe'
if(-not(Test-Path -LiteralPath $uvPath)){
    $installedUv=Get-Command uv.exe -ErrorAction SilentlyContinue
    if($installedUv){$uvPath=$installedUv.Source}
    else{
        python -m pip install uv --target .tools --cache-dir .cache/pip
        if($LASTEXITCODE -ne 0){throw 'Could not install project-local uv'}
        $uvPath=@('.tools/bin/uv.exe','.tools/Scripts/uv.exe','.tools/uv/uv.exe') |
            Where-Object {Test-Path -LiteralPath $_} | Select-Object -First 1
        if(-not $uvPath){throw 'uv executable was not found after installation'}
        $uvPath=(Resolve-Path -LiteralPath $uvPath).Path
    }
}
& $uvPath python install 3.11 --no-bin
if($LASTEXITCODE -ne 0){throw 'Python 3.11 installation failed'}
if(-not(Test-Path -LiteralPath '.venv311/Scripts/python.exe')){
    & $uvPath venv --python 3.11 --relocatable .venv311
    if($LASTEXITCODE -ne 0){throw 'Application environment creation failed'}
}
& $uvPath pip install --python .venv311/Scripts/python.exe --require-hashes -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
if($LASTEXITCODE -ne 0){throw 'Application dependency installation failed'}
if($IncludeExtraction){
    & $uvPath pip install --python .venv311/Scripts/python.exe -r requirements-extraction.txt --constraint requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    if($LASTEXITCODE -ne 0){throw 'Extraction-comparison dependency installation failed'}
}
if($IncludeEvaluation){
    if(-not(Test-Path -LiteralPath '.venv-eval311/Scripts/python.exe')){
        & $uvPath venv --python 3.11 --relocatable .venv-eval311
        if($LASTEXITCODE -ne 0){throw 'Ragas environment creation failed'}
    }
    & $uvPath pip install --python .venv-eval311/Scripts/python.exe -r requirements-eval.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    if($LASTEXITCODE -ne 0){throw 'Ragas dependency installation failed'}
    & $uvPath pip check --python .venv-eval311/Scripts/python.exe
    if($LASTEXITCODE -ne 0){throw 'Ragas dependency consistency check failed'}
}
& $uvPath pip check --python .venv311/Scripts/python.exe
if($LASTEXITCODE -ne 0){throw 'Application dependency consistency check failed'}
Write-Output 'Project environments are ready.'
