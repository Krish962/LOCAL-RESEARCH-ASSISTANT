$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
try {
    $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3
    Write-Output 'Reusing the running local Ollama server.'
    return
} catch { }
$runtime = Join-Path $projectRoot '.runtime/ollama/ollama.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    $installedOllama = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if (-not $installedOllama) { throw 'Ollama is missing. Install Ollama and restart the terminal.' }
    $runtime = $installedOllama.Source
}
New-Item -ItemType Directory -Path (Join-Path $projectRoot '.runtime') -Force | Out-Null
$env:OLLAMA_MODELS = Join-Path $projectRoot '.cache/ollama_models'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = 'true'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_CONTEXT_LENGTH = '4096'
$server = Start-Process -FilePath $runtime -ArgumentList 'serve' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $projectRoot '.runtime/ollama-server.out.log') -RedirectStandardError (Join-Path $projectRoot '.runtime/ollama-server.err.log')
@{process_id=$server.Id;url='http://127.0.0.1:11434';models=$env:OLLAMA_MODELS} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot '.runtime/ollama-server.json')
for ($attempt=0; $attempt -lt 20; $attempt++) {
    try { $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2; Write-Output 'Local Ollama is ready.'; return } catch { Start-Sleep -Seconds 1 }
}
throw 'Local Ollama did not become ready; inspect .runtime/ollama-server.err.log'
