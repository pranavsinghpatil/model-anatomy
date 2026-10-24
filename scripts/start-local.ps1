param(
    [switch]$Supabase,
    [string]$EnvFile,
    [ValidateRange(1024, 65535)]
    [int]$ApiPort = 8000,
    [int]$FrontendPort = 5173,
    [string]$PythonPath,
    [Nullable[int]]$GuestJobsPerHour,
    [Nullable[int]]$GlobalJobsPerDay,
    [Nullable[int]]$GuestQuestionsPerHour,
    [Nullable[int]]$GlobalQuestionsPerDay
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $repoRoot '.local/app-v1'
$profileName = if ($Supabase) { 'supabase' } else { 'local-testing' }
$runDir = Join-Path $dataDir $profileName
$python = if ($PythonPath) { $PythonPath } else { Join-Path $repoRoot '.venv/Scripts/python.exe' }
$frontendDir = Join-Path $repoRoot 'frontend'
$databasePath = Join-Path $dataDir 'jobs.sqlite3'
$databaseUrl = "sqlite:///$($databasePath.Replace('\', '/'))"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

foreach ($port in @($ApiPort, $FrontendPort)) {
    if ($port -lt 1024 -or $port -gt 65535) { throw 'API and frontend ports must be between 1024 and 65535.' }
}
if ($ApiPort -eq $FrontendPort) { throw 'API and frontend ports must be different.' }

$quotaDefaults = if ($Supabase) { @(2, 20, 10, 100) } else { @(100, 500, 100, 500) }
$quotaArgs = @($GuestJobsPerHour, $GlobalJobsPerDay, $GuestQuestionsPerHour, $GlobalQuestionsPerDay)
for ($i = 0; $i -lt $quotaArgs.Count; $i++) {
    if ($null -eq $quotaArgs[$i]) { $quotaArgs[$i] = $quotaDefaults[$i] }
}

$quotaSettings = @{
    MODEL_ANATOMY_GUEST_JOBS_PER_HOUR = $quotaArgs[0]
    MODEL_ANATOMY_GLOBAL_JOBS_PER_DAY = $quotaArgs[1]
    MODEL_ANATOMY_GUEST_QUESTIONS_PER_HOUR = $quotaArgs[2]
    MODEL_ANATOMY_GLOBAL_QUESTIONS_PER_DAY = $quotaArgs[3]
}
foreach ($name in $quotaSettings.Keys) {
    if ($quotaSettings[$name] -lt 1 -or $quotaSettings[$name] -gt 1000) {
        throw "$name must be between 1 and 1000."
    }
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository virtualenv Python was not found: $python"
}
$dependencyErrorLog = Join-Path $runDir 'dependency-preflight-error.log'
& $python -c 'import fastapi, pydantic_settings, sqlalchemy, uvicorn, requests, torch, transformers, safetensors, gguf' 2> $dependencyErrorLog
if ($LASTEXITCODE -ne 0) {
    throw "API or engine dependencies are missing from $python. See $dependencyErrorLog and install the repository API/engine environment before starting."
}
$viteCli = Join-Path $frontendDir 'node_modules/vite/bin/vite.js'
$node = Get-Command node.exe -ErrorAction SilentlyContinue
if (-not $node -or -not (Test-Path -LiteralPath $viteCli -PathType Leaf)) {
    throw 'Frontend dependencies are unavailable. Install Node.js and run npm ci in frontend.'
}
if (Get-NetTCPConnection -State Listen -LocalPort $ApiPort -ErrorAction SilentlyContinue) {
    throw "Port $ApiPort already has a listener. Choose another API port."
}
if (Get-NetTCPConnection -State Listen -LocalPort $FrontendPort -ErrorAction SilentlyContinue) {
    throw "Port $FrontendPort already has a listener. Choose another frontend port."
}

$settings = @{
    MODEL_ANATOMY_ENVIRONMENT = 'local'
    MODEL_ANATOMY_PORT = "$ApiPort"
    MODEL_ANATOMY_DATA_DIR = $dataDir
    MODEL_ANATOMY_STORAGE_BACKEND = 'local'
    MODEL_ANATOMY_ALLOWED_HOSTS = '["127.0.0.1","localhost","[::1]"]'
    MODEL_ANATOMY_CORS_ORIGINS = '["http://localhost:5173","http://127.0.0.1:5173","http://localhost:5174","http://127.0.0.1:5174"]'
}
if ($Supabase) {
    $settings.MODEL_ANATOMY_STORAGE_BACKEND = 'supabase'
    $settings.MODEL_ANATOMY_CORS_ORIGINS = "[`"http://localhost:$FrontendPort`",`"http://127.0.0.1:$FrontendPort`",`"http://localhost:5173`",`"http://127.0.0.1:5173`",`"http://localhost:5174`",`"http://127.0.0.1:5174`"]"
    if (-not $EnvFile) { $EnvFile = Join-Path $repoRoot 'infra/.env' }
    if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) { throw 'Supabase profile environment file was not found.' }
    $profileJson = & $python -c 'import json,sys; sys.path.insert(0, sys.argv[1]); from scripts.supabase_profile import load_supabase_profile; print(json.dumps(load_supabase_profile(sys.argv[2])))' $repoRoot $EnvFile 2> (Join-Path $runDir 'profile-error.log')
    if ($LASTEXITCODE -ne 0) { throw 'Supabase profile configuration is invalid. See the redacted profile error log.' }
    $profileSettings = $profileJson | ConvertFrom-Json
    foreach ($property in $profileSettings.PSObject.Properties) {
        $settings[$property.Name] = [string]$property.Value
    }
} else {
    $settings.MODEL_ANATOMY_DATABASE_URL = $databaseUrl
    $settings.DATABASE_URL = $databaseUrl
}
foreach ($name in $quotaSettings.Keys) {
    $settings[$name] = "$($quotaSettings[$name])"
}

$oldEnvironment = @{}
foreach ($name in $settings.Keys) {
    $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $settings[$name], 'Process')
}

$processes = @{}
$oldPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
$savedFrontendEnv = @{}
$oldProxy = [Environment]::GetEnvironmentVariable('API_PROXY_TARGET', 'Process')
if ($PythonPath) { [Environment]::SetEnvironmentVariable('PYTHONPATH', (Join-Path $repoRoot 'backend'), 'Process') }
try {
    $preflightErrorLog = Join-Path $runDir 'worker-preflight-error.log'
    $workerState = & $python -c 'from model_anatomy.api.settings import Settings; from model_anatomy.service.jobs import JobService; s=Settings(); j=JobService(s); print(int(j.worker_available())); j.db.engine.dispose()' 2>> $preflightErrorLog
    if ($Supabase -and $LASTEXITCODE -ne 0) {
        throw 'Supabase worker preflight failed. No services were started; check the private configuration and migration head.'
    }
    if ($LASTEXITCODE -eq 0 -and $workerState -match '^\s*1\s*$') {
        throw 'A worker already owns the configured local database. Stop the identified old worker before starting this profile; existing workers are never reused.'
    }

    $processes.Api = Start-Process -FilePath $python -ArgumentList @('-m', 'uvicorn', 'model_anatomy.api.app:create_app', '--factory', '--host', '127.0.0.1', '--port', "$ApiPort", '--no-access-log') -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDir 'api.log') -RedirectStandardError (Join-Path $runDir 'api-error.log')
    $processes.Worker = Start-Process -FilePath $python -ArgumentList @('-m', 'model_anatomy.service.worker') -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDir 'worker.log') -RedirectStandardError (Join-Path $runDir 'worker-error.log')

    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    $ready = $false
    while ([DateTime]::UtcNow -lt $deadline) {
        foreach ($name in @('Api', 'Worker')) {
            $processes[$name].Refresh()
            if ($processes[$name].HasExited) {
                throw "$name process exited during startup (PID $($processes[$name].Id)). See logs in $runDir."
            }
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2
            if ($health.analysis_available -and $health.checks.database -and $health.checks.storage -and $health.checks.worker) {
                $ready = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
            continue
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        throw "API and worker did not become ready within 30 seconds. See logs in $runDir."
    }

    $oldProxy = [Environment]::GetEnvironmentVariable('API_PROXY_TARGET', 'Process')
    $savedFrontendEnv = @{}
    $frontendSecretNames = @($settings.Keys) + @('DATABASE_URL', 'MODEL_ANATOMY_SUPABASE_SERVICE_ROLE_KEY', 'NEBIUS_API_KEY', 'MODEL_ANATOMY_NEBIUS_API_KEY', 'HF_TOKEN')
    foreach ($name in $frontendSecretNames | Select-Object -Unique) {
        $savedFrontendEnv[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
    [Environment]::SetEnvironmentVariable('API_PROXY_TARGET', "http://127.0.0.1:$ApiPort", 'Process')
    $processes.Frontend = Start-Process -FilePath $node.Source -ArgumentList @("`"$viteCli`"", '--host', '127.0.0.1', '--port', "$FrontendPort", '--strictPort') -WorkingDirectory $frontendDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDir 'frontend.log') -RedirectStandardError (Join-Path $runDir 'frontend-error.log')
    [Environment]::SetEnvironmentVariable('API_PROXY_TARGET', $oldProxy, 'Process')
    foreach ($name in $savedFrontendEnv.Keys) { [Environment]::SetEnvironmentVariable($name, $savedFrontendEnv[$name], 'Process') }
    $frontendDeadline = [DateTime]::UtcNow.AddSeconds(30)
    $frontendReady = $false
    while ([DateTime]::UtcNow -lt $frontendDeadline) {
        $processes.Frontend.Refresh()
        if ($processes.Frontend.HasExited) {
            throw "Frontend process exited during startup (PID $($processes.Frontend.Id)). See logs in $runDir."
        }
        try {
            $null = Invoke-WebRequest -Uri "http://127.0.0.1:$FrontendPort/" -UseBasicParsing -TimeoutSec 2
            $frontendReady = $true
            break
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $frontendReady) {
        throw "Frontend did not become ready within 30 seconds. See logs in $runDir."
    }
}
catch {
    $ownedIds = @($processes.Values | ForEach-Object { $_.Id }) -join ', '
    foreach ($process in $processes.Values) {
        $process.Refresh()
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Error "$($_.Exception.Message) Stopped only this launch's processes (PIDs: $ownedIds). Logs: $runDir"
    throw
}
finally {
    [Environment]::SetEnvironmentVariable('PYTHONPATH', $oldPythonPath, 'Process')
    [Environment]::SetEnvironmentVariable('API_PROXY_TARGET', $oldProxy, 'Process')
    foreach ($name in $savedFrontendEnv.Keys) { [Environment]::SetEnvironmentVariable($name, $savedFrontendEnv[$name], 'Process') }
    foreach ($name in $oldEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], 'Process')
    }
}

Write-Output "Local app: http://127.0.0.1:$FrontendPort/"
Write-Output "API docs:  http://127.0.0.1:$ApiPort/docs"
Write-Output "Process IDs: $(($processes.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value.Id)" }) -join ', ')"
Write-Output "Logs:      $runDir"
