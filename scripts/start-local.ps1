param(
    [int]$GuestJobsPerHour = 100,
    [int]$GlobalJobsPerDay = 500,
    [int]$GuestQuestionsPerHour = 100,
    [int]$GlobalQuestionsPerDay = 500
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $repoRoot '.local/app-v1'
$runDir = Join-Path $dataDir 'local-testing'
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
$frontendDir = Join-Path $repoRoot 'frontend'
$databasePath = Join-Path $dataDir 'jobs.sqlite3'
$databaseUrl = "sqlite:///$($databasePath.Replace('\', '/'))"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$quotaSettings = @{
    MODEL_ANATOMY_GUEST_JOBS_PER_HOUR = $GuestJobsPerHour
    MODEL_ANATOMY_GLOBAL_JOBS_PER_DAY = $GlobalJobsPerDay
    MODEL_ANATOMY_GUEST_QUESTIONS_PER_HOUR = $GuestQuestionsPerHour
    MODEL_ANATOMY_GLOBAL_QUESTIONS_PER_DAY = $GlobalQuestionsPerDay
}
foreach ($name in $quotaSettings.Keys) {
    if ($quotaSettings[$name] -lt 1 -or $quotaSettings[$name] -gt 1000) {
        throw "$name must be between 1 and 1000 for local testing."
    }
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository virtualenv Python was not found: $python"
}
$dependencyErrorLog = Join-Path $runDir 'dependency-preflight-error.log'
& $python -c 'import fastapi, pydantic_settings, sqlalchemy, uvicorn, requests, torch, transformers, safetensors' 2> $dependencyErrorLog
if ($LASTEXITCODE -ne 0) {
    throw "API or engine dependencies are missing from $python. See $dependencyErrorLog and install the repository API/engine environment before starting."
}
$viteCli = Join-Path $frontendDir 'node_modules/vite/bin/vite.js'
$node = Get-Command node.exe -ErrorAction SilentlyContinue
if (-not $node -or -not (Test-Path -LiteralPath $viteCli -PathType Leaf)) {
    throw 'Frontend dependencies are unavailable. Install Node.js and run npm ci in frontend.'
}
if (Get-NetTCPConnection -State Listen -LocalPort 8001 -ErrorAction SilentlyContinue) {
    throw 'Port 8001 already has a listener. Stop the existing API, then run this script to apply the local profile.'
}
if (Get-NetTCPConnection -State Listen -LocalPort 5173 -ErrorAction SilentlyContinue) {
    throw 'Port 5173 already has a listener. Stop the existing frontend, then run this script to start all three local services.'
}

$settings = @{
    MODEL_ANATOMY_ENVIRONMENT = 'local'
    MODEL_ANATOMY_PORT = '8001'
    MODEL_ANATOMY_DATA_DIR = $dataDir
    MODEL_ANATOMY_DATABASE_URL = $databaseUrl
    DATABASE_URL = $databaseUrl
    MODEL_ANATOMY_STORAGE_BACKEND = 'local'
    MODEL_ANATOMY_ALLOWED_HOSTS = '["127.0.0.1","localhost","[::1]"]'
    MODEL_ANATOMY_CORS_ORIGINS = '["http://localhost:5173","http://127.0.0.1:5173","http://localhost:5174","http://127.0.0.1:5174"]'
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
try {
    $preflightErrorLog = Join-Path $runDir 'worker-preflight-error.log'
    $workerState = & $python -c 'from model_anatomy.api.settings import Settings; from model_anatomy.service.jobs import JobService; s=Settings(); j=JobService(s); print(int(j.worker_available())); j.db.engine.dispose()' 2>> $preflightErrorLog
    if ($LASTEXITCODE -eq 0 -and $workerState -match '^\s*1\s*$') {
        throw 'A worker already owns the configured local database. Stop the identified old worker before starting this profile; existing workers are never reused.'
    }

    $processes.Api = Start-Process -FilePath $python -ArgumentList @('-m', 'uvicorn', 'model_anatomy.api.app:create_app', '--factory', '--host', '127.0.0.1', '--port', '8001', '--no-access-log') -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDir 'api.log') -RedirectStandardError (Join-Path $runDir 'api-error.log')
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
            $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8001/health' -TimeoutSec 2
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

    $processes.Frontend = Start-Process -FilePath $node.Source -ArgumentList @("`"$viteCli`"", '--host', '127.0.0.1', '--port', '5173', '--strictPort') -WorkingDirectory $frontendDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDir 'frontend.log') -RedirectStandardError (Join-Path $runDir 'frontend-error.log')
    $frontendDeadline = [DateTime]::UtcNow.AddSeconds(30)
    $frontendReady = $false
    while ([DateTime]::UtcNow -lt $frontendDeadline) {
        $processes.Frontend.Refresh()
        if ($processes.Frontend.HasExited) {
            throw "Frontend process exited during startup (PID $($processes.Frontend.Id)). See logs in $runDir."
        }
        try {
            $null = Invoke-WebRequest -Uri 'http://127.0.0.1:5173/' -UseBasicParsing -TimeoutSec 2
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
    foreach ($name in $oldEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], 'Process')
    }
}

Write-Output 'Local app: http://127.0.0.1:5173/'
Write-Output 'API docs:  http://127.0.0.1:8001/docs'
Write-Output "Process IDs: $(($processes.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value.Id)" }) -join ', ')"
Write-Output "Logs:      $runDir"
