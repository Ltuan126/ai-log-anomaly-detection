param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 5000,
    [switch]$KeepServerRunning
)

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $repoRoot

try {
    Write-Host "Starting MLflow server..." -ForegroundColor Green
    $serverJob = Start-Job -ScriptBlock {
        param($root, $hostAddr, $port)
        Set-Location $root
        & .\venv\Scripts\mlflow.exe server `
          --backend-store-uri sqlite:///mlflow.db `
          --default-artifact-root ./mlartifacts `
          --host $hostAddr `
          --port $port
    } -ArgumentList $repoRoot, $HostAddress, $Port

    Write-Host "Waiting for MLflow server to be ready..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3

    Write-Host "Server is running. Starting training..." -ForegroundColor Green
    & .\venv\Scripts\python.exe src/train.py

    if ($KeepServerRunning) {
        Write-Host "`nMLflow server is still running at http://$($HostAddress):$Port" -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
        Wait-Job $serverJob
    }
    else {
        Write-Host "Stopping MLflow server..." -ForegroundColor Yellow
        Stop-Job $serverJob
        Remove-Job $serverJob
        Write-Host "Done!" -ForegroundColor Green
    }
}
finally {
    Pop-Location
}
