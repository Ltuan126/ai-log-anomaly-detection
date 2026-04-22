param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 5000
)

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $repoRoot

try {
    mlflow server `
      --backend-store-uri sqlite:///mlflow.db `
      --default-artifact-root ./mlartifacts `
      --host $HostAddress `
      --port $Port
}
finally {
    Pop-Location
}
