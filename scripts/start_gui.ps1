param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8050,
    [switch]$Debug,
    [switch]$NoBrowser,
    [switch]$Wait,
    [int]$TimeoutSec = 30
)

$ErrorActionPreference = "Stop"

if (-not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = (Join-Path (Get-Location) ".uv_cache")
}

function Test-TcpPort {
    param(
        [string]$TargetHost,
        [int]$TargetPort
    )
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($TargetHost, $TargetPort, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(250)) {
            return $false
        }
        $client.EndConnect($async) | Out-Null
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

$args = @("run", "veldra-gui", "--host", $BindHost, "--port", "$Port")
if ($Debug.IsPresent) {
    $args += "--debug"
}

Write-Host "Starting Veldra GUI: uv $($args -join ' ')"
$server = Start-Process -FilePath "uv" -ArgumentList $args -PassThru

$url = "http://${BindHost}:${Port}"
$deadline = (Get-Date).AddSeconds($TimeoutSec)

while ((Get-Date) -lt $deadline) {
    if ($server.HasExited) {
        throw "GUI server exited early with code $($server.ExitCode)."
    }
    if (Test-TcpPort -TargetHost $BindHost -TargetPort $Port) {
        if (-not $NoBrowser.IsPresent) {
            Start-Process $url | Out-Null
        }
        Write-Host "GUI is ready: $url"
        break
    }
    Start-Sleep -Milliseconds 300
}

if (-not (Test-TcpPort -TargetHost $BindHost -TargetPort $Port)) {
    try {
        if (-not $server.HasExited) {
            Stop-Process -Id $server.Id -Force
        }
    } catch {
        # Ignore cleanup errors.
    }
    throw "GUI did not become ready within ${TimeoutSec}s."
}

if ($Wait.IsPresent) {
    Write-Host "Press Ctrl+C to stop the GUI server."
    try {
        Wait-Process -Id $server.Id
    } finally {
        if (-not $server.HasExited) {
            Stop-Process -Id $server.Id -Force
        }
    }
} else {
    Write-Host "Server process id: $($server.Id)"
}
