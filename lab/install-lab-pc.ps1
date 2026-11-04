# Sets up one Windows lab PC to run gpu-relay jobs automatically in the background.
#
# Run in an Administrator PowerShell, from the folder that holds this script and the gpu_relay .whl file:
#   powershell -ExecutionPolicy Bypass -File .\install-lab-pc.ps1 -Token <your lab token> -AuthKey <Tailscale auth key>
#
# What it does (and nothing else):
#   1. Installs Python 3.12 for all users (if missing)
#   2. Installs Tailscale and joins your private network (if -AuthKey is given)
#   3. Installs gpu-relay, PyTorch and common libraries
#   4. Creates C:\gpu-relay (jobs, log, token); students cannot read the token
#   5. Opens port 8765 in the firewall, only for Tailscale addresses (100.64.0.0/10)
#   6. Adds a scheduled task "gpu-relay" that starts at boot as the low-privilege LOCAL SERVICE account
#      (not an administrator), at below-normal priority, with no window
# Undo everything with uninstall-lab-pc.ps1.

param(
    [Parameter(Mandatory = $true)] [string] $Token,
    [string] $AuthKey = "",
    [string] $Name = $env:COMPUTERNAME.ToLower(),
    [string] $Packages = "numpy pandas scikit-learn matplotlib tqdm",
    [string] $Dir = "C:\gpu-relay",
    [switch] $AllowLan  # also accept jobs from the same local network, without Tailscale
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # makes downloads much faster in Windows PowerShell

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this from an Administrator PowerShell (right-click PowerShell > Run as administrator)."
}

function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }

$py = "C:\Program Files\Python312\python.exe"
$ts = "C:\Program Files\Tailscale\tailscale.exe"
$tmp = Join-Path $env:TEMP "gpu-relay-setup"
New-Item -ItemType Directory -Force $tmp | Out-Null

Write-Host "1/6 Python 3.12"
if (-not (Test-Path $py)) {
    $exe = Join-Path $tmp "python-3.12.10-amd64.exe"
    Invoke-WebRequest "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" -OutFile $exe -UseBasicParsing
    Start-Process $exe -Wait -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0"
    if (-not (Test-Path $py)) { throw "Python did not install to $py" }
}

Write-Host "2/6 Tailscale"
if (-not (Test-Path $ts)) {
    $msi = Join-Path $tmp "tailscale-setup-latest-amd64.msi"
    Invoke-WebRequest "https://pkgs.tailscale.com/stable/tailscale-setup-latest-amd64.msi" -OutFile $msi -UseBasicParsing
    Start-Process msiexec.exe -Wait -ArgumentList "/i `"$msi`" /quiet /norestart"
    Start-Sleep -Seconds 5
}
if ($AuthKey) {
    # --unattended keeps Tailscale connected when nobody is logged in to Windows.
    & $ts up --unattended --authkey $AuthKey --hostname $Name
    Check "tailscale up"
} else {
    Write-Host "   No -AuthKey given; skipping Tailscale sign-in." -ForegroundColor Yellow
}

Write-Host "3/6 gpu-relay, PyTorch and libraries (this takes a few minutes)"
$whl = Get-ChildItem -Path $PSScriptRoot -Filter "gpu_relay-*.whl" | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $whl) { throw "Put the gpu_relay-*.whl file next to this script." }
& $py -m pip install --upgrade pip --quiet; Check "pip upgrade"
& $py -m pip install --upgrade --force-reinstall $whl.FullName --quiet; Check "gpu-relay install"
$torchIndex = "https://download.pytorch.org/whl/cpu"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $torchIndex = "https://download.pytorch.org/whl/cu126" }
& $py -m pip install torch torchvision --index-url $torchIndex --quiet; Check "PyTorch install"
if ($Packages) { & $py -m pip install ($Packages -split "\s+") --quiet; Check "library install" }

Write-Host "4/6 Folder $Dir"
New-Item -ItemType Directory -Force (Join-Path $Dir "jobs") | Out-Null
Set-Content -Path (Join-Path $Dir "relay-token.txt") -Value $Token -NoNewline -Encoding Ascii
# Only Administrators, SYSTEM and LOCAL SERVICE (the account jobs run as) may open this folder.
icacls $Dir /inheritance:r /grant:r "*S-1-5-32-544:(OI)(CI)F" "*S-1-5-18:(OI)(CI)F" "*S-1-5-19:(OI)(CI)M" | Out-Null
Check "icacls"

Write-Host "5/6 Firewall"
Get-NetFirewallRule -DisplayName "gpu-relay" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
$remote = @("100.64.0.0/10")
if ($AllowLan) { $remote += "LocalSubnet" }
New-NetFirewallRule -DisplayName "gpu-relay" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765 `
    -Program $py -RemoteAddress $remote | Out-Null

Write-Host "6/6 Background task"
$log = Join-Path $Dir "worker.log"
$jobs = Join-Path $Dir "jobs"
$action = New-ScheduledTaskAction -Execute $py -WorkingDirectory $Dir `
    -Argument "-m gpu_relay worker --lab --name $Name --jobs `"$jobs`" --log `"$log`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$account = New-ScheduledTaskPrincipal -UserId "NT AUTHORITY\LOCAL SERVICE" -LogonType ServiceAccount
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -Priority 7
Register-ScheduledTask -TaskName "gpu-relay" -Action $action -Trigger $trigger -Principal $account `
    -Settings $settings -Force | Out-Null
Stop-ScheduledTask -TaskName "gpu-relay" -ErrorAction SilentlyContinue
Start-ScheduledTask -TaskName "gpu-relay"

Write-Host "Checking it answers..."
Start-Sleep -Seconds 8
try {
    $s = Invoke-WebRequest "http://127.0.0.1:8765/status" -Headers @{ "X-Token" = $Token } -UseBasicParsing
    Write-Host "OK: $($s.Content)" -ForegroundColor Green
} catch {
    Write-Host "The worker is not answering yet. Its log:" -ForegroundColor Yellow
    if (Test-Path $log) { Get-Content $log -Tail 20 }
}
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "Done. This PC is '$Name'. On your Mac:  gpu-relay add-many <token> $Name"
