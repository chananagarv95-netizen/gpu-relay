# Removes gpu-relay from a lab PC: stops any running job, deletes the background task, the firewall rule
# and C:\gpu-relay. Run in an Administrator PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\uninstall-lab-pc.ps1                 (keep Python and Tailscale)
#   powershell -ExecutionPolicy Bypass -File .\uninstall-lab-pc.ps1 -RemoveTailscale
#
# To only PAUSE it instead (e.g. during exams):
#   Disable-ScheduledTask -TaskName gpu-relay; Stop-ScheduledTask -TaskName gpu-relay
# and to turn it back on:
#   Enable-ScheduledTask -TaskName gpu-relay; Start-ScheduledTask -TaskName gpu-relay

param(
    [string] $Dir = "C:\gpu-relay",
    [switch] $RemoveTailscale
)
$ErrorActionPreference = "Continue"

Write-Host "Stopping gpu-relay and any job it is running"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*gpu_relay worker*" } |
    ForEach-Object { taskkill /F /T /PID $_.ProcessId | Out-Null }
Stop-ScheduledTask -TaskName "gpu-relay" -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName "gpu-relay" -Confirm:$false -ErrorAction SilentlyContinue

Write-Host "Removing firewall rule and $Dir"
Get-NetFirewallRule -DisplayName "gpu-relay" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
Remove-Item $Dir -Recurse -Force -ErrorAction SilentlyContinue

if ($RemoveTailscale) {
    Write-Host "Signing out of and removing Tailscale"
    & "C:\Program Files\Tailscale\tailscale.exe" logout
    Get-CimInstance Win32_Product -Filter "Name LIKE 'Tailscale%'" | Invoke-CimMethod -MethodName Uninstall | Out-Null
}
Write-Host "Done. Python 3.12 is left installed (remove it from Settings > Apps if you want)."
