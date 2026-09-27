<#
.SYNOPSIS
    Registers the "Ultron" Task Scheduler task so the assistant starts
    automatically at logon (docs/tz/09_autostart.md §3.3).

.DESCRIPTION
    Idempotent: re-running this updates the existing task in place
    (Register-ScheduledTask -Force).

    Does NOT require admin rights: run level is "Limited" (§3.1 -- Ultron
    needs the interactive user session for the microphone, tray icon, and
    launching apps/the browser, which a Session-0 Windows *Service* cannot
    do, so this is a logon task, not a service).

    Two triggers are registered:
      - At log on (current user)
      - On workstation unlock (a plain re-arrival at the desktop after
        locking -- the supervisor's own single-instance mutex means this
        never launches a duplicate if the first instance is still running;
        it only matters if that instance had died in the meantime)

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\install_autostart.ps1

    Run as the normal (non-admin) user -- no elevation needed or wanted.
#>

$ErrorActionPreference = "Stop"

$TaskName    = "Ultron"
$ProjectRoot = "C:\ultron"
$PythonwExe  = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"

if (-not (Test-Path $PythonwExe)) {
    Write-Error "Topilmadi: $PythonwExe`nAvval loyiha papkasida 'uv sync' ishga tushiring."
    exit 1
}

$Action = New-ScheduledTaskAction `
    -Execute $PythonwExe `
    -Argument "-m ultron.supervisor" `
    -WorkingDirectory $ProjectRoot

$LogonTrigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"

# Task Scheduler's "On workstation unlock" has no dedicated
# New-ScheduledTaskTrigger shortcut -- built via the CIM trigger class
# instead (StateChange 8 = TASK_SESSION_UNLOCK).
$UnlockTriggerClass = Get-CimClass -ClassName MSFT_TaskSessionStateChangeTrigger `
    -Namespace Root/Microsoft/Windows/TaskScheduler
$UnlockTrigger = New-CimInstance -CimClass $UnlockTriggerClass -ClientOnly -Property @{
    StateChange = 8  # TASK_SESSION_UNLOCK
    UserId      = "$env:USERDOMAIN\$env:USERNAME"
    Enabled     = $true
}

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable

$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

$Task = New-ScheduledTask `
    -Action $Action `
    -Trigger @($LogonTrigger, $UnlockTrigger) `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Ultron ovozli AI yordamchisini kirishda avtomatik ishga tushiradi (docs/tz/09_autostart.md)."

Register-ScheduledTask -TaskName $TaskName -InputObject $Task -Force | Out-Null

Write-Host "OK: '$TaskName' vazifasi o'rnatildi/yangilandi."
Write-Host ""
Write-Host "Tekshirish:"
Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host ""
Write-Host "Hoziroq qo'lda sinash uchun (kirishni kutmasdan):"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
