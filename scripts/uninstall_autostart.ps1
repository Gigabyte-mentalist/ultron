<#
.SYNOPSIS
    Removes the "Ultron" Task Scheduler task (docs/tz/09_autostart.md §3.3,
    §5's "uninstall_autostart.ps1 hammasini tozalaydi").

.DESCRIPTION
    Does NOT require admin rights. Safe to re-run: if the task doesn't
    exist, this just reports that and exits successfully rather than
    erroring.

    Only removes the Task Scheduler entry -- it does not kill an already
    running Ultron/supervisor process (stop it via the tray icon's "Chiqish"
    first, or Task Manager) and does not touch Ollama's own separate Startup
    entry (§3.3: "Ollama o'zining avtostart yozuvini saqlaydi. Supervisor
    faqat zaxira" -- that's Ollama's own installer's concern, not this
    project's).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\uninstall_autostart.ps1
#>

$ErrorActionPreference = "Stop"

$TaskName = "Ultron"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $existing) {
    Write-Host "'$TaskName' vazifasi topilmadi -- o'rnatilmagan yoki allaqachon o'chirilgan."
    exit 0
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false

Write-Host "OK: '$TaskName' vazifasi o'chirildi."
