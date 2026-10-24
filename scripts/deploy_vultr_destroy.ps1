<#
.SYNOPSIS
Destroy the Vultr instance created by scripts\deploy_vultr.ps1 (stops the hourly billing).

.DESCRIPTION
Reads the instance id from %LOCALAPPDATA%\vbw\vultr-instance.json (or -InstanceId), shows what it is, and deletes it
only with -Yes. The uploaded SSH key is kept (it costs nothing); pass -DeleteSshKey to remove it too.
VULTR_API_KEY comes from the environment or the repo-root .env. Windows PowerShell 5.1 compatible.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\deploy_vultr_destroy.ps1            # show what would be deleted
powershell -ExecutionPolicy Bypass -File scripts\deploy_vultr_destroy.ps1 -Yes       # delete it
#>
[CmdletBinding()]
param(
    [string]$InstanceId = "",
    [switch]$Yes,
    [switch]$DryRun,
    [switch]$DeleteSshKey
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Root = Split-Path -Parent $PSScriptRoot
$Api = "https://api.vultr.com/v2"
$StateFile = Join-Path $env:LOCALAPPDATA "vbw\vultr-instance.json"

$apiKey = $env:VULTR_API_KEY
if (-not $apiKey -and (Test-Path (Join-Path $Root ".env"))) {
    $line = Get-Content (Join-Path $Root ".env") | Where-Object { $_ -match "^\s*VULTR_API_KEY\s*=" } | Select-Object -First 1
    if ($line) { $apiKey = $line.Split("=", 2)[1].Trim().Trim("'").Trim('"') }
}
if (-not $apiKey) { throw "VULTR_API_KEY is not set (environment or $Root\.env)" }
$headers = @{ Authorization = "Bearer $apiKey" }

if (-not $InstanceId) {
    if (-not (Test-Path $StateFile)) { throw "No saved instance ($StateFile); pass -InstanceId" }
    $InstanceId = (Get-Content $StateFile -Raw | ConvertFrom-Json).id
}

try {
    $inst = (Invoke-RestMethod -Method GET -Uri "$Api/instances/$InstanceId" -Headers $headers -TimeoutSec 60).instance
} catch {
    throw "Could not read instance ${InstanceId}: $($_.Exception.Message)"
}
Write-Host ("Instance {0}: label={1} ip={2} plan={3} region={4} status={5} created={6}" -f `
    $inst.id, $inst.label, $inst.main_ip, $inst.plan, $inst.region, $inst.status, $inst.date_created)

if ($DryRun -or -not $Yes) {
    Write-Host "[dry-run] DELETE $Api/instances/$InstanceId" -ForegroundColor DarkYellow
    if (-not $Yes) { Write-Host "Nothing deleted. Re-run with -Yes to destroy it." -ForegroundColor Yellow }
    exit 0
}

Invoke-RestMethod -Method DELETE -Uri "$Api/instances/$InstanceId" -Headers $headers -TimeoutSec 60 | Out-Null
Write-Host "Deleted instance $InstanceId." -ForegroundColor Green

if ($DeleteSshKey) {
    $keys = (Invoke-RestMethod -Method GET -Uri "$Api/ssh-keys?per_page=500" -Headers $headers -TimeoutSec 60).ssh_keys |
        Where-Object { $_.name -eq "vbw-deploy" }
    foreach ($k in $keys) {
        Invoke-RestMethod -Method DELETE -Uri "$Api/ssh-keys/$($k.id)" -Headers $headers -TimeoutSec 60 | Out-Null
        Write-Host "Deleted SSH key $($k.id)."
    }
}

if (Test-Path $StateFile) {
    $saved = Get-Content $StateFile -Raw | ConvertFrom-Json
    if ($saved.id -eq $InstanceId) { Remove-Item $StateFile -Force }
}
