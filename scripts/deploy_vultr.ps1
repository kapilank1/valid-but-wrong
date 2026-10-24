<#
.SYNOPSIS
Deploy Valid-But-Wrong (pipeline server + mission console) to one Vultr cloud instance with docker compose.

.DESCRIPTION
Windows PowerShell 5.1 compatible. Uses the Vultr API v2 (https://api.vultr.com/v2) and the OpenSSH client that ships
with Windows (ssh, scp, ssh-keygen) plus tar.exe.

  1. Reads VULTR_API_KEY, DATABASE_URL (Tiger Cloud), CARTESIA_API_KEY from the environment, else from the repo-root
     .env and apps/console/.env.local (both gitignored). Secrets are never printed and never put in user-data.
  2. Picks the cheapest plan in -Region with >= 4 GB RAM and >= 2 vCPU, and Ubuntu 24.04 x64. Prints the price.
  3. Only with -Yes: uploads your SSH public key (~/.ssh/id_ed25519.pub; created if missing), creates the instance
     with infra/cloud-init.yaml as user-data (installs Docker, swap, opens ports), waits for it to boot.
  4. Uploads the code (a tarball of this checkout, or -GitUrl to clone instead), writes infra/.env on the host
     over SSH, and runs scripts/deploy_remote.sh: docker compose up --build server + console (+ local TimescaleDB if
     no DATABASE_URL, + Caddy HTTPS if -Domain), then waits for /api/health.

State (instance id + IP, no secrets) is kept in %LOCALAPPDATA%\vbw\vultr-instance.json for -Update and for
scripts\deploy_vultr_destroy.ps1. Vultr bills hourly until the instance is destroyed.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\deploy_vultr.ps1 -DryRun
Shows the plan, price, OS and every API call body without creating anything.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\deploy_vultr.ps1 -Yes
Creates the instance and deploys.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\deploy_vultr.ps1 -Update
Re-uploads the current code and models to the existing instance and rebuilds.
#>
[CmdletBinding()]
param(
    [string]$Region = "sjc",                 # Silicon Valley, closest to Cal Hacks (SF)
    [string]$Plan = "",                      # explicit plan id, e.g. vc2-2c-4gb; default = cheapest that fits
    [int]$MinRamMb = 4096,
    [int]$MinVcpu = 2,
    [string]$Label = "vbw-demo",
    [string]$GitUrl = "",                    # clone this instead of uploading the local checkout
    [string]$Branch = "main",
    [string]$Domain = "",                    # optional: DNS A record -> instance IP, enables Caddy HTTPS
    [switch]$LocalDb,                        # force the local TimescaleDB container even if DATABASE_URL is set
    [switch]$DryRun,
    [switch]$Yes,
    [switch]$Update,
    [string]$SshKeyPath = (Join-Path $HOME ".ssh\id_ed25519")
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Root = Split-Path -Parent $PSScriptRoot
$Api = "https://api.vultr.com/v2"
$StateDir = Join-Path $env:LOCALAPPDATA "vbw"
$StateFile = Join-Path $StateDir "vultr-instance.json"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
# The SMAP/MSL scenarios in vbw/sources/registry.py (SPACECRAFT_CHANNELS); uploaded so the host needn't fetch 165 MB.
$SpacecraftChannels = @("P-1", "D-5", "E-7", "T-3", "M-7", "F-5", "P-11", "T-9")

function Write-Step([string]$Text) { Write-Host ""; Write-Host "== $Text" -ForegroundColor Cyan }

function Read-DotEnv([string]$Path) {
    $values = @{}
    if (Test-Path $Path) {
        foreach ($line in Get-Content $Path) {
            $t = $line.Trim()
            if ($t -and -not $t.StartsWith("#") -and $t.Contains("=")) {
                $i = $t.IndexOf("=")
                $values[$t.Substring(0, $i).Trim()] = $t.Substring($i + 1).Trim().Trim("'").Trim('"')
            }
        }
    }
    return $values
}

function Get-Setting([string]$Name, [hashtable[]]$Files) {
    $v = [Environment]::GetEnvironmentVariable($Name)
    if ($v) { return $v.Trim() }
    foreach ($f in $Files) { if ($f.ContainsKey($Name) -and $f[$Name]) { return $f[$Name] } }
    return ""
}

function Write-TextFile([string]$Path, [string]$Text) {
    # LF line endings and no BOM: these files are read by bash, cloud-init and docker compose on Linux.
    [IO.File]::WriteAllText($Path, ($Text -replace "`r`n", "`n"), $Utf8NoBom)
}

function Invoke-Vultr([string]$Method, [string]$Path, $Body = $null, [switch]$Public) {
    # -Public: unauthenticated catalog reads (plans, os, regions); they run even in -DryRun.
    $uri = "$Api$Path"
    $json = $null
    if ($null -ne $Body) { $json = $Body | ConvertTo-Json -Depth 6 -Compress }
    if ($DryRun -and -not $Public) {
        Write-Host "  [dry-run] $Method $uri" -ForegroundColor DarkYellow
        if ($json) { Write-Host "            body: $json" -ForegroundColor DarkYellow }
        return $null
    }
    $headers = @{}
    if (-not $Public) { $headers["Authorization"] = "Bearer $script:ApiKey" }
    $params = @{ Method = $Method; Uri = $uri; Headers = $headers; TimeoutSec = 60 }
    if ($json) { $params["Body"] = $json; $params["ContentType"] = "application/json" }
    try {
        return Invoke-RestMethod @params
    } catch {
        $detail = ""
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) { $detail = $_.ErrorDetails.Message }
        throw "Vultr API $Method $Path failed: $($_.Exception.Message) $detail"
    }
}

function Invoke-Native([string]$Exe, [string[]]$ArgList) {
    # Native tools write progress to stderr; don't let PowerShell 5.1 turn that into terminating errors.
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Exe @ArgList | Out-Host; return $LASTEXITCODE } finally { $ErrorActionPreference = $old }
}

function Save-State($State) {
    if (-not (Test-Path $StateDir)) { New-Item -ItemType Directory -Force $StateDir | Out-Null }
    Write-TextFile $StateFile ($State | ConvertTo-Json)
}

# --------------------------------------------------------------------------------------------- settings

$dotenv = Read-DotEnv (Join-Path $Root ".env")
$consoleEnv = Read-DotEnv (Join-Path $Root "apps\console\.env.local")
$script:ApiKey = Get-Setting "VULTR_API_KEY" @($dotenv)
$databaseUrl = Get-Setting "DATABASE_URL" @($dotenv)
$cartesiaKey = Get-Setting "CARTESIA_API_KEY" @($consoleEnv, $dotenv)
$cartesiaVoice = Get-Setting "CARTESIA_VOICE_ID" @($consoleEnv, $dotenv)
$cartesiaModel = Get-Setting "CARTESIA_MODEL_ID" @($consoleEnv, $dotenv)
$useLocalDb = $LocalDb -or -not $databaseUrl

Write-Step "Settings"
Write-Host ("  VULTR_API_KEY     : " + $(if ($script:ApiKey) { "set" } else { "MISSING" }))
Write-Host ("  DATABASE_URL      : " + $(if ($useLocalDb) { "not used -> local TimescaleDB container on the instance" } else { "set (Tiger Cloud)" }))
Write-Host ("  CARTESIA_API_KEY  : " + $(if ($cartesiaKey) { "set" } else { "not set -> browser speech fallback" }))
Write-Host ("  code              : " + $(if ($GitUrl) { "git clone $GitUrl ($Branch)" } else { "upload this checkout ($Root)" }))
if ($Domain) { Write-Host "  domain            : $Domain (Caddy HTTPS; point its A record at the instance IP)" }
if (-not $script:ApiKey -and -not $DryRun -and -not $Update) {
    throw "VULTR_API_KEY is not set. Put VULTR_API_KEY=... in $Root\.env (gitignored) or the environment."
}
foreach ($tool in @("ssh", "scp", "ssh-keygen", "tar")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not found (Windows OpenSSH client / tar.exe)" }
}

# --------------------------------------------------------------------------------------------- instance

$state = $null
if ($Update) {
    if (-not (Test-Path $StateFile)) { throw "No saved instance ($StateFile). Deploy first without -Update." }
    $state = Get-Content $StateFile -Raw | ConvertFrom-Json
    Write-Step "Updating existing instance $($state.id) at $($state.ip)"
} else {
    Write-Step "Choosing a plan in region '$Region'"
    $plans = (Invoke-Vultr GET "/plans?type=all&per_page=500" -Public).plans
    if ($Plan) {
        $chosen = $plans | Where-Object { $_.id -eq $Plan } | Select-Object -First 1
        if (-not $chosen) { throw "plan '$Plan' not found" }
    } else {
        $chosen = $plans |
            Where-Object { $_.ram -ge $MinRamMb -and $_.vcpu_count -ge $MinVcpu -and ($_.type -in @("vc2", "vhf", "vhp")) -and ($_.locations -contains $Region) } |
            Sort-Object { [double]$_.monthly_cost } | Select-Object -First 1
        if (-not $chosen) { throw "no plan with >= $MinRamMb MB / $MinVcpu vCPU in region '$Region'; try -Region lax or -Plan" }
    }
    $hourly = if ($chosen.PSObject.Properties.Name -contains "hourly_cost") { [double]$chosen.hourly_cost } else { [math]::Round([double]$chosen.monthly_cost / 672, 4) }
    Write-Host ("  plan  : {0}  ({1} vCPU, {2} MB RAM, {3} GB disk)" -f $chosen.id, $chosen.vcpu_count, $chosen.ram, $chosen.disk)
    Write-Host ("  price : `${0}/hour, `${1}/month max (billed hourly until destroyed)" -f $hourly, $chosen.monthly_cost) -ForegroundColor Yellow

    $os = (Invoke-Vultr GET "/os?per_page=500" -Public).os | Where-Object { $_.name -like "Ubuntu 24.04*" -and $_.arch -eq "x64" } | Select-Object -First 1
    if (-not $os) { throw "Ubuntu 24.04 x64 not found in the Vultr OS list" }
    Write-Host "  os    : $($os.name) (id $($os.id))"

    if (-not $Yes -and -not $DryRun) {
        Write-Host ""
        Write-Host "Nothing created. Re-run with -Yes to create this instance and deploy (or -DryRun to preview)." -ForegroundColor Yellow
        exit 1
    }

    Write-Step "SSH key"
    if (-not (Test-Path "$SshKeyPath.pub")) {
        if ($DryRun) {
            Write-Host "  [dry-run] would run: ssh-keygen -t ed25519 -N `"`" -f $SshKeyPath"
        } else {
            $sshDir = Split-Path -Parent $SshKeyPath
            if (-not (Test-Path $sshDir)) { New-Item -ItemType Directory -Force $sshDir | Out-Null }
            if ((Invoke-Native "ssh-keygen" @("-t", "ed25519", "-N", '""', "-C", "vbw-deploy", "-f", $SshKeyPath)) -ne 0) { throw "ssh-keygen failed" }
        }
    }
    $pubKey = if (Test-Path "$SshKeyPath.pub") { (Get-Content "$SshKeyPath.pub" -Raw).Trim() } else { "<new key>" }
    $keyId = $null
    if (-not $DryRun) {
        $existing = (Invoke-Vultr GET "/ssh-keys?per_page=500").ssh_keys | Where-Object { $_.ssh_key.Trim() -eq $pubKey } | Select-Object -First 1
        if ($existing) { $keyId = $existing.id; Write-Host "  reusing Vultr SSH key $keyId" }
    }
    if (-not $keyId) {
        $created = Invoke-Vultr POST "/ssh-keys" @{ name = "vbw-deploy"; ssh_key = $pubKey }
        if ($created) { $keyId = $created.ssh_key.id; Write-Host "  uploaded SSH key $keyId" } else { $keyId = "<ssh-key-id>" }
    }

    Write-Step "Creating the instance"
    $cloudInit = (Get-Content (Join-Path $Root "infra\cloud-init.yaml") -Raw) -replace "`r`n", "`n"
    $userData = [Convert]::ToBase64String($Utf8NoBom.GetBytes($cloudInit))
    $body = @{
        region = $Region; plan = $chosen.id; os_id = $os.id; label = $Label; hostname = $Label
        sshkey_id = @($keyId); user_data = $userData; backups = "disabled"; enable_ipv6 = $false; tags = @("vbw", "calhacks")
    }
    $result = Invoke-Vultr POST "/instances" $body
    if ($DryRun) {
        Write-Host ""
        Write-Host "Dry run: nothing was created. Then the script would wait for boot, upload the code and" -ForegroundColor Yellow
        Write-Host "infra/.env over SSH, and run scripts/deploy_remote.sh $(if ($useLocalDb) { '--local-db ' })$(if ($Domain) { '--edge' })" -ForegroundColor Yellow
        exit 0
    }
    $id = $result.instance.id
    Write-Host "  instance $id created; waiting for it to boot"
    Save-State @{ id = $id; ip = ""; label = $Label; region = $Region; plan = $chosen.id; created = (Get-Date).ToString("s") }
    $deadline = (Get-Date).AddMinutes(10)
    do {
        Start-Sleep -Seconds 10
        $inst = (Invoke-Vultr GET "/instances/$id").instance
        Write-Host ("  status={0} power={1} ip={2}" -f $inst.status, $inst.power_status, $inst.main_ip)
    } until (($inst.status -eq "active" -and $inst.power_status -eq "running" -and $inst.main_ip -ne "0.0.0.0") -or (Get-Date) -gt $deadline)
    if ($inst.status -ne "active") { throw "instance $id did not become active in 10 minutes (it is still billed; destroy it with deploy_vultr_destroy.ps1)" }
    $state = [pscustomobject]@{ id = $id; ip = $inst.main_ip; label = $Label; region = $Region; plan = $chosen.id; created = (Get-Date).ToString("s") }
    Save-State $state
}

$ip = $state.ip
$knownHosts = Join-Path $StateDir "known_hosts_$($state.id)"
$sshOpts = @("-i", $SshKeyPath, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=$knownHosts", "-o", "ConnectTimeout=10")

Write-Step "Waiting for SSH on $ip"
$deadline = (Get-Date).AddMinutes(10)
while ((Invoke-Native "ssh" ($sshOpts + @("root@$ip", "true"))) -ne 0) {
    if ((Get-Date) -gt $deadline) { throw "SSH to root@$ip did not come up in 10 minutes" }
    Start-Sleep -Seconds 10
}

# --------------------------------------------------------------------------------------------- code + env

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("vbw-deploy-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force $tmp | Out-Null
try {
    $uploads = @()
    if (-not $GitUrl) {
        Write-Step "Bundling this checkout"
        $entries = @("pyproject.toml", "uv.lock", ".python-version", "vbw", "scripts", "infra", "models", "results", "apps", "data/processed", "docs", "README.md") |
            Where-Object { Test-Path (Join-Path $Root $_) }
        foreach ($chan in $SpacecraftChannels) {
            foreach ($split in @("train", "test")) {
                $rel = "data/raw/smap_msl/$split/$chan.npy"
                if (Test-Path (Join-Path $Root $rel)) { $entries += $rel }
            }
        }
        if (Test-Path (Join-Path $Root "data/raw/smap_msl/labeled_anomalies.csv")) { $entries += "data/raw/smap_msl/labeled_anomalies.csv" }
        $bundle = Join-Path $tmp "vbw.tgz"
        $tarArgs = @("-czf", $bundle,
            "--exclude=apps/console/node_modules", "--exclude=apps/console/.next", "--exclude=apps/console/.env*",
            "--exclude=infra/.env", "--exclude=*/__pycache__", "--exclude=*.pyc",
            "-C", $Root) + $entries
        if ((Invoke-Native "tar" $tarArgs) -ne 0) { throw "tar failed" }
        Write-Host ("  {0:N1} MB" -f ((Get-Item $bundle).Length / 1MB))
        $uploads += $bundle
    }

    # infra/.env for docker compose on the host. Single quotes: compose takes them literally (no $-interpolation).
    $apiUrl = if ($Domain) { "https://$Domain" } else { "http://${ip}:8000" }
    $origins = @("http://${ip}:3000")
    if ($Domain) { $origins += "https://$Domain" }
    $envLines = @(
        "# Written by scripts/deploy_vultr.ps1 on $(Get-Date -Format s). Holds secrets; never commit.",
        "DATABASE_URL='$(if ($useLocalDb) { '' } else { $databaseUrl })'",
        "VBW_DB_AUTO_INIT=1",
        "VBW_CORS_ORIGINS=$($origins -join ',')",
        "NEXT_PUBLIC_VBW_API=$apiUrl",
        "CARTESIA_API_KEY='$cartesiaKey'"
    )
    if ($cartesiaVoice) { $envLines += "CARTESIA_VOICE_ID=$cartesiaVoice" }
    if ($cartesiaModel) { $envLines += "CARTESIA_MODEL_ID=$cartesiaModel" }
    if ($Domain) { $envLines += "VBW_DOMAIN=$Domain" }
    $envFile = Join-Path $tmp "vbw.env"
    Write-TextFile $envFile (($envLines -join "`n") + "`n")
    $remoteScript = Join-Path $tmp "deploy_remote.sh"
    Write-TextFile $remoteScript (Get-Content (Join-Path $Root "scripts\deploy_remote.sh") -Raw)
    $uploads += @($envFile, $remoteScript)

    Write-Step "Uploading to root@${ip}:/root/"
    if ((Invoke-Native "scp" ($sshOpts + $uploads + @("root@${ip}:/root/"))) -ne 0) { throw "scp failed" }

    Write-Step "Building and starting the stack on the instance (first build takes several minutes)"
    $remoteArgs = if ($GitUrl) { "--git '$GitUrl' --branch '$Branch'" } else { "--bundle /root/vbw.tgz" }
    if ($useLocalDb) { $remoteArgs += " --local-db" }
    if ($Domain) { $remoteArgs += " --edge" }
    $code = Invoke-Native "ssh" ($sshOpts + @("root@$ip", "bash /root/deploy_remote.sh $remoteArgs --env /root/vbw.env"))
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

Write-Step "Result"
$consoleUrl = if ($Domain) { "https://$Domain" } else { "http://${ip}:3000" }
try {
    $health = Invoke-RestMethod -Uri "$apiUrl/api/health" -TimeoutSec 10
    Write-Host "  API health   : ok (db: $($health.db | ConvertTo-Json -Compress))" -ForegroundColor Green
} catch {
    Write-Host "  API health   : not reachable from here yet ($($_.Exception.Message))" -ForegroundColor Yellow
}
Write-Host "  console      : $consoleUrl"
Write-Host "  API          : $apiUrl/api/health"
Write-Host "  ssh          : ssh -i $SshKeyPath root@$ip"
Write-Host "  logs         : ssh -i $SshKeyPath root@$ip `"cd /opt/vbw; docker compose -f infra/docker-compose.yml logs -f server`""
Write-Host "  tear down    : powershell -ExecutionPolicy Bypass -File scripts\deploy_vultr_destroy.ps1 -Yes" -ForegroundColor Yellow
if ($code -ne 0) { exit $code }
