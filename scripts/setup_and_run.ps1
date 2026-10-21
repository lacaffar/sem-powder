# Set up (or update) the Python environment, then run a stage.
# Called by hunt.ps1 and run.cmd as a script block (works even where Group Policy
# blocks .ps1 files). No admin rights needed: Python and packages go under the
# user's folders via uv.
param(
    [string]$RootDir = "",
    [string]$Stage = "auto",
    [switch]$Cpu,
    [switch]$SetupOnly,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = "Continue"   # native tools write progress to stderr; check exit codes instead
$Root = if ($RootDir) { (Resolve-Path $RootDir).Path } else { Split-Path -Parent $PSScriptRoot }
$Tools = if ($env:SEM_TOOLS) { $env:SEM_TOOLS } else { Join-Path $HOME "sem-tools" }
$Uv = Join-Path $Tools "uv\uv.exe"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools "python"
$env:UV_CACHE_DIR = Join-Path $Tools "uv-cache"
$env:UV_LINK_MODE = "copy"
$env:PYTHONIOENCODING = "utf-8"
$Py = Join-Path $Root ".venv\Scripts\python.exe"

function Say($m) { Write-Host "[setup] $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "[setup] ERROR: $m" -ForegroundColor Red; throw $m }

if (-not (Test-Path $Uv)) { Fail "uv not found at $Uv. Run the hunt.ps1 one-liner first (it installs the tools)." }

# ---- choose the PyTorch build from the NVIDIA driver
$smi = @("nvidia-smi", "$env:SystemRoot\System32\nvidia-smi.exe",
         "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe") |
       Where-Object { Get-Command $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
$driver = $null
if ($smi -and -not $Cpu) {
    $q = & $smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1
    if ($q) {
        Say "GPU: $q"
        try { $driver = [version](($q -split ",")[1].Trim()) } catch { $driver = $null }
    }
}
if ($Cpu -or -not $driver) {
    $flavor = "cpu"; $torch = "2.14.1"; $tv = "0.29.1"
    Say "No NVIDIA GPU detected (or -Cpu): CPU build of PyTorch"
} elseif ($driver -ge [version]"528.33") {
    $flavor = "cu126"; $torch = "2.14.1"; $tv = "0.29.1"
} elseif ($driver -ge [version]"452.39") {
    $flavor = "cu118"; $torch = "2.7.1"; $tv = "0.22.1"
} else {
    $flavor = "cpu"; $torch = "2.14.1"; $tv = "0.29.1"
    Say "NVIDIA driver $driver is too old for CUDA builds: CPU build"
}

# ---- environment (skipped when nothing changed)
$req = Join-Path $Root "requirements.txt"
$stampWant = "$flavor torch=$torch tv=$tv req=" + (Get-FileHash $req -Algorithm SHA256).Hash
$stampFile = Join-Path $Root ".venv\semchar-env.txt"
$stampHave = if (Test-Path $stampFile) { (Get-Content $stampFile -Raw).Trim() } else { "" }
if (-not (Test-Path $Py) -or $stampHave -ne $stampWant) {
    Say "Installing Python 3.12 + packages ($flavor). First time: ~10-20 min, ~6 GB."
    & $Uv python install 3.12 --no-bin
    if ($LASTEXITCODE -ne 0) { Fail "Python install failed" }
    if (-not (Test-Path $Py)) {
        & $Uv venv --python 3.12 (Join-Path $Root ".venv")
        if ($LASTEXITCODE -ne 0) { Fail "venv creation failed" }
    }
    # exact local tags (+cu126/+cu118/+cpu) so a change of build replaces the old one
    & $Uv pip install --python $Py "torch==$torch+$flavor" "torchvision==$tv+$flavor" --index-url "https://download.pytorch.org/whl/$flavor"
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed" }
    & $Uv pip install --python $Py -r $req
    if ($LASTEXITCODE -ne 0) { Fail "package install failed" }
    Set-Content -Path $stampFile -Value $stampWant
    Say "Environment ready."
} else {
    Say "Environment up to date ($flavor)."
}
if ($SetupOnly) { return }

# ---- run (Python keeps its own full log in runs\logs\)
Say "Running stage '$Stage' $Rest"
Push-Location $Root
$started = Get-Date
try {
    & $Py -m semchar $Stage @Rest
    $code = $LASTEXITCODE
} finally { Pop-Location }
if ($code -ne 0) {
    Write-Host "`n[setup] Stage '$Stage' ended with an error (exit $code). Logs: $Root\runs\logs\" -ForegroundColor Red
} else {
    Write-Host "`n[setup] Stage '$Stage' finished." -ForegroundColor Green
    $latest = Join-Path $Root "runs\LATEST.txt"
    if ((Test-Path $latest) -and ((Get-Item $latest).LastWriteTime -ge $started)) {   # only results from this run
        $name = (Get-Content $latest -Raw).Trim()
        Write-Host "[setup] Results: $Root\runs\$name\index.html" -ForegroundColor Green
        Write-Host "[setup] To bring results back: copy $Root\runs\$name.zip (OneDrive / USB)." -ForegroundColor Green
    }
}
