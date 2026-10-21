# One command on the Hunt GPU PC (PowerShell, no admin):
#
#   irm https://raw.githubusercontent.com/lacaffar/sem-powder/main/hunt.ps1 | iex
#
# 1. installs uv (Python manager) and portable Git into %USERPROFILE%\sem-tools (pinned, hash-checked)
# 2. clones or updates the repository into %USERPROFILE%\sem-powder
# 3. sets up Python + CUDA PyTorch + packages, downloads the data
# 4. runs the next stage automatically: visit 1 first time, visit 2 once labels are reviewed
# To force a stage, first run e.g.:  $env:SEM_STAGE = "visit1"   (visit2, label, status, data, envcheck)

& {
    $ErrorActionPreference = "Stop"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is much faster without the progress bar
    $SemRoot = if ($env:SEM_ROOT) { $env:SEM_ROOT } else { Join-Path $HOME "sem-powder" }
    $SemTools = if ($env:SEM_TOOLS) { $env:SEM_TOOLS } else { Join-Path $HOME "sem-tools" }
    $SemStage = if ($env:SEM_STAGE) { $env:SEM_STAGE } else { "auto" }
    $SemRepo = "https://github.com/lacaffar/sem-powder.git"

    function Get-PinnedZip($url, $sha, $dest) {
        if (Test-Path $dest) { return }
        $zip = Join-Path $SemTools ([IO.Path]::GetFileName($url))
        Write-Host "[hunt] downloading $url" -ForegroundColor Cyan
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        $h = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if ($h -ne $sha) { Remove-Item $zip; throw "checksum mismatch for $url" }
        Expand-Archive -Path $zip -DestinationPath $dest -Force
        Remove-Item $zip
    }

    New-Item -ItemType Directory -Force -Path $SemTools | Out-Null
    Get-PinnedZip "https://github.com/astral-sh/uv/releases/download/0.12.21/uv-x86_64-pc-windows-msvc.zip" `
        "5d223efa0bf00208c3853246af09420419dfbd352536aa6bb8163d6170e23890" (Join-Path $SemTools "uv")
    Get-PinnedZip "https://github.com/git-for-windows/git/releases/download/v2.55.0.windows.5/MinGit-2.55.0.5-64-bit.zip" `
        "56d7b226b7693196cfc71fef26568f536c4a021ab6c37ff2db4287bed908e96e" (Join-Path $SemTools "git")
    $SemGit = Join-Path $SemTools "git\cmd\git.exe"

    $ErrorActionPreference = "Continue"   # git reports progress on stderr
    if (Test-Path (Join-Path $SemRoot ".git")) {
        Write-Host "[hunt] updating $SemRoot" -ForegroundColor Cyan
        & $SemGit -C $SemRoot pull --ff-only
        if ($LASTEXITCODE -ne 0) { Write-Host "[hunt] git pull failed; continuing with the existing code" -ForegroundColor Yellow }
    } else {
        if (Test-Path $SemRoot) {
            $bak = "$SemRoot-old-" + (Get-Date -Format "yyyyMMdd-HHmmss")
            Write-Host "[hunt] $SemRoot exists but is not a git checkout; moving it to $bak" -ForegroundColor Yellow
            Move-Item $SemRoot $bak
        }
        Write-Host "[hunt] cloning into $SemRoot" -ForegroundColor Cyan
        & $SemGit clone $SemRepo $SemRoot
        if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    }

    if ($env:SEM_NO_SETUP) { Write-Host "[hunt] SEM_NO_SETUP set: stopping before setup"; return }
    $setup = Join-Path $SemRoot "scripts\setup_and_run.ps1"
    & ([scriptblock]::Create([IO.File]::ReadAllText($setup))) -RootDir $SemRoot -Stage $SemStage
}
