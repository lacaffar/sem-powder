@echo off
rem Double-click to run the next stage, or from a terminal in this folder:
rem   run.cmd                   next stage (visit 1, then visit 2 once labels are reviewed)
rem   run.cmd label             open the outline editor on the next crop
rem   run.cmd status            labeling progress
rem   run.cmd analyze C:\path\to\images [--model cpsam_ft]   (no spaces in the path)
rem   run.cmd visit1 / visit2 / data / envcheck
setlocal
set "STAGE=%~1"
set "DBLCLICK="
if "%STAGE%"=="" (set "STAGE=auto" & set "DBLCLICK=1")
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& ([scriptblock]::Create([IO.File]::ReadAllText('%HERE%\scripts\setup_and_run.ps1'))) -RootDir '%HERE%' -Stage '%STAGE%' %2 %3 %4 %5 %6"
if defined DBLCLICK pause
endlocal
