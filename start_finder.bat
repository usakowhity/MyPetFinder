@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
chcp 65001 >nul

set "PY="
if exist "python\python.exe" set "PY=python\python.exe"
if defined PY goto havepy
for %%V in (3.12 3.13 3.11) do (
  if not defined PY (
    py -%%V --version >nul 2>nul
    if not errorlevel 1 set "PY=py -%%V"
  )
)
if not defined PY goto nopy

:havepy
echo Using: %PY%
echo Starting MyPetFinder. Close this window to stop.
%PY% finder.py
echo.
echo The search page stopped (exit code %errorlevel%). If there is an error message above, please copy it.
pause
goto end

:nopy
echo Python 3.11+ was not found. Put the portable python folder here, or install Python 3.12.
pause

:end
endlocal
