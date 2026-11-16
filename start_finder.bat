@echo off
setlocal enableextensions
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
chcp 65001 >nul

rem --- first run: set up Python and libraries automatically ----------
if not exist "python\.ready" (
  echo First run: setting up. This takes a few minutes and needs the internet.
  call "%~dp0setup_finder.bat" /auto
  if errorlevel 1 (
    pause
    goto end
  )
)

echo Starting MyPetFinder. Close this window to stop.
python\python.exe finder.py
echo.
echo MyPetFinder stopped (exit code %errorlevel%). If there is an error message above, please copy it.
pause

:end
endlocal
