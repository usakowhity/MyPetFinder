@echo off
setlocal enableextensions
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1

echo ==============================================
echo   MyPetFinder setup (first time only)
echo ==============================================

rem --- 0) checks -------------------------------------------------
if /i not "%PROCESSOR_ARCHITECTURE%"=="AMD64" if /i not "%PROCESSOR_ARCHITEW6432%"=="AMD64" (
  echo This app needs 64-bit Windows 10 or 11.
  goto fail
)
if exist "python\.ready" goto done

for /f %%F in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive -Name (Get-Location).Drive.Name).Free/1GB)"') do set FREEGB=%%F
if defined FREEGB if %FREEGB% LSS 3 (
  echo Not enough free disk space on this drive: %FREEGB% GB free, about 3 GB needed.
  goto fail
)

rem --- 1) portable Python ---------------------------------------
if exist "python\python.exe" goto havepy
echo.
echo [1/3] Downloading portable Python 3.12 (about 10 MB) ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; try { Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.12.7/python-3.12.7-embed-amd64.zip' -OutFile 'python_embed.zip'; Expand-Archive -Force 'python_embed.zip' 'python'; Remove-Item 'python_embed.zip' } catch { Write-Host $_; exit 1 }"
if errorlevel 1 goto netfail
if not exist "python\python.exe" goto fail
:havepy
set "PY=python\python.exe"

rem --- 2) enable site-packages, install pip ---------------------
set "PTH="
for %%F in (python\python3*._pth) do set "PTH=%%F"
if not defined PTH (
  echo python3xx._pth was not found in the python folder.
  goto fail
)
%PY% -I -c "import sys,re;p=sys.argv[1];t=open(p,encoding='utf-8').read();t=re.sub(r'^#\s*import site','import site',t,flags=re.M);t=t if 'Lib\\site-packages' in t else t.rstrip()+'\nLib\\site-packages\n';open(p,'w',encoding='utf-8').write(t)" "%PTH%"
if not exist "python\Lib\site-packages" mkdir "python\Lib\site-packages"

%PY% -m pip --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo [2/3] Installing pip ...
  %PY% -I -c "import urllib.request;urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py','python\\get-pip.py')"
  if errorlevel 1 goto netfail
  %PY% python\get-pip.py --no-warn-script-location
  if errorlevel 1 goto netfail
)

rem --- 3) libraries ---------------------------------------------
echo.
echo [3/3] Installing libraries (about 600 MB, a few minutes) ...
%PY% -m pip install --no-warn-script-location numpy pillow onnxruntime opencv-python-headless
if errorlevel 1 goto netfail
%PY% -I -c "import numpy, PIL, onnxruntime, cv2; print('libraries OK')"
if errorlevel 1 goto fail
echo ok> "python\.ready"

:done
echo.
echo Setup finished.
if /i "%~1"=="/auto" goto end
choice /c YN /t 15 /d N /m "Create a desktop shortcut"
if errorlevel 2 goto skipshortcut
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\MyPetFinder.lnk'); $s.TargetPath='%~dp0start_finder.bat'; $s.WorkingDirectory='%~dp0'; $s.Save()"
:skipshortcut
echo Next: run start_finder.bat
pause
goto end

:netfail
echo.
echo A download failed. Please check your internet connection (and any security software), then run this file again.
:fail
echo.
echo Setup did not finish. The messages above show the reason.
if /i "%~1"=="/auto" exit /b 1
pause
exit /b 1

:end
endlocal
