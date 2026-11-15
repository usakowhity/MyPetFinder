@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
chcp 65001 >nul

rem --- Install the libraries needed for Step 2 (AI analysis) into the portable python\ folder ---
if exist "python\python.exe" goto havepy
echo Downloading portable Python 3.12 (about 10 MB) ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; try { Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.12.7/python-3.12.7-embed-amd64.zip' -OutFile 'python_embed.zip'; Expand-Archive -Force 'python_embed.zip' 'python'; Remove-Item 'python_embed.zip' } catch { Write-Host $_; exit 1 }"
if errorlevel 1 goto fail
if not exist "python\python.exe" goto fail
:havepy
set "PY=python\python.exe"

rem 1) enable site-packages in the embeddable Python
set "PTH="
for %%F in (python\python3*._pth) do set "PTH=%%F"
if not defined PTH (
  echo python3xx._pth was not found in the python folder.
  pause
  goto end
)
%PY% -I -c "import sys,re;p=sys.argv[1];t=open(p,encoding='utf-8').read();t=re.sub(r'^#\s*import site','import site',t,flags=re.M);t=t if 'Lib\\site-packages' in t else t.rstrip()+'\nLib\\site-packages\n';open(p,'w',encoding='utf-8').write(t)" "%PTH%"
if not exist "python\Lib\site-packages" mkdir "python\Lib\site-packages"

rem 2) pip
%PY% -m pip --version >nul 2>nul
if errorlevel 1 (
  echo Downloading get-pip.py ...
  %PY% -I -c "import urllib.request;urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py','python\\get-pip.py')"
  if errorlevel 1 goto fail
  %PY% python\get-pip.py --no-warn-script-location
  if errorlevel 1 goto fail
)

rem 3) libraries
%PY% -m pip install --no-warn-script-location numpy pillow onnxruntime openvino opencv-python-headless
if errorlevel 1 goto fail
echo.
echo Done. Now run start_finder.bat.
pause
goto end

:fail
echo.
echo Setup failed. Please copy the messages above and send them to Claude.
pause

:end
endlocal
