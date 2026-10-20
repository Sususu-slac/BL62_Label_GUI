@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ENV_NAME=BL62_Label_GUI"
set "LOGFILE=%~dp0install_log.txt"
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
echo BL62 Label GUI installation started %date% %time%>"%LOGFILE%"

set "CONDA_EXE="
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%I"
if not defined CONDA_EXE if exist "%USERPROFILE%\AppData\Local\miniforge3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\AppData\Local\miniforge3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\miniforge3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniforge3\Scripts\conda.exe"
if not defined CONDA_EXE (
  echo ERROR: Conda was not found.
  echo Install Miniforge, then run this file from Miniforge Prompt.
  pause
  exit /b 1
)

echo Creating or updating the isolated %ENV_NAME% environment...
call "%CONDA_EXE%" run -n %ENV_NAME% python --version >nul 2>nul
if errorlevel 1 call "%CONDA_EXE%" create -n %ENV_NAME% python=3.12 pip -y >>"%LOGFILE%" 2>&1
if errorlevel 1 goto :error

echo Installing NVIDIA PyTorch. This download can take several minutes...
call "%CONDA_EXE%" run -n %ENV_NAME% python -m pip install torch==2.10.0 torchvision --index-url %TORCH_INDEX_URL% >>"%LOGFILE%" 2>&1
if errorlevel 1 goto :error

echo Installing GUI and SAM3 runtime dependencies...
call "%CONDA_EXE%" run -n %ENV_NAME% python -m pip install -r "%~dp0requirements.txt" >>"%LOGFILE%" 2>&1
if errorlevel 1 goto :error

echo Running installation diagnostics...
call "%CONDA_EXE%" run --no-capture-output -n %ENV_NAME% python "%~dp0diagnose_installation.py" >>"%LOGFILE%" 2>&1
if errorlevel 1 goto :error
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_shortcut.ps1" >>"%LOGFILE%" 2>&1

echo.
echo Installation completed successfully.
echo A BL62 Label GUI shortcut was created on the Desktop.
echo Double-click BL62 Label GUI.vbs to start without a console window.
echo SAM3 and checkpoint paths are auto-detected; use the selection buttons if needed.
pause
exit /b 0

:error
echo.
echo Installation failed. See:
echo %LOGFILE%
pause
exit /b 1
