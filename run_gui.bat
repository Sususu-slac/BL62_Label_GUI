@echo off
setlocal
cd /d "%~dp0"
set "CONDA_EXE="
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%I"
if not defined CONDA_EXE if exist "%USERPROFILE%\AppData\Local\miniforge3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\AppData\Local\miniforge3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\miniforge3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniforge3\Scripts\conda.exe"
if not defined CONDA_EXE exit /b 1
call "%CONDA_EXE%" run --no-capture-output -n BL62_Label_GUI pythonw "%~dp0app_en.py"
