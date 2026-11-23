@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ---------------------------------------------------------------
rem  Rebuild the ChordEdit environment from scratch.
rem  Needs: Python 3.12 (py -3.12), internet access.
rem  Installed layout: .venv\  +  models\sd-turbo\
rem ---------------------------------------------------------------

set "MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PYEXE=.venv\Scripts\python.exe"

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python launcher "py" not found. Install Python 3.12 first.
    pause
    exit /b 1
)

echo [1/5] creating virtualenv with Python 3.12 ...
py -3.12 -m venv .venv
if errorlevel 1 (
    echo [ERROR] venv creation failed.
    pause
    exit /b 1
)
%PYEXE% -m pip install --upgrade pip

echo [2/5] installing torch 2.5.0 + cu118 (works with driver >= 452; cu124 needs >= 527) ...
%PYEXE% -m pip install torch==2.5.0+cu118 torchvision==0.20.0+cu118 torchaudio==2.5.0+cu118 ^
    --index-url https://download.pytorch.org/whl/cu118 --extra-index-url %MIRROR%
if errorlevel 1 goto :fail

echo [3/5] installing the rest of the requirements ...
%PYEXE% -m pip install -i %MIRROR% ^
    "numpy==1.26.4" "Pillow==10.4.0" "pyyaml" "pandas==2.2.3" "matplotlib==3.9.2" "seaborn==0.13.2" ^
    "contourpy==1.3.0" "setuptools" "scipy==1.13.1" "pyarrow==17.0.0" ^
    "diffusers==0.32.2" "transformers==4.46.3" "gradio==5.50.0" ^
    "torchmetrics==1.8.2" "torch-fidelity==0.3.0" "datasets>=3,<4"
if errorlevel 1 goto :fail

rem gradio pulls huggingface_hub 1.x/2.x which breaks transformers 4.46 -> pin it back
echo       pinning huggingface_hub back to the 0.x line ...
%PYEXE% -m pip install -i %MIRROR% "huggingface_hub==0.36.2"
if errorlevel 1 goto :fail

echo [4/5] downloading sd-turbo weights (about 2.5 GB) ...
%PYEXE% tools\download_sd_turbo.py
if errorlevel 1 goto :fail

echo [5/5] environment self-check ...
%PYEXE% tools\check_env.py

echo.
echo Done. Start the demo with:  run_app.bat
pause
exit /b 0

:fail
echo.
echo [ERROR] a step failed, see the output above.
pause
exit /b 1
