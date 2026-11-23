@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "HF_HOME=%~dp0.cache\hf"
set "HF_HUB_CACHE=%~dp0.cache\hf\hub"
set "TRANSFORMERS_CACHE=%~dp0.cache\hf\transformers"
set "GRADIO_ANALYTICS_ENABLED=False"
set "GRADIO_TEMP_DIR=%~dp0.cache\gradio"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run setup_env.bat first.
    pause
    exit /b 1
)
if not exist "models\sd-turbo\unet\config.json" (
    echo [ERROR] sd-turbo weights missing under models\sd-turbo
    echo         download them with: .venv\Scripts\python.exe tools\download_sd_turbo.py
    pause
    exit /b 1
)

set "PORT=7860"
if not "%~1"=="" set "PORT=%~1"

echo ============================================================
echo  ChordEdit web demo
echo  model : models\sd-turbo   precision: fp16
echo  url   : http://127.0.0.1:%PORT%
echo  press Ctrl+C to stop
echo ============================================================

.venv\Scripts\python.exe app.py --model-root models\sd-turbo --server-port %PORT% --precision fp16

echo.
echo [app exited]
pause
