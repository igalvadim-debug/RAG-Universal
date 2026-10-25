@echo off
chcp 65001 >nul
title RAG WebUI
setlocal

set "ROOT=%~dp0"
set "CONDA=%ROOT%miniforge"
set "ENV=%ROOT%env"

:: ── llama.cpp / Modelle ─────────────────────────────────────────────
:: Diese Variablen MUESSEN hier gesetzt sein (nicht in einer anderen CMD!),
:: damit webui.py / rag_core.py sie beim Start sieht.
set "LLAMA_EXE=G:\llama.cpp\llama-b9204-bin-win-cuda-13.1-x64\llama-server.exe"
set "MODELS_ROOT=C:\Users\bunny\.lmstudio\models"
set "MODELS_ROOT2=G:\ComfyUI\models\LLM"
set "CTX_SIZE=8192"
set "THREADS=8"
set "NGL=99"

if not exist "%LLAMA_EXE%" echo [WARN] LLAMA_EXE nicht gefunden: %LLAMA_EXE%

:: Optionale extra Variablen aus env.bat (z.B. API-Keys), falls vorhanden
if exist "%ROOT%env.bat" call "%ROOT%env.bat"

:: Отключаем пользовательские пакеты из AppData\Roaming
:: (они конфликтуют с portable окружением)
set "PYTHONNOUSERSITE=1"

if not exist "%ENV%\python.exe" (
    echo [FEHLER] Umgebung nicht gefunden. Bitte zuerst setup.bat ausfuhren.
    pause
    exit /b 1
)

:: Alte WebUI-Instanz auf Port 7860 automatisch beenden (sonst startet die neue nicht)
:: Hinweis: Zustand kann 'LISTENING' (EN) oder 'ABHÖREN' (DE) heissen — nur auf :7860 filtern
echo --- Alte WebUI-Instanz pruefen (Port 7860) ---
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7860"') do (
    if not "%%a"=="0" (
        echo   Beende alten WebUI-Prozess PID %%a
        taskkill /f /pid %%a >nul 2>&1
    )
)
echo.

echo --- RAG WebUI ---
echo Adresse: http://127.0.0.1:7860
echo KoboldCpp auf Port 5001 wird automatisch erkannt.
echo.
echo --- Settings (settings.txt) ---
type "%ROOT%settings.txt"
echo.
echo --------------------------------
echo.

"%CONDA%\Scripts\conda.exe" run -p "%ENV%" --no-capture-output python -s "%ROOT%webui.py"

pause
