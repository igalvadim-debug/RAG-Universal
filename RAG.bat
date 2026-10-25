@echo off
chcp 850 >nul
title RAG + Qwen2.5 Coder + llama.cpp
setlocal

set "ROOT=%~dp0"

:: = Modelle & Server =
if not defined LLAMA_EXE set "LLAMA_EXE=llama-server.exe"
if not defined MODEL_PATH set "MODEL_PATH=C:\Users\bunny\.lmstudio\models\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
if not defined MODELS_ROOT set "MODELS_ROOT=C:\Users\bunny\.lmstudio\models"
if not defined WEBUI_PORT set "WEBUI_PORT=7960"
if not defined WEBUI_HOST set "WEBUI_HOST=0.0.0.0"
if not defined CTX_SIZE set "CTX_SIZE=8192"
if not defined THREADS set "THREADS=8"
if not defined NGL set "NGL=99"

echo ================================
echo   RAG-CLI mit lokalem llama-server
echo ================================
echo Modell: "%MODEL_PATH%"
echo llama.cpp: "%LLAMA_EXE%"
echo WebUI-Port im Skript: %WEBUI_PORT% (CLI nutzt aber nur rag_query.py)
echo.

if not exist "%LLAMA_EXE%" echo [WARN] LLAMA_EXE nicht gefunden: %LLAMA_EXE%
if not exist "%MODEL_PATH%" echo [WARN] MODEL_PATH nicht gefunden: %MODEL_PATH%

echo.
echo ================================
echo   Starte llama-server (lokal, Port 8080)
echo ================================
start "" /MIN cmd /c ^
    "%LLAMA_EXE%" ^
    -m "%MODEL_PATH%" ^
    -ngl %NGL% ^
    -c %CTX_SIZE% ^
    --flash-attn on ^
    --port 8080 ^
    --host 127.0.0.1 ^
    --threads %THREADS%

echo.
echo Warte 3 Sekunden, bis der Server bereit ist...
timeout /t 3 >nul

echo.
echo ================================
echo   Starte RAG-Client
echo ================================
python "%ROOT%rag_query.py"

echo.
echo ================================
echo   Fertig
echo ================================
pause
