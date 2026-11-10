@echo off
chcp 850 >nul
title RAG-CLI
setlocal EnableExtensions

set "ROOT=%~dp0"
set "ENV=%ROOT%env"

if exist "%ROOT%env.bat" call "%ROOT%env.bat"
if not defined LLAMA_EXE if exist "%ROOT%llama\llama-server.exe" set "LLAMA_EXE=%ROOT%llama\llama-server.exe"
if not defined MODELS_ROOT set "MODELS_ROOT=%ROOT%models\LLM"
if not defined CTX_SIZE set "CTX_SIZE=8192"
if not defined THREADS set "THREADS=8"
if not defined NGL set "NGL=99"

if not exist "%ENV%\python.exe" (
    echo [FEHLER] Python-Umgebung fehlt. Bitte zuerst setup.bat ausfuehren.
    pause
    exit /b 1
)
set "PYTHONNOUSERSITE=1"
set "PATH=%ENV%;%ENV%\Library\bin;%ENV%\Scripts;%PATH%"

echo ================================
echo   RAG-CLI
echo ================================

rem llama-server nur starten, wenn MODEL_PATH gesetzt ist und existiert
rem (sonst fragt rag_query.py nach der Modell-Datei)
if defined MODEL_PATH if exist "%MODEL_PATH%" if defined LLAMA_EXE if exist "%LLAMA_EXE%" (
    echo Starte llama-server: %MODEL_PATH%
    start "" /MIN "%LLAMA_EXE%" -m "%MODEL_PATH%" -ngl %NGL% -c %CTX_SIZE% --flash-attn on --port 8080 --host 127.0.0.1 --threads %THREADS%
    echo Warte auf den Server ...
    powershell -NoProfile -Command "$t=0; while($t -lt 120){ try { if((Invoke-WebRequest http://127.0.0.1:8080/health -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200){ exit 0 } } catch {}; Start-Sleep 1; $t++ }; exit 1"
)

"%ENV%\python.exe" -s "%ROOT%rag_query.py"

echo.
pause