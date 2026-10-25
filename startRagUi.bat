@echo off
chcp 850 >nul
setlocal EnableExtensions

rem ============================================================
rem  RAG WebUI Starter (Alternative zu start.bat) - 2026-10-25
rem
rem  FRUEHERE VERSIONEN waren kaputt:
rem   * "%node --version%" ist keine gueltige Variable -> Fehlermeldung
rem     "... kann syntaktisch an dieser Stelle nicht verarbeitet werden"
rem   * taskkill /f /im node.exe hat ALLE Node-Prozesse des PCs
rem     abgeschossen (VS Code, Discord, ...)
rem   * hat ein React/Vite-Frontend gestartet, das es in diesem Projekt
rem     gar nicht gibt (src/ fehlt) -> "totale kaputte Seite"
rem   * lief auf Port 7860/7960 durch den Vite-Dev-Server, die echte
rem     WebUI (Gradio) kam nie hoch
rem
rem  Diese Version startet nur die echte Python/Gradio-WebUI auf Port 7960.
rem  Reines ASCII + CRLF, damit cmd.exe (auch DE-Windows) nicht stolpert.
rem ============================================================

set "ROOT=%~dp0"
set "CONDA=%ROOT%miniforge"
set "ENV=%ROOT%env"
set "WEBUI_PORT=7960"
set "WEBUI_HOST=0.0.0.0"

rem llama.cpp-Pfade (nur setzen, falls nicht schon gesetzt)
if not defined LLAMA_EXE set "LLAMA_EXE=G:\llama.cpp\llama-b9204-bin-win-cuda-13.1-x64\llama-server.exe"
if not defined MODELS_ROOT set "MODELS_ROOT=C:\Users\bunny\.lmstudio\models"
if not defined MODELS_ROOT2 set "MODELS_ROOT2=G:\ComfyUI\models\LLM"
if not defined CTX_SIZE set "CTX_SIZE=8192"
if not defined NGL set "NGL=99"
set "PYTHONNOUSERSITE=1"

if exist "%ROOT%env.bat" call "%ROOT%env.bat"

if not exist "%ENV%\python.exe" goto :no_env

echo --- RAG WebUI (Python/Gradio) ---
echo Adresse: http://127.0.0.1:%WEBUI_PORT%
echo.

rem Alte Instanz auf Port %WEBUI_PORT% beenden (nur deren PID, NIE pauschal)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /r /i /c:"[.:]%WEBUI_PORT% " /c:"[.:]%WEBUI_PORT%] "') do (
    if not "%%a"=="0" (
        echo     Beende alten Prozess auf Port %WEBUI_PORT% - PID %%a
        taskkill /f /pid %%a >nul 2>&1
    )
)

"%CONDA%\Scripts\conda.exe" run -p "%ENV%" --no-capture-output python -s "%ROOT%webui.py"

echo.
echo WebUI beendet.
pause
exit /b 0

:no_env
echo [FEHLER] Keine portable Umgebung gefunden: %ENV%\python.exe
echo          Bitte zuerst setup.bat ausfuehren (oder start.bat nutzen).
pause
exit /b 1
