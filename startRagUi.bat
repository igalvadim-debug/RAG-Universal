@echo off
chcp 850 >nul
title RAG WebUI (schlank)
setlocal EnableExtensions

rem ============================================================
rem  RAG WebUI Starter (Alternative zu start.bat) - 2026-10-25b
rem
rem  Startet NUR die Python/Gradio-WebUI auf Port 7960.
rem
rem  FRUEHERE VERSIONEN waren kaputt:
rem   * "%node --version%" ist keine gueltige Variable -> "... kann
rem     syntaktisch an dieser Stelle nicht verarbeitet werden"
rem   * taskkill /f /im node.exe hat ALLE Node-Prozesse des PCs
rem     abgeschossen (VS Code, Discord, ...)
rem   * React/Vite-Frontend auf Port 7960 gestartet -> belegte den Port
rem     der Gradio-WebUI, die echte Oberflaeche kam nie hoch
rem   * Erkennung alter Instanzen ueber findstr "LISTENING" - auf
rem     deutschem Windows heisst der Status "ABHOEREN", es wurde also nie
rem     etwas beendet und der Start scheiterte an "Port already in use"
rem
rem  Reines ASCII + CRLF, damit cmd.exe (auch DE-Windows) nicht stolpert.
rem ============================================================

set "ROOT=%~dp0"
set "CONDA=%ROOT%miniforge"
set "ENV=%ROOT%env"

rem --- llama.cpp / Modelle -------------------------------------
if not defined LLAMA_EXE set "LLAMA_EXE=G:\llama.cpp\llama-b9204-bin-win-cuda-13.1-x64\llama-server.exe"
if not defined MODELS_ROOT set "MODELS_ROOT=C:\Users\bunny\.lmstudio\models"
if not defined MODELS_ROOT2 set "MODELS_ROOT2=G:\ComfyUI\models\LLM"
if not defined CTX_SIZE set "CTX_SIZE=8192"
if not defined THREADS set "THREADS=8"
if not defined NGL set "NGL=99"

rem --- Netzwerk / Ports ---------------------------------------
set "WEBUI_PORT=7960"
set "WEBUI_HOST=0.0.0.0"

if exist "%ROOT%env.bat" call "%ROOT%env.bat"
set "PYTHONNOUSERSITE=1"

if not exist "%ENV%\python.exe" goto :no_env

echo --- RAG WebUI (Python/Gradio) ---
echo Adresse: http://127.0.0.1:%WEBUI_PORT%

rem Alte Instanz auf Port %WEBUI_PORT% beenden (nur der Listener, NIE pauschal)
call :kill_port %WEBUI_PORT% WebUI

echo.
"%CONDA%\Scripts\conda.exe" run -p "%ENV%" --no-capture-output python -s "%ROOT%webui.py"
set "RC=%ERRORLEVEL%"
echo.
echo WebUI beendet (Exit-Code %RC%).
if not "%RC%"=="0" echo [HINWEIS] Fehler oben pruefen: Port belegt, Gradio-Version, fehlendes Paket.
echo.
pause
exit /b %RC%

:no_env
echo [FEHLER] Umgebung nicht gefunden: %ENV%\python.exe
echo          Bitte zuerst setup.bat ausfuehren (oder start.bat nutzen).
pause
exit /b 1

:kill_port
rem Aufruf: call :kill_port <PORT> <Anzeigename>
set "_P=%~1"
set "_NAME=%~2"
echo --- Pruefe alten %_NAME%-Prozess auf Port %_P% ---
rem Nur der LISTENER auf genau diesem Port wird beendet (locale-unabhaengig
rem ueber PowerShell). Niemals pauschal python.exe / node.exe abschiessen.
powershell -NoProfile -Command "$p = Get-NetTCPConnection -LocalPort %_P% -State Listen -ErrorAction SilentlyContinue; if ($p) { $p | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Write-Host ('    Beende alten %_NAME% - PID ' + $_); Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } }"
exit /b 0
