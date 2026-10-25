@echo off
chcp 850 >nul
title RAG WebUI
setlocal EnableExtensions

rem ============================================================
rem  RAG WebUI Starter - Version 2026-10-25b
rem
rem  Reine ASCII-Syntax + CRLF-Zeilenenden (keine Umlaute im Code),
rem  damit cmd.exe (auch deutsche Windows-Versionen) die Klammer-
rem  blocke nicht zerstoert.
rem
rem  Startet die Python/Gradio-WebUI auf Port 7960.
rem
rem  Bewusst NICHT gestartet:
rem   * Vite-Dev-Server - er wuerde denselben Port 7960 belegen und die
rem     Gradio-WebUI blockieren. Fuer die React-UI manuell: npm run dev
rem   * Node-Backend (Port 7961) - wird von der Gradio-WebUI nicht genutzt.
rem     Falls doch benoetigt: vorher  set START_NODE_BACKEND=1  setzen.
rem ============================================================

set "ROOT=%~dp0"
set "CONDA=%ROOT%miniforge"
set "ENV=%ROOT%env"

rem --- llama.cpp / Modelle -------------------------------------
rem Diese Variablen MUESSEN hier gesetzt werden (nicht in einer anderen CMD!),
rem damit webui.py / rag_core.py sie beim Start sieht.
if not defined LLAMA_EXE set "LLAMA_EXE=G:\llama.cpp\llama-b9204-bin-win-cuda-13.1-x64\llama-server.exe"
if not defined MODELS_ROOT set "MODELS_ROOT=C:\Users\bunny\.lmstudio\models"
if not defined MODELS_ROOT2 set "MODELS_ROOT2=G:\ComfyUI\models\LLM"
if not defined CTX_SIZE set "CTX_SIZE=8192"
if not defined THREADS set "THREADS=8"
if not defined NGL set "NGL=99"

rem --- Netzwerk / Ports ---------------------------------------
rem 0.0.0.0 = WebUI auch im LAN erreichbar (http://<IP-des-PCs>:7960)
rem 127.0.0.1 = nur dieser PC
set "WEBUI_PORT=7960"
set "WEBUI_HOST=0.0.0.0"
set "BACKEND_PORT=7961"

if not exist "%LLAMA_EXE%" echo [WARN] LLAMA_EXE nicht gefunden: %LLAMA_EXE%

rem Optionale extra Variablen aus env.bat (z.B. API-Keys), falls vorhanden
if exist "%ROOT%env.bat" call "%ROOT%env.bat"

rem Benutzerpakete aus AppData\Roaming abschalten (kollidieren mit der portablen Umgebung)
set "PYTHONNOUSERSITE=1"

if not exist "%ENV%\python.exe" goto :no_env

rem ------------------------------------------------------------
rem Alte Instanzen auf den eigenen Ports beenden (nur deren PID).
rem ------------------------------------------------------------
call :kill_port %WEBUI_PORT% WebUI
call :kill_port %BACKEND_PORT% Backend
goto :main

:main
echo.
echo ============================================
echo  RAG WebUI  -  http://127.0.0.1:%WEBUI_PORT%
echo  KoboldCpp auf Port 5001 wird automatisch erkannt.
echo ============================================
echo.
echo --- Einstellungen (settings.txt) ---
if exist "%ROOT%settings.txt" type "%ROOT%settings.txt"
echo.
echo ------------------------------------
echo.

if not defined START_NODE_BACKEND goto :skip_node
if not exist "%ROOT%package.json" goto :skip_node
if not exist "%ROOT%src\backend\server.ts" goto :skip_node
echo --- Starte Node-Backend (Port %BACKEND_PORT%) ---
if not exist "%ROOT%node_modules" (
    pushd "%ROOT%"
    call npm install
    popd
)
start "RAG Node-Backend" /D "%ROOT%" cmd /k "set BACKEND_PORT=%BACKEND_PORT%&& npm run start:backend"
timeout /t 2 >nul
:skip_node

echo --- Starte Python-WebUI (Gradio, Port %WEBUI_PORT%) ---
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
echo          Bitte zuerst setup.bat ausfuehren.
pause
exit /b 1

:kill_port
rem Aufruf: call :kill_port <PORT> <Anzeigename>
set "_P=%~1"
set "_NAME=%~2"
echo --- Pruefe alten %_NAME%-Prozess auf Port %_P% ---
rem Es wird NUR der LISTENER auf genau diesem Port beendet (locale-unabhaengig
rem ueber PowerShell). Niemals pauschal python.exe / node.exe abschiessen -
rem das wuerde fremde Programme (VS Code, Discord, ...) mitkillen.
powershell -NoProfile -Command "$p = Get-NetTCPConnection -LocalPort %_P% -State Listen -ErrorAction SilentlyContinue; if ($p) { $p | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Write-Host ('    Beende alten %_NAME% - PID ' + $_); Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } }"
exit /b 0
