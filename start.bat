@echo off
chcp 850 >nul
title RAG WebUI
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem  RAG WebUI Starter - Version 2026-10-25
rem  Reine ASCII-Syntax + CRLF-Zeilenenden (keine Umlaute im Code),
rem  damit cmd.exe (auch deutsche Windows-Versionen) die Klammer-
rem  blocke nicht mehr zerstoert ("... kann syntaktisch an dieser
rem  Stelle nicht verarbeitet werden").
rem ============================================================

set "ROOT=%~dp0"
set "CONDA=%ROOT%miniforge"
set "ENV=%ROOT%env"

rem = llama.cpp / Modelle =
rem Diese Variablen MUESSEN hier gesetzt sein (nicht in einer anderen CMD!),
rem damit webui.py / rag_core.py sie beim Start sieht.
if not defined LLAMA_EXE set "LLAMA_EXE=G:\llama.cpp\llama-b9204-bin-win-cuda-13.1-x64\llama-server.exe"
if not defined MODELS_ROOT set "MODELS_ROOT=C:\Users\bunny\.lmstudio\models"
if not defined MODELS_ROOT2 set "MODELS_ROOT2=G:\ComfyUI\models\LLM"
if not defined CTX_SIZE set "CTX_SIZE=8192"
if not defined THREADS set "THREADS=8"
if not defined NGL set "NGL=99"

rem = WebUI-Port und Netzwerk =
set "WEBUI_PORT=7960"
rem 0.0.0.0 = WebUI im lokalen Netzwerk erreichbar (andere PCs: http://<IP-dieses-PCs>:7960)
rem 127.0.0.1 wuerde den Zugriff auf diesen Rechner beschraenken
set "WEBUI_HOST=0.0.0.0"

rem Node-Backend fuer die React/Vite-Oberflaeche (API + WebSocket, Port 7961)
set "BACKEND_PORT=7961"

if not exist "%LLAMA_EXE%" echo [WARN] LLAMA_EXE nicht gefunden: %LLAMA_EXE%

rem Optionale extra Variablen aus env.bat (z.B. API-Keys), falls vorhanden
if exist "%ROOT%env.bat" call "%ROOT%env.bat"

rem Benutzerpakete aus AppData\Roaming abschalten
rem (sie kollidieren mit der portablen Umgebung)
set "PYTHONNOUSERSITE=1"

if not exist "%ENV%\python.exe" goto :no_env

rem ------------------------------------------------------------
rem Alte Instanzen auf den Ports beenden (nur deren eigene PIDs,
rem NIE pauschal python.exe/node.exe - das wuerde andere Programme
rem des Systems abschiessen).
rem Der Zustand heisst je nach Windows 'LISTENING' (EN) oder
rem 'ABHOEREN' (DE) -- deshalb nur ":PORT " und IPv4/IPv6 filtern.
rem ------------------------------------------------------------
call :kill_port %WEBUI_PORT%   WebUI
call :kill_port %BACKEND_PORT% Backend
goto :main

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
rem netstat-Spalten: Protokoll(1) Lokale Adresse(2) Fremde(3) Status(4) PID(5).
rem Erst auf TCP+Status filtern, dann die PID-Spalte (tokens=5) nehmen.
for /f "tokens=5" %%a in ('netstat -nao ^| findstr /r /i "[.:]%_P% .*LISTENING .*ABHOEREN" ^| findstr /b /r /i "TCP"') do (
    if not "%%a"=="0" (
        echo     Beende alten %_NAME% - PID %%a
        taskkill /f /pid %%a >nul 2>&1
    )
)
exit /b 0

:main
echo.
echo ============================================
echo  RAG WebUI  -  http://127.0.0.1:%WEBUI_PORT%
echo  KoboldCpp auf Port 5001 wird automatisch erkannt.
echo ============================================
echo.
echo --- Einstellungen (settings.txt) ---
type "%ROOT%settings.txt"
echo.
echo ------------------------------------
echo.

rem ------------------------------------------------------------
rem Node-Backend (API + WebSocket) starten, falls vorhanden.
rem Das React/Vite-Frontend wird bewusst NICHT gestartet: index.html
rem verweist auf src/frontend/main.tsx, aber das src-Verzeichnis fehlt
rem in diesem Repo -> Vite liefert nur eine weisse/kaputte Seite und
rem blockiert zusaetzlich den Port, den die Gradio-WebUI braucht.
rem Wenn du spaeter doch src/ hast: manuell "npm run dev".
rem ------------------------------------------------------------
if not exist "%ROOT%package.json" goto :skip_node
if not exist "%ROOT%src\backend\server.ts" (
    echo [INFO] src\backend\server.ts fehlt - Node-Backend wird uebersprungen.
    goto :skip_node
)
echo --- Starte Node-Backend (Port %BACKEND_PORT%) ---
if not exist "%ROOT%node_modules" (
    echo     [INFO] node_modules fehlt - fuehre npm install aus...
    pushd "%ROOT%"
    call npm install
    popd
)
start "RAG Node-Backend" cmd /k "cd /d "%ROOT%" && set BACKEND_PORT=%BACKEND_PORT% && npm run start:backend"
timeout /t 2 >nul
:skip_node

echo --- Starte Python-WebUI (Gradio, Port %WEBUI_PORT%) ---
"%CONDA%\Scripts\conda.exe" run -p "%ENV%" --no-capture-output python -s "%ROOT%webui.py"

echo.
echo WebUI beendet.
pause
exit /b 0
