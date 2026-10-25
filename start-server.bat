@echo off
chcp 850 >nul
title RAG WebUI (DB-Auswahl)
setlocal EnableExtensions

rem ============================================================
rem  RAG WebUI mit Datenbank-Auswahl (Haupt-RAG / Buch-RAG)
rem  Version 2026-10-25b
rem
rem  REPARIERT:
rem   * dieses Skript rief frueher "%ROOT%webui_server.py" auf - diese
rem     Datei existiert im Projekt nicht, der Start brach also immer mit
rem     "can't open file ... webui_server.py" ab. Jetzt wird - wie bei
rem     start.bat - webui.py gestartet.
rem   * kein Vite-Dev-Server mehr auf Port 7960: src\frontend\main.tsx
rem     existiert inzwischen, dadurch haette Vite den Port der
rem     Gradio-WebUI belegt und nur eine kaputte Seite geliefert.
rem     React-UI bei Bedarf manuell: npm run dev
rem   * alte Instanzen werden jetzt locale-unabhaengig ueber PowerShell
rem     beendet (vorher findstr "LISTENING" - auf deutschem Windows
rem     heisst der Status "ABHOEREN", es wurde nie etwas beendet)
rem
rem  Reines ASCII + CRLF.
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
set "BACKEND_PORT=7961"

if exist "%ROOT%env.bat" call "%ROOT%env.bat"
set "PYTHONNOUSERSITE=1"

if not exist "%ENV%\python.exe" goto :no_env

echo.
echo  +------------------------------------------+
echo  ^|      RAG WebUI -- Datenbank auswaehlen   ^|
echo  +------------------------------------------+
echo.
echo  [1]  Haupt-RAG   (chroma_zaebalo)
echo  [2]  Buch-RAG    (bookrag)
echo.
set "WAHL="
set /p "WAHL= Deine Wahl [1/2]: "

if "%WAHL%"=="2" goto :db_books
set "RAG_DB=%ROOT%chroma_zaebalo"
set "RAG_COLLECTION=zaebalo"
echo.
echo  [OK] Haupt-RAG gewaehlt: chroma_zaebalo
goto :db_done

:db_books
set "RAG_DB=%ROOT%bookrag"
set "RAG_COLLECTION=books"
echo.
echo  [OK] Buch-RAG gewaehlt: bookrag

:db_done
echo.
echo  RAG_DB         : %RAG_DB%
echo  RAG_COLLECTION : %RAG_COLLECTION%
echo  Adresse        : http://127.0.0.1:%WEBUI_PORT%  (LAN: http://PC-IP:%WEBUI_PORT%)
echo  KoboldCpp auf Port 5001 wird automatisch erkannt.
echo.

call :kill_port %WEBUI_PORT% WebUI
call :kill_port %BACKEND_PORT% Backend

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
rem Nur der LISTENER auf genau diesem Port wird beendet (locale-unabhaengig
rem ueber PowerShell). Niemals pauschal python.exe / node.exe abschiessen.
powershell -NoProfile -Command "$p = Get-NetTCPConnection -LocalPort %_P% -State Listen -ErrorAction SilentlyContinue; if ($p) { $p | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Write-Host ('    Beende alten %_NAME% - PID ' + $_); Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } }"
exit /b 0
