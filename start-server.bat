@echo off
chcp 850 >nul
title RAG WebUI
setlocal

set "ROOT=%~dp0"
set "CONDA=%ROOT%miniforge"
set "ENV=%ROOT%env"
set "PYTHONNOUSERSITE=1"

:: = llama.cpp / Modelle (muss in derselben BAT stehen wie der WebUI-Start) =
set "LLAMA_EXE=G:\llama.cpp\llama-b9204-bin-win-cuda-13.1-x64\llama-server.exe"
set "MODELS_ROOT=C:\Users\bunny\.lmstudio\models"
set "MODELS_ROOT2=G:\ComfyUI\models\LLM"
set "CTX_SIZE=8192"
set "THREADS=8"
set "NGL=99"

:: = WebUI-Port und Netzwerk =
set "WEBUI_PORT=7960"
:: 0.0.0.0 = WebUI im lokalen Netzwerk erreichbar (andere PCs: http://<IP-dieses-PCs>:7960)
set "WEBUI_HOST=0.0.0.0"

:: Node-Backend fuer die React/Vite-Oberflaeche (API + WebSocket, Port 7961)
set "BACKEND_PORT=7961"

if not exist "%ENV%\python.exe" (
    echo [FEHLER] Umgebung nicht gefunden. Bitte zuerst setup.bat ausfuehren.
    pause
    exit /b 1
)

echo.
echo  +------------------------------------------+
echo  ^|      RAG WebUI -- Datenbank auswaehlen   ^|
echo  +------------------------------------------+
echo.
echo  [1]  Haupt-RAG   (chroma_zaebalo)
echo  [2]  Buch-RAG    (bookrag)
echo.
set /p WAHL= Deine Wahl [1/2]: 

if "%WAHL%"=="2" (
    set "RAG_DB=%ROOT%bookrag"
    set "RAG_COLLECTION=books"
    echo.
    echo  [OK] Buch-RAG gewaehlt: bookrag
) else (
    set "RAG_DB=%ROOT%chroma_zaebalo"
    set "RAG_COLLECTION=zaebalo"
    echo.
    echo  [OK] Haupt-RAG gewaehlt: chroma_zaebalo
)

echo.
echo  Adresse : http://127.0.0.1:%WEBUI_PORT%  (LAN: http://<PC-IP>:%WEBUI_PORT%)
echo  KoboldCpp auf Port 5001 wird automatisch erkannt.
echo.

rem Node-Backend (API + WebSocket) und React/Vite-Frontend starten, falls vorhanden.
rem ACHTUNG: Das Vite-Frontend braucht das src-Verzeichnis (index.html -> src/frontend/main.tsx).
rem Wenn src/ fehlt, wird KEIN Vite-Dev-Server gestartet - er wuerde nur eine kaputte
rem Seite liefern und den Port blockieren, den die Gradio-WebUI braucht.
if not exist "%ROOT%package.json" goto :skip_node
if not exist "%ROOT%src\backend\server.ts" goto :skip_node
echo --- Starte Node-Backend (Port %BACKEND_PORT%) fuer die React/Vite-Oberflaeche ---
if not exist "%ROOT%node_modules" (
    echo   [INFO] node_modules fehlt - fuehre npm install aus...
    call npm install
)
start "RAG Node-Backend" cmd /k "cd /d "%ROOT%" && set BACKEND_PORT=%BACKEND_PORT% && npm run start:backend"
timeout /t 2 >nul
if not exist "%ROOT%src\frontend\main.tsx" goto :skip_vite
echo --- Starte Vite-Dev-Server (Port %WEBUI_PORT%) ---
start "RAG Vite-Frontend" cmd /k "cd /d "%ROOT%" && npm run dev -- --port %WEBUI_PORT%"
echo   React-UI: http://127.0.0.1:%WEBUI_PORT%
echo   API:      http://127.0.0.1:%BACKEND_PORT%
echo.
goto :skip_vite_done
:skip_vite
echo [INFO] src\frontend\main.tsx fehlt - Vite-Frontend wird NICHT gestartet.
:skip_vite_done
goto :after_node
:skip_node
echo [INFO] Kein vollstaendiges Node-Projekt gefunden (package.json / src/) - WebUI laeuft nur als Python/Gradio.
:after_node

"%CONDA%\Scripts\conda.exe" run -p "%ENV%" --no-capture-output python -s "%ROOT%webui_server.py"

pause
