@echo off
chcp 65001 >nul
title RAG Video Prompts

echo  RAG
echo.

cd /d "g:\RAG"

echo [1/3] Pruefe Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Node.js nicht gefunden. Installiere Node.js 18+.
    pause
    exit /b 1
)
echo       OK: %node --version%

echo [2/3] Pruefe Abhaengigkeiten...
if not exist "node_modules" (
    echo       Installiere npm Pakete...
    npm install
    if errorlevel 1 (
        echo FEHLER: npm install fehlgeschlagen.
        pause
        exit /b 1
    )
) else (
    echo       OK: node_modules vorhanden
)

echo [3/3] Starte Backend + Frontend...
echo.
echo --------------------------------------------------
echo  Frontend:  http://localhost:7860
echo  Backend:   http://localhost:7861 (API + WebSocket)
echo  ChromaDB:  g:\RAG\chroma_zaebalo
echo --------------------------------------------------
echo.

start "RAG Backend" cmd /k "npm run start:backend"
timeout /t 2 >nul
start "RAG Frontend" cmd /k "npm run dev"

echo Beide Server gestartet. Browser oeffnet sich automatisch...
timeout /t 3 >nul
start "" http://localhost:7860

echo.
echo Druecke eine Taste zum Beenden aller Fenster...
pause >nul

taskkill /f /im node.exe >nul 2>&1
echo Fertig.