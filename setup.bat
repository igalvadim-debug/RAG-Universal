@echo off
title RAG Setup
setlocal

set "ROOT=%~dp0"
set "CONDA=%ROOT%miniforge"
set "ENV=%ROOT%env"
set "INSTALLER=%ROOT%Miniforge3-installer.exe"

echo ============================================
echo  RAG Setup - Erstinstallation
echo  Benoetigt: ~5GB freier Speicher, Internet
echo ============================================
echo.

:: 1. Miniforge schon vorhanden?
if exist "%CONDA%\Scripts\conda.exe" (
    echo [OK] Miniforge gefunden: %CONDA%
    goto :create_env
)

echo [1/4] Lade Miniforge3 herunter...
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe' -OutFile '%INSTALLER%'"

if not exist "%INSTALLER%" (
    echo [FEHLER] Download fehlgeschlagen. Bitte Internetverbindung pruefen.
    pause
    exit /b 1
)

echo [2/4] Installiere Miniforge3 nach %CONDA%...
"%INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%CONDA%

if not exist "%CONDA%\Scripts\conda.exe" (
    echo [FEHLER] Miniforge-Installation fehlgeschlagen.
    pause
    exit /b 1
)
echo [OK] Miniforge installiert.
del "%INSTALLER%" 2>nul

:create_env
:: 2. Python-Umgebung schon vorhanden?
if exist "%ENV%\python.exe" (
    echo [OK] Python-Umgebung gefunden: %ENV%
    goto :install_packages
)

echo [3/4] Erstelle Python 3.11 Umgebung in %ENV%...
"%CONDA%\Scripts\conda.exe" create -p "%ENV%" python=3.11 -y --no-default-packages

if not exist "%ENV%\python.exe" (
    echo [FEHLER] Umgebung konnte nicht erstellt werden.
    pause
    exit /b 1
)
echo [OK] Python-Umgebung erstellt.

:install_packages
echo [4/4] Installiere Pakete...
echo.

:: Torch zuerst mit CUDA-Index
"%ENV%\Scripts\pip.exe" install torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [FEHLER] Torch-Installation fehlgeschlagen!
    pause
    exit /b 1
)

:: Dann Rest aus requirements.txt
"%ENV%\Scripts\pip.exe" install -r "%ROOT%requirements.txt"
if errorlevel 1 (
    echo [FEHLER] Paket-Installation fehlgeschlagen!
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Setup abgeschlossen!
echo  Starte das System mit: start.bat
echo ============================================
echo.
pause
