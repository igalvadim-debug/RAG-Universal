@echo off
title RAG - Install Packages
setlocal

set "ROOT=%~dp0"
set "ENV=%ROOT%env"
set "PIP=%ENV%\Scripts\pip.exe"
set "PYTHONNOUSERSITE=1"

if not exist "%PIP%" (
    echo [FEHLER] Python-Umgebung nicht gefunden: %ENV%
    echo Bitte zuerst setup.bat ausfuehren!
    pause
    exit /b 1
)

echo ============================================
echo  RAG - Paket-Installation
echo ============================================
echo  Umgebung: %ENV%
echo.

echo [1/2] Installiere Torch mit CUDA 12.8 Support...
"%PIP%" install torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [FEHLER] Torch-Installation fehlgeschlagen!
    pause
    exit /b 1
)

echo.
echo [2/2] Installiere RAG-Pakete aus requirements.txt...
"%PIP%" install -r "%ROOT%requirements.txt"
if errorlevel 1 (
    echo [FEHLER] Paket-Installation fehlgeschlagen!
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Fertig! Starte mit start.bat
echo ============================================
pause
