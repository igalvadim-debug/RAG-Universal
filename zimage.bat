@echo off
chcp 850 >nul
title Z-Image-Turbo Bildgenerator
setlocal

:: Python mit diffusers + torch auswaehlen (ComfyUi-venv)
set "PY="
if exist "D:\ComfyUiCuda\venv\Scripts\python.exe" set "PY=D:\ComfyUiCuda\venv\Scripts\python.exe"
if exist "C:\ComfyUi\venv\Scripts\python.exe" set "PY=C:\ComfyUi\venv\Scripts\python.exe"

if not defined PY (
    echo [FEHLER] Kein ComfyUi-venv gefunden
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Z-Image-Turbo Bildgenerator
echo ============================================
echo   Python : %PY%
echo   Prompt : %*
echo ============================================
echo.

"%PY%" "%~dp0zimage_generate.py" %*

echo.
echo ============================================
echo   Skript beendet (Fehlerlevel: %ERRORLEVEL%)
echo ============================================
echo.
pause
