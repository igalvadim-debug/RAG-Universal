@echo off
chcp 850 >nul
title RAG WebUI
setlocal EnableExtensions

set "ROOT=%~dp0"
set "ENV=%ROOT%env"

rem Lokale Einstellungen dieses PCs (Modelle, llama-server, ComfyUI) - optional
if exist "%ROOT%env.bat" call "%ROOT%env.bat"

rem Portable Standardwerte (relativ zum Projektordner)
if not defined LLAMA_EXE if exist "%ROOT%llama\llama-server.exe" set "LLAMA_EXE=%ROOT%llama\llama-server.exe"
if not defined MODELS_ROOT set "MODELS_ROOT=%ROOT%models\LLM"
if not defined WEBUI_PORT set "WEBUI_PORT=7960"
if not defined WEBUI_HOST set "WEBUI_HOST=0.0.0.0"
if not defined CTX_SIZE set "CTX_SIZE=8192"
if not defined THREADS set "THREADS=8"
if not defined NGL set "NGL=99"
rem --mlock: 1 = передавать llama-server флаг --мlock (модель не свопится в своп),
rem 0 = не передавать. Некоторые сборки llama-server (в т.ч. D:\Llama.cpp\llama-server.exe)
rem не поддерживают --mlock и падают с "invalid argument: --mlock" — по умолчанию 0.
if not defined MLOCK set "MLOCK=0"

rem Erstinstallation automatisch, falls die Python-Umgebung fehlt
if not exist "%ENV%\python.exe" (
    echo [INFO] Python-Umgebung fehlt - starte setup.bat
    call "%ROOT%setup.bat"
    if not exist "%ENV%\python.exe" exit /b 1
)

set "PYTHONNOUSERSITE=1"
set "PATH=%ENV%;%ENV%\Library\bin;%ENV%\Scripts;%PATH%"

echo ================================
echo   RAG WebUI (Port %WEBUI_PORT%)
echo ================================
echo llama-server: %LLAMA_EXE%
echo Modelle:      %MODELS_ROOT%
echo.

rem Alte WebUI-Instanz auf dem Port beenden
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%WEBUI_PORT% " ^| findstr LISTENING') do taskkill /PID %%P /F >nul 2>&1

"%ENV%\python.exe" -s "%ROOT%webui.py"

echo.
echo WebUI beendet.
pause