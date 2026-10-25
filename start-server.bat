@echo off
title RAG WebUI
setlocal

set "ROOT=%~dp0"
set "CONDA=%ROOT%miniforge"
set "ENV=%ROOT%env"
set "PYTHONNOUSERSITE=1"

:: ── llama.cpp / Modelle (muss in derselben BAT stehen wie der WebUI-Start) ──
set "LLAMA_EXE=G:\llama.cpp\llama-b9204-bin-win-cuda-13.1-x64\llama-server.exe"
set "MODELS_ROOT=C:\Users\bunny\.lmstudio\models"
set "MODELS_ROOT2=G:\ComfyUI\models\LLM"
set "CTX_SIZE=8192"
set "THREADS=8"
set "NGL=99"

if not exist "%ENV%\python.exe" (
    echo [ERROR] Environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo.
echo  +------------------------------------------+
echo  ^|      RAG WebUI -- Select Database        ^|
echo  +------------------------------------------+
echo.
echo  [1]  Main RAG   (chroma_zaebalo)
echo  [2]  Book RAG   (bookrag)
echo.
set /p CHOICE= Your choice [1/2]: 

if "%CHOICE%"=="2" (
    set "RAG_DB=%ROOT%bookrag"
    set "RAG_COLLECTION=books"
    echo.
    echo  [OK] Book RAG selected: bookrag
) else (
    set "RAG_DB=%ROOT%chroma_zaebalo"
    set "RAG_COLLECTION=zaebalo"
    echo.
    echo  [OK] Main RAG selected: chroma_zaebalo
)

echo.
echo  Address : http://127.0.0.1:7860
echo  KoboldCpp on port 5001 will be detected automatically.
echo.

"%CONDA%\Scripts\conda.exe" run -p "%ENV%" --no-capture-output python -s "%ROOT%webui_server.py"

pause
