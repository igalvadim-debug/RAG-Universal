@echo off
title RAG + Qwen2.5 Coder + llama.cpp

echo ================================
echo   Активирую виртуальное окружение
echo ================================
call "D:\ComfyUiCuda\venv\Scripts\activate.bat"

echo.
echo ================================
echo   Запускаю Qwen2.5-Coder-7B-Instruct (GGUF)
echo ================================
set MODEL_PATH=D:\Unsloth\C-claude\models\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf

echo Модель: %MODEL_PATH%
echo.

start "" /MIN cmd /c ^
    llama-server.exe ^
    -m "%MODEL_PATH%" ^
    -ngl 99 ^
    -c 8192 ^
    --port 8080 ^
    --host 127.0.0.1 ^
    --threads 8

echo.
echo Жду 3 секунды, пока сервер поднимется...
timeout /t 3 >nul

echo.
echo ================================
echo   Запускаю RAG-клиент
echo ================================
python "C:\Users\Startklar\Desktop\zaebalo_mirror\rag_query.py"

echo.
echo ================================
echo   Готово
echo ================================
pause
