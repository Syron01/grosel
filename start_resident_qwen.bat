@echo off
chcp 65001 >nul
title Grosel Studio - Qwen 4-bit resident
set PYTHONIOENCODING=utf-8

rem Quality path: NF4 transformer stays in VRAM, phases run one at a time.
set HYPERFLUX_BACKEND=qwen
set HYPERFLUX_QWEN_RESIDENT=1
set HYPERFLUX_QWEN_BUDGET=5
set HYPERFLUX_PORT=7870

echo Starting Grosel Studio (Qwen quality path)...
echo First load takes about 100 s; later starts are faster.
echo Open http://127.0.0.1:7870 in your browser.
echo.
python -m web_app.server
pause
