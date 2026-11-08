@echo off
chcp 65001 >nul
title Grosel Studio
set PYTHONIOENCODING=utf-8

rem fast = FP8 resident path (fast, measured 4.2-4.9 GB peak)
rem Close other GPU apps first: the fast path needs most of the 8 GB card.
set HYPERFLUX_BACKEND=fast
set HYPERFLUX_PORT=7870

echo Starting Grosel Studio...
echo Open http://127.0.0.1:7870 in your browser.
echo.
python -m web_app.server
pause
