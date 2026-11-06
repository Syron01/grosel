@echo off
title HyperFlux Studio - Qwen 4-bit Yerlesik
color 0A
echo ========================================================
echo    HyperFlux Studio - Qwen 4-bit Yerlesik (5 GB profil)
echo    Transformer tamamen VRAM'de; blok akisi yok.
echo ========================================================
echo.
echo [*] Tarayicida acilacak adres: http://127.0.0.1:7860
echo [*] Ilk acilis ~100 s (BF16 -> NF4 paketleme); sonraki acilislar hizli.
echo.
set HYPERFLUX_BACKEND=qwen
set HYPERFLUX_QWEN_RESIDENT=1
set HYPERFLUX_QWEN_BUDGET=5
set PYTHONIOENCODING=utf-8
python -m web_app.server
pause
