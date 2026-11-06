@echo off
title HyperFlux Studio Pro - Non-AI Slop Generator
color 0A
echo ========================================================
echo        HyperFlux Studio Pro - Web Inference Engine
echo      (Sifir AI Slop - Gercek Zamanli Telemetri ^& Canli Akis)
echo ========================================================
echo.
echo [-] Web Sunucusu ve GPU Motoru Baslatiliyor...
echo [*] Tarayicinizda acilacak adres: http://127.0.0.1:7860
echo.
rem fast   = FP8 resident motor (hizli, ~5.3 GB VRAM)
rem classic = BF16 + CPU offload (yavas, ~1.4 GB VRAM)
set HYPERFLUX_BACKEND=fast
python -m web_app.server
pause
