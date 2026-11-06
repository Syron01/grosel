# experiments — park edilmiş kod

Ürünün çalıştırdığı tek yol `web_app/server.py` + üç motor: `fast_engine` (Studio),
`qwen_engine` (Atelier), `engine` (ortak kütüphane: stil ön ayarları, prompt derleyici,
sidecar). Buradakiler o yola bağlı değil; referans ve ölçüm geçmişi olarak duruyor.

| Ne | Neden burada |
|---|---|
| `lowvram_engine.py` | Sunucunun `BACKEND` zincirinde yok; yalnız `benchmarks/` çağırıyor. |
| `realtime_engine.py` | Yerini `fast_engine` + oturma planı aldı; ölçümü `runs.json`'da duruyor. |
| `run_*.py` (10) | Tek seferlik çekirdek denemeleri; her biri kendi prompt'unu gömüyor. |
| `sbc_crisp_kernel.py`, `pt_radiant_physics.py`, `cinematic_dof.py` | Kenar/yapay optik hileleri — kalite sözleşmesi "boru hattı kayıpsız olsun" diyor, filtre hilesi değil. |
| `test_callback.py`, `test_paris_fix.py`, `test_preview_gen.py` | pytest testleri değil, elle çalıştırılan betikler (test fonksiyonu içermiyorlar). |
| `ws_smoke.py` | 7860'taki sunucuya elle WebSocket isteği atan eskiz istemci. |

Çalıştırma: depo kökünden `python experiments/<ad>.py`. Silinmedi — geri almak için
olduğu yerden çıkarıp import'u eski adına çevirmek yeter.
