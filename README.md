# Grosel

Grosel, FLUX.2 Klein ve Qwen-Image 2.1 modellerini 8 GB'lık bir RTX 5060 Laptop kartında çalıştıran yerel bir görüntü üretim stüdyosudur. Ölçülen ürün sözü şudur: hiçbir fazda tepe VRAM 5.00 GB'ı aşmaz ve bu sınır `HYPERFLUX_VRAM_CEILING_GB` ile ayarlanır.

## Ne yapar

- Web arayüzünden prompt, stil, oran, kademe ve adım seçilir; üretim WebSocket üzerinden adım adım ilerler. İki çalışma yolu vardır: FLUX.2 Klein 4B (FP8, `fast` ve `classic`) ile Qwen-Image 2.1 (NF4 transformer, `qwen`).
- Her kare kalite sözleşmesinden geçer: diki z-skoru, yapısal detay indeksi, pütür indeksi, VAE örtüşmesi ve tepe VRAM.
- Kısıtlı Çıkarım Matematiği (KÇM) ölçülen koşulardan Pareto cephesi kurar ve bütçe ile süre kısıtı altında plan seçer.

## Kurulum

Python 3.11 veya üzeri gerekir.

```
pip install -e .
```

Model ağırlıkları repoda değildir ve kod `local_files_only=True` ile açılır; ağırlıklar önce indirilmelidir:

```
python -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.2-klein-4B')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen-Image-2.1')"
```

Qwen yolu yaklaşık 31.7 GB ağırlık ister. GPU'suz bir makinede yalnız `grosel plan` ve `grosel solve` çalışır.

## Çalıştırma (Windows)

```
start_studio.bat
start_resident_qwen.bat
```

- `start_studio.bat`: `HYPERFLUX_BACKEND=fast`, `HYPERFLUX_PORT=7870`, ardından `python -m web_app.server`.
- `start_resident_qwen.bat`: `HYPERFLUX_BACKEND=qwen`, `HYPERFLUX_QWEN_RESIDENT=1`, `HYPERFLUX_QWEN_BUDGET=5`, aynı port.

Elle başlatmak için:

```
set HYPERFLUX_BACKEND=fast
set HYPERFLUX_PORT=7870
python -m web_app.server
```

`HYPERFLUX_BACKEND` değerleri `fast`, `classic` ve `qwen`; başka bir değer sunucuyu açılışta durdurur. Port varsayılanı 7870'dir.

## Ölçülen değerler

Donanım: RTX 5060 Laptop, 8 GB. Sayılar PlanTreni.md ve web_app/resident_math.py içindeki ölçümlerden alınmıştır.

| Yol | Kare / adım | Süre | Tepe VRAM |
|---|---|---|---|
| FLUX fast | 1080p 16:9, native 1680x944, soğuk | 12.83 s | 4.92 GB |
| FLUX fast | 1080p 16:9, native 1680x944, sıcak | 11.08 s | 4.84 GB |
| Qwen yerleşik | 1280x704 / 20 | 59.3 s | 3.96 GB |
| Qwen yerleşik | 1024x576 / 20 | 35.1 s | 3.82 GB |
| Qwen yerleşik | 720p / 16, sıcak | 29.3 s | 3.91 GB |
| Qwen yerleşik | 1080p 2:3 / 40 | 132.3 s | 4.06 GB |

Dürüst sınırlar:

- 30 s hedefi yalnız 720p/16'da tutuyor; 1080p/40 Qwen yaklaşık 132 s sürüyor.
- 5.00 GB tavan kartın tamamına çok yakın; sürücü ve işletim sistemi payı sınırda kalıyor, 6 GB rahat.
- KÇM kalite ekseni CLIP ViT-B/32 prompt uyumudur; anatomi veya estetik ölçmez, LPIPS ve insan A/B karşılaştırması yapılmadı.

## API

Tümü web_app/server.py içinde tanımlıdır.

- `GET /api/info`: backend, VRAM tavanı, kademe merdiveni, ölçülmüş tepeler ve sözleşme eşikleri.
- `GET /api/system`: motor ve kart bilgisi; FLUX yollarında kademe, adım ve boyut listeleri.
- `GET /api/math`: KÇM anlık görüntüsü (planlar, Pareto cephesi, uyum katsayıları).
- `POST /api/math/solve`: bütçe, süre, native MP ve adım kısıtlarıyla plan seçer.
- `GET /api/resident-plan`: faz planı; `budget_gb`, `width`, `height`, `bits` ve `strict` sorgu parametreleri.
- `GET /api/gallery`: son 40 görselin yan dosya alanları ve kalite raporu.
- `WS /ws/generate`: üretim akışı; adım, durum, varyant ve tamamlanma iletileri.

## Komut satırı

`pip install -e .` sonrası `grosel` komutu kurulur; kurulum yapmadan `python -m web_app.cli` de kullanılabilir.

```
grosel plan --budget-gb 5 --width 1280 --height 704 [--json]
grosel solve --budget-gb 5 --max-time-s 8 --min-mp 0.45 --steps 4 [--cold] [--json]
grosel serve
```

`plan` model ağırlığı ve GPU istemez; faz planını, tepe VRAM'i ve 4/3/2-bit seçeneklerini verir. `solve` yerel `web_app/data/runs.json` telemetrisini okur; depo bu dosyayı içermediği için temiz bir kopyada `no-data` döner. `serve` stüdyoyu başlatır.

## Sınırlar

- Tek GPU, tek işçi; eşzamanlı üretim iddiası yok.
- Model ağırlıkları repoda değil, ayrıca indirilir. Qwen'in ilk yükü yaklaşık 100 s sürer.
- VAE tiling dikiş riski yüksek çözünürlükte sürüyor; 1536 testi temiz, 2K doğrulanmadı.
- NF4 aynı seed'de kompozisyonu FP8'den ayrıştırıyor; iki örnekte artefakt görülmedi ama LPIPS/insan A/B yok.
- Fast motor açılışta kartın çoğunu ister; Chrome, Steam veya Spotify açıkken warmup OOM verir.
- Enerji tüketimi ayrı ölçülmedi. İşlem süresi düştükçe enerji de düşer; ama bu bir türetim, ölçüm değil.

## Testler

```
python -m pytest -q
```

187 test geçiyor. `pyproject.toml` içindeki testpaths değeri `tests` olduğu için GPU betikleri koşuya girmez.

## Dizinler

- `web_app/`: sunucu, motorlar, matematik modülleri, şablonlar ve statik dosyalar.
- `tests/`: birim testleri.
- `benchmarks/`: ölçüm betikleri.
- `gpf/`: Grosel Pack Format deneyi (SPEC.md, encode ve verify).
