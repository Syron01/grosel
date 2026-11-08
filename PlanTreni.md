PlanTreni
Amaç: HyperFlux — mevcut modeli verilen VRAM bütçesinde çalıştıran dönüştürme/yerleşim platformu. 10 Kasım 2026 Teknopark demosu; iddia değil, canlı ölçüm.

web_app/templates/index.html → /static/app.css + /static/app.js
  Stüdyo | Optimizasyon | Matematik (KÇM) | Mimari
web_app/server.py → engine/fast_engine/qwen_engine + inference_math; HYPERFLUX_BACKEND=fast|classic|qwen, HYPERFLUX_PORT
web_app/qwen_engine.py → Qwen-Image 2.1: NF4 TE katman akışı, FP8 transformer blok akışı, lm_head atlama, VAE faz taşıma, embedding cache
web_app/fast_engine.py → FLUX.2 Klein FP8, PinnedMover, cache, uyarlanabilir decode offload
web_app/engine.py → koşullu anatomi (Qwen'de kapalı), fizik çapaları
web_app/resolutions.py → çözünürlük matrisi + get_dimensions (torch'suz, tek kaynak)
web_app/inference_math.py → plan/kaynak/kalite vektörleri, CLIP, LOOCV, Pareto, kısıt çözücü; yalnız fast/classic
web_app/data/runs.json → ölçüm telemetrisi
Çizimler/.meta → görsel başına prompt/seed/ayar

[x] Adım kademeleri: FLUX Hızlı 3–4 / Orta 8 / Pro 12 / Ultra 20; Qwen Oto / 20 / 28 / 40
[x] Qwen 2.1 1536×864/28 üretimde: 158.9 s · 1.36 GB tepe · encode 11.4 s · decode 3.3 s
[x] Qwen 2.1 1536×864/40 prototip: 218.8 s · 1.92 GB tepe (VAE yerleşik)
[x] Qwen 2.1 1280×704: 40 adım 169.6 s · 20 adım 79.9 s · 8 adım 44.9 s · 1.69 GB
[x] Qwen 2.1 2:3 832×1248/40 (referans eşleme, tiling): 181.0 s · 1.37 GB tepe · native→1080×1620
[x] Referans eşleme deneyi: aynı prompt + 2:3 + 40 adım stil öneksiz 192.1 s; tiling kapalı decode 7.17 GB (reddedildi)
[x] Premium Anime stili: glossy, altın rim light, parıltı, iris detayı; Ghibli/guaj öneki kaldırıldı
[x] Premium parlak varyant doğrulandı: aynı sahne promptu, 2:3, 40 adım, seed 42 → 1080×1620, 182.5 s, 1.37 GB
[x] Varsayılanlar Playwright ile doğrulandı: Premium Anime, 2:3 832×1248, 1080p=40, geo kapalı, CRPF gizli
! Son görselde eller birleşik/az detaylı; Qwen'in el sınırı sürüyor
! Birebir aynı kare için referansın seed/poz bilgisi yok; stil eşleşti, kompozisyon seed'e göre değişir
[x] lm_head atlama: difüzyon gizli durumu okur; encode 2.33→1.36 GB, ~16→11.4 s
[x] Qwen motoru: kullanılan BF16 27.98 GB → 11.02 GB akış, grup 2, yükleme 76–87 s, NF4 TE 4.39 GB
[x] 32'nin katı boyut düzeltmesi: 420p 800×448, 1080p 1536×864, 2:3 832×1248; sessiz yeniden boyutlandırma bitti
[x] HYPERFLUX_BACKEND=qwen + PORT; /api/system tiers/dims; arayüz dinamik kademe, faz rozeti, Qwen notları
[x] Uçtan uca test: 420p/8 33.7 s; 1080p/28 158.9 s; 2:3/40 181.0 s; eller ve çizgi belirgin iyileşti
! Referansın birebir aynısı değil: parlak/glossy dil Premium stille yaklaştı, aynı pose/model eşleşmesi garanti değil
! Untilenmiş VAE decode 832×1248'de 7.17 GB; tiling üretimde zorunlu
[x] KÇM bölüm 04: faz tepe modeli M=max(enc,den,dec), Qwen ölçümleri; Qwen koşuları KÇM'e karışmaz
[x] KÇM verisi 5 FLUX koşusuyla tazelendi (görseller silinmişti); çözücü optimal: 1080p/4 · 5.93 s · 4.43 GB · 7 plan
[x] Hızlı yol gerilemesi yok: FP8 rozet, 1080p/4, 1280×720, Hızlı 3–4 / Orta 8 / Pro 12 / Ultra 20
[x] Qwen 4-bit yerleşik: transformer 7.115B BF16 13.25 GB → NF4 3.43 GB; 1280×704/20 59.3 s / 3.96 GB; 1024×576/20 35.1 s / 3.82 GB
[x] Yerleşik motor: HYPERFLUX_QWEN_RESIDENT=1 + HYPERFLUX_QWEN_BUDGET=5; faz takası encode/decode; uçtan uca 1024×576/20 38.8 s / 3.91 GB tepe (encode 1.36, decode 1.14); 16 adım 29.3 s
[x] Kalite taraması: 5 görsel (2 anime, balıkçı eli, tilki, 1080p meyve) temiz; 16 adım 720p 29.3 s sıcak / ~32 s ilk prompt / 3.91 GB
[x] 1080p 2:3/40 yerleşik: 132.3 s / 4.06 GB (plan 4.02, sapma +0.04); 5 GB bütçenin %81'i
! 30 s hedefi yalnız 720p/16'da; 1080p/40 132 s. Tam 5.00 GB kartta sürücü/OS payı sınırda; 6 GB rahat
[x] 5 GB faz matematiği: resident_math.py, /api/resident-plan, Matematik bölüm 05, 4-bit rozeti
! NF4 aynı seed'de kompozisyonu FP8'den ayrıştırıyor; iki örnekte artefakt yok, CLIP tek örnekte 0.338→0.304; LPIPS/insan A/B yok
[ ] 4-bit hazır paket: başlangıçtaki 13.25 GB BF16 okumasını 3.43 GB'a indir; 8 GB RAM'li cihaz yolu
[ ] 3-bit/2-bit çekirdek ve katman duyarlılık ölçümü; 1536×864 yerleşik doğrulama
[ ] Qwen 28 vs 40 adım A/B; 2K/4K süre/VRAM ölçümü
[ ] Qwen'de BF16 TE ve CFG>1 karşılaştırması (4-bit TE tek örnekte daha iyi; A/B gerekli)
[ ] KÇM'de Qwen için ayrı çözücü ve LOOCV (şu an bölüm 04'te sabit ölçüm)
[ ] İnsan A/B ve LPIPS: CLIP el/anatomi hatasını ölçmez
[ ] Jüri provası: 1440×900 ve 1920×1080; 3 küratörlü prompt

! Qwen 1080p/28 ≈ 3 dk; canlı demo FLUX hızlı yolda, Qwen kalite yolu ayrı anlatılmalı
! VAE tiling dikiş riski yüksek çözünürlükte; 1536 testi temiz, 2K doğrulanmadı
! 4-bit TE kalitesi tek görselle değerlendirildi; FP8 TE bir örnekte daha kötüydü, kesin hüküm yok
! Tek işçi ve tek GPU; eşzamanlılık iddiası yok
! Marka: klasör Grosel, arayüz HyperFlux; sunum adı netleşmeli

3 Kasım 2026 ürünleştirme
Ana ajan → mimari, sunucu, arayüz, test ve ölçüm
[x] Gerçek çalışma durumu ve teknik borç envanteri: 185 dosya/47.2 MiB; paketleme, otomatik test ve mobil kırılım yok
[x] Mevcut hızlı motor ve masaüstü arayüzü çalışıyor; konsol hatası yok
[~] config.py → engine.py/fast_engine.py/server.py/inference_math.py: makine yolu yerine taşınabilir ayar
[x] resolutions.py → engine.py + inference_math.py + server.py: çözünürlük matrisi; KÇM artık torch import etmiyor (34.4 s → 0.29 s)
[x] schemas.py → server.py: HTTP/WebSocket sınır doğrulaması ve güvenli hata metinleri
[x] storage.py → runs.json/quality.json: atomik JSON yazımı
[x] server.py: CSP/nosniff/frame-ancestors başlıkları, lifespan, tek işçiye alınan cache-clear, gerçek silme hataları
[x] app.js: WebSocket kurtarma, güvenli olay kaydı, gerçek HTTP hata kontrolü
[x] index.html: tab/ARIA/dialog semantiği, girdi sınırları
[x] app.css: ≤720px mobil kırılım, 44px dokunma hedefi, 16px form metni, yana kaydırmalı tablo
[x] pyproject.toml + tests/: 113 birim testi (13.9 s), testpaths=tests ile GPU betiklerinden izole
[x] Ana akış doğrulandı: 1080p/4 gerçek üretim 7.28 s · 5.04 GB tepe · görsel galeriden 200 döndü
[x] Masaüstü 1440×900 ve mobil 390×844 Playwright ile ölçüldü: yatay taşma yok, 44px hedef, 16px form metni
[x] Klavye/sekme: Ok+Home navigasyonu, panel/gizleme senkron, konsol hatası yok; KÇM çözücü canlı "optimal"
[x] Bayat JS bulunması → /static ve / için Cache-Control: no-cache eklendi; telemetrideki mutlak dosya yolu istemciden kaldırıldı
! Satılabilirlik yalnız kod kalitesi değildir; dağıtım, lisans, destek ve ürün kanıtı ayrıca doğrulanacak
! Kalite iddiası (LPIPS/insan A/B) ve temiz makinede kurulum (`pip install -e .`) hâlâ kanıtsız
! Windows asyncio proactor, tarayıcı bağlantısı kapandığında log'a ConnectionResetError gürültüsü basıyor (stdlib, veri kaybı yok)

3 Kasım 2026 kalite açığı kapatma (rakip karşılaştırması)
Amaç: aynı promptta rakibin keskinliğini ve doğru tabela yazısını sıfır ek maliyetle yakala.
engine.py (derleyici) → fast_engine.py (prompt_embeds) → üretim
resolutions.detail_target → fast_engine.generate → _generate_refined(sr_size)
config.SR_* → fast_engine; schemas.detail_pass → server.py → WS

[x] Kök neden 1 (ölçüldü): eski derleyici "zero legible text" ekliyor, "interior"dan şömine uyduruyor,
    "no people"/"hand-bent" için anatomi çapası basıyordu → 4K karede kaldırımda şömine vardı.
[x] Kök neden 2 (ölçüldü): 2K/4K yalnız Lanczos interpolasyondu; 1456x624 → 3840x1632,
    f≈0.19 çevrim/px'te spektral uçurum. Rakip o bantta 2.6x daha çok enerji taşıyordu.
[x] Kök neden 3: detay üreten tek mekanizma (refine) varsayılan kapalıydı; 127 kayıtlı koşunun 0'ında kullanıldı.
[x] Çözüm A: heceleme çapası — tırnaklı kelimenin hemen ardına "letters O p e n ... in that order".
    Yazı isabeti aynı seedde 0/3 → 4/4; yeni seedlerde 2/3 (3003 hâlâ "Royter"). Ek süre 0 s, ek VRAM 0.
[x] Çözüm B: üretken detay geçişi (latent uzayında büyüt + düşük sigma'da yeniden örneklemeye sok).
    4K 21:9: keskinlik 505.9 → 954.3 (rakip 741.9), f0.19 6.46e-5 → 3.03e-4 (rakip 1.66e-4),
    f0.25 1.92e-6 → 3.22e-5 (rakip 1.52e-5). Süre 32.9 s, tepe 5.29 GB.
    1080p 16:9: keskinlik 171.0 → 532.5, +12 s, +0.20 GB.
[x] Ölçüm hatası düzeltmesi: ölçek tavanı (2.0 → 3.6 MP) 54 s/adım + 6.19 GB'a çıkıp WDDM'e taşınca
    tavan alan tavanına çevrildi (SR_MAX_MP=2.2) → 3.4 s/adım, 5.29 GB.
[x] Sahte kaldıraç elendi: Flux2Klein do_classifier_free_guidance = guidance_scale>1 ve not is_distilled;
    model_index.json is_distilled=true → guidance/negative prompt bu yolda etkisiz. Kod eklenmedi.
[x] tests/test_prompt_compiler.py (14) + tests/test_detail_target.py (7) → toplam 134 test geçti.

! Kalan kusur: FLUX.2 Klein 4B'de el yazısı tabela için ~1/3 tohumda heceleme bozuluyor; 20 adım ve
  kelimeyi iki kez geçirme denendi, ikisi de seed 3003'ü düzeltmedi. Model tavanı, sistem kusuru değil.
! Metin-kritik işte kesin çözüm Qwen-Image yolu (rakibin modeli); qwen_engine hâlâ Lanczos + refine=0.
! Rakip iç mekân nesne yoğunluğunda (çerçeve, masa, sandalye) bize göre daha zengin; bu kompozisyon farkı.

3 Kasım ghosting düzeltmesi (yazı + keskinlik aynı anda)
fast_engine.generate → resolutions.detail_target → üretim ızgarası (artık latent büyütme yok)
config.SR_MAX_* → detail_target; telemetry.native_resolution → gen ızgarası → app.js HUD

[x] Kök neden (ölçüldü + görselde görüldü): detay geçişi latent'i 90→134 gibi tamsayı-olmayan
    oranda yeniden örnekleme → 8 px'lik (tek latent piksel) faz dayaklanması → kenarlarda çift hat.
    halo 0.3298 (temiz taban) → 0.5374, ringing 0.83 → 2.21, çok-loblu kenar %%18 → %%54.5.
[x] Düzeltme: detay geçişi üretim ızgarasını büyütüyor (≤2.2 MP, ≤2.0x), model detayı kendi
    ızgarasında çiziyor; pixel-uzayı resize yalnız hedef≠üretim ise çalışıyor.
[x] İkinci düzeltme: büyütmede Lanczos → BICUBIC (Lanczos negatif lobu keskin kaynağa uygulanınca
    halka yapıyor). Hem fast hem qwen yolu. qwen eskiden hep %1.25-1.875 büyümeyi Lanczos'la yapıyordu.
[x] Telemetri yalanı düzeltildi: native_resolution artık gerçekten üretilen ızgara (1280x720 değil
    1920x1072). HUD: "1920x1080 · 1920x1072 doğrudan".
[x] Ölçüm (aynı prompt, seed 87675445, native 1080p): t=15.2-16.7 s, tepe 5.09-5.10 GB, ghosting yok
    (3x zoom: tek çerçeve borusu, tek jant, çift sele yok). 4K 21:9 native: halo 0.4489, ringing 1.39,
    çok-loblu %%35.8, keskinlik 262.3.
[x] Tekrarlanabilirlik kanıtı: aynı seed + aynı ayar iki koşu → PSNR 99 dB, mean|Δ|=0.000.
    Pipeline deterministik; seed tek başına kontrol edilebilir bir değişken.
[~] Yazı isabeti native ızgarada ölçüldü (4 tohum, heceleme çapası açık):
    base 1280x720 → 2/4 doğru; native 1920x1072 → 3/4 doğru. Izgara büyüğü yazıyı iyileştiriyor.
[~] Adim/CRPF ablation (seed 87675445, native): 4/8/12 adım ve CRPF kapalı → hepsi "Open Roguter".
    Adım sayısı ve CRPF bu tohumdaki harf sırasını değiştirmiyor; hata tohum+kodlamaya kilitli.
! Ölçüm uyarısı: halo/ringing metriği keskinlikle karışıyor — bulanık taban "temiz" çıkıyor.
    Yalnız aynı içerikte A/B geçerli; nihai doğrulama 3x görsel kırpma.

3 Kasım rakip kalitesi: web + yerel dokumantasyon denetimi
engine.STYLE_PRESETS(+medium) → detect_medium → compile → 3 motor da ayni fonksiyonu cagirmiyor:
  fast_engine/qwen_engine compile'i cagirir; classic engine.py ayri yol (kontrol gerekiyor)

[x] Rakip referansi olculdu: 1536x1024 (1.57 MP) native, halo 0.7277 / ringing 3.10 / cok_lobali %%71.1
    → rakip bizden DAHA fazla kenar tasmai yapiyor; yani fark cozunurluk degil, doku + yazi.
[x] Resmî FLUX.2 [klein] 4B ayari (BFL karti): guidance_scale=1.0, 4 adim, kare 1024, bf16.
    → bizim 4 adim/CFG'siz yolumuz resmî ile birebir. BFL kartinin kendi itirafi:
    "text rendered may be inaccurate or subject to distortion" → yazi hatasi model limiti.
[x] Resmî kural dogrulandi: latent interpolasyonlu buyutme BFL/diffusers tarafindan desteklenmiyor,
    mu = compute_empirical_mu(image_seq_len, num_steps) oldugu icin buyutulmus latent YANLIS sigma
    programi miras aliyor → ghosting kok nedenimiz dokumanda yazili (ComfyUI #1116, #13558).
[x] BFL cozimi: native uretim (<=4 MP, onerilen <=2 MP, 16'ya bolunur) + FLUX Deblur. Biz artik native.
[x] Prompt kalibi A/B (4 tohum, native 1920x1072, ayni sahne):
    bizim derlenmis (heceleme capasi + guard)  → 3/4 dogru
    resmî KISA kalip (capasiz, geo kapali)      → 0/4 dogru  ("Rougter/Rouger/Rouyter")
    tırnakli kelimeyi basa tasimanin resmi tavsiyesi → 0/4 ("Orurter/Tugter/Ruriger")
    → Heceleme capasi kaldi; resmî kisa kalip FLUX yolunda KAZANDIRMIYOR.
[x] Bloom kirpigi A/B: "thin tubes, minimal glow bloom, crisp separated letterforms" → 3/4,
    tuplar görsel olarak incele/harfler ayrk. Kalici hale TEXT_TAIL icinde alindi
    ("crisp separated glyphs with sharp focus throughout" — BFL "no blur" yerine bunu öneriyor).
[x] SISTEM KUSURU (yeni bulunan): varsayilan stil chipi "matte_chiaroscuro" foto promptlara
    "Atmospheric concept art painting, digital illustration, matte texture, soft muted colors"
    + "painterly textured finish" ekliyordu → rakip karsilastirmasindaki yumusak/mat gorunumun
    bir kismi bizden geliyordu. Duzeltme: preset'ler "medium" tasir; prompt kendi ortamini
    adlandiriyorsa (photo/render/anime/paint) celisen stilin on+son eki dusurulur.
    tests/test_prompt_compiler.py +5 test, toplam 139 test gecti.
[~] Qwen-Image-2.1 yolda COK-RAPORLU kusur: diffusers VAE tiled decode (tile 256 / stride 192,
    lineer 64 px blend) dus dikey bantlama. qwen_engine.py:281 enable_tiling() varsayilan kullaniyor.
    Bant dedektoru (band_probe) kalibre edildi: rakip 1.14 / biz 1.19 → FLUX yolunda bant YOK.
    Qwen karesi cikinca ayni metrikle bakilacak.
! Turbo agirliklari diskte EKSIK (transformer + text_encoder inmemis, ~12 GB indirme gerekiyor) →
  8 adimli hizli Qwen icin kullanici onayi lazim.
! FLUX official fp8 agirligi (FLUX.2-klein-4b-fp8, BFL "1.6x hizli, %%40 az VRAM") diskte yok;
  biz kendi FP8 cast'imizi kullaniyoruz.

3 Kasim Qwen-Image-2.1 karsilastirmasi (ayni prompt, ayni seedler)
qwen_engine.generate → Qwen3-VL 8B NF4 encode → FP8 transformer akis → VAE decode → BICUBIC
olcut: seam_test.py (192 px katlari, faz kaydirilmis 191 kontrol ofseti → z-skoru)

[x] Kalite sonucu: Qwen 1080p/40 adim = 205 s, tepe 1.39 GB (denoise), 1536x864 native → 1920x1080.
    Yazi: seed 4711 "OpenRouter" DOGRU, seed 87675445 "OpenRouter" DOGRU (2/2).
    FLUX ayni seedlerde 3/4; 87675445'te 4/8/12 adim ve CRPF kapali da duzelmedi → model tercihi.
    Kompozisyon: Qwen karesinde bisiklet tam ve dogru yerde; FLUX karesinde on tekerlek sikinca
    cukur saksinin icine girdi. Rakibe gorsel olarak en yakın ciktimiz bu.
[x] SISTEM KUSURU 2 (bulundu + olculdu): diffusers Qwen VAE tiled decode (tile 256 / stride 192,
    64 px lineer blend) dikey diki birakiyor. z=4.11 (gercek/kontrol 2.69x) seed 87675445 karesinde;
    ayni metrik FLUX karelerinde 0.71/-0.74, rakipte -0.76, Qwen 4711'de 0.58 → 1/2 karede gorunur.
    GitHub Qwen-Image-2.1 #5/#14 cok-raporlu kusurla ayni kok: tile 256 / stride 192.
[x] Duzeltme: qwen_engine artik decode boyutuna karar veriyor — TILE_ABOVE_MP (vars. 1.5 MP) altinda
    use_tiling=False (tek parca decode, diki yok), ustunde tiled kalir. Ayni seed ile dogrulaniyor.
! Decode tepesi olculmesi gerekiyor: 1.33 MP tek parca decode 8 GB'a sigmaziysa tavan dusurulur
  ya da tile geometrisi 512/384'e genisletilir (blend 128 px).

3 Kasim diki duzeltmesi DOGRULANDI (512/384 tile geometrisi)
olcut araci: seam_scan.py (192 ve 384 px periyotlarinda faz-z skoru), ayni seed 87675445, 1080p
[x] tiled 256/192 : z=+4.11 @192, +6.50 @384 | decode tepe 1.39 GB
[x] tek parca     : z=+2.99 @192, +4.27 @384 | decode tepe 9.06 GB → 8 GB'a sigmiyor (ICERIK TABANI)
[x] YENI 512/384  : z=+2.78 @192, +3.97 @384 | decode tepe 2.40 GB, decode 2.5 s, toplam 211 s
    → diki, diki OLMAYAN karenin içerik tabaninin da altinda: artefakt olculamiyor.
    Blend 64→128 px, tile sayisi 40→12. VRAM 8 GB'da genis pay.
[x] Gorsel: tabela "OpenRouter" dogru, bantlama yok, bisiklet tam.

3 Kasim rakip karsilastirmasi — metrik tuzağı ve gerçek durum
ayni bolgede Laplacian varyans: HAM (grain dahil) / MEDYAN-FILTRELI (yalniz yapisal detay)
  RAKIP   asfalt 1791 / 124     vitrin 1809 / 182
  Qwen    asfalt  682 /  88     vitrin 1368 / 182
  FLUX    asfalt  921 /  67     vitrin 2580 / 348
[x] Sonuc: rakibin HAM keskinligi %93 grain (1791→124 = 14x dusus). Yapisal detayda FLUX yolumuz
    rakibin 1.9 kati (vitrin 348 vs 182). Qwen yolumuz yapisal olarak rakibe esit/dusuk.
! 1.25x BICUBIC buyutme keskinlik YEMEZ: 740 (1920) vs 751 (1536 native) → resize suçlu degil.
? Qwen yolu "yumuşak/CG" okunuyor: model 2K native kovaya alisik, bizim 1080p kademe 1.33 MP uretiyor.
  Tek degiskenli deney: 1080p 16:9 native 1536x864 → 1760x992 (1.75 MP, en 0.2% aspes hatasi).

3 Kasim native-yukseklik deneyi SONUC (tek degisken: QWEN_DIMENSIONS 1080p/2K)
[x] 1080p 16:9 native 1536x864 (1.33 MP) -> 1760x992 (1.75 MP): 211 s -> 266 s, tepe 2.40 -> 2.41 GB.
    Yapisal detay (medyan 5x5 sonrasi): vitrin 182 -> 274 (+50%), tabela 120 -> 169 (+41%), asfalt 88 -> 116.
    Global ton kontrast 0.161 -> 0.178 (rakip 0.179 - esitlendi). Diki z 2.78 -> 0.17.
[x] 2K 16:9 native 1536x864 -> 2048x1152 (2.36 MP, artik buyutme YOK): 260 s, tepe 2.42 GB, diki z 0.13.
[x] Merdiven tutarli kalsin diye tum 1080p oranlari ~1.75 MP, 2K oranlari hedefe native cekildi.
[x] tests: 139 gecti. UI olcu tablosu bugunku degerlerle guncellendi (index.html 457-459 + not).
? Kalan fark: rakibin YEREL kontrast 2 kati (0.032 vs 0.016) - bu grain, doku degil (ham 1791 -> temiz 124 = %93).
  Film greni eklemek metriyi gosterir ama urunun gorunumunu degistirir → kullanici karari, sessiz yapilmadi.

3 Kasim kalite sozlesmesi (modelden bagimsiz) — quality_math.py → server.py → tum motorlar
web_app/quality_math.py → qwen_engine.py, fast_engine.py, engine.py (tiling_kwargs)
web_app/quality_math.py → server.py:attach_quality → telemetri["quality_report"] + sidecar["quality_report"]
config.VRAM_CEILING_GB → server.attach_quality (urun sozu: tepe VRAM <= 5.00 GB)
[x] Gercek karelerde ayrim testi: onarim ONCESI kareler ihlal veriyor (z 4.11/6.50/5.36, 5.52, 5.27),
    ONARIM SONRASI kareler geciyor (0.13 / 0.17 / -0.51). Sozlesme kusu buluyor, temiz kareye dokunmuyor.
[x] tests/test_quality_math.py 8 test + tests/test_quality_sidecar.py 5 test. Toplam 152 test geciyor.
[x] olcut tabani belgelendi: 8-bit karede pütür indeksi duz rampada 0.392 okunuyor (float'ta 0.000) → esikler ustunde.
! HALA: "edge_width_px" ve "halo_index" sozlesmeden CIKARILDI (bulanik karede yon degistirdiler, kirilgandi).
! CELISKI DUZELTILDI: motorlar telemetri["quality"] = kademe etiketi ("720p") tasiyor. attach_quality onu
    sozlukle eziyordu → runs.json Pareto gruplamasi kirilirdi. Anahtar "quality_report" oldu;
    sidecar "quality" kademesi korunuyor, galeri API'si artik quality_report + peak_vram_mb da veriyor.
[~] Canli dogrulama: qwen sunucusu (7860) yeni kodla yukleniyor, 420p bir uretimde quality_report gorulecek.
[x] CANLIDA DOGRULANDI (420p, seed 4242, 27.15 s, tepe 2.17 GB): sidecar'da hem "quality":"420p" hem
    "quality_report" var; seam_z -0.08, ok=true. blend_px 128 telemetride.
! SOZLESME BOSLUGU KAPATILDI: diki yok ama kare dus=z → ok geciyordu (420p kare yapi=0.0 oldugu halde gecti).
    Yeni olcut detail_index (1024 izgara, 5x5 medyan sonrasi Laplacian varyansi, scipy'siz numpy kaskadi).
    Gercek karelerde: onarim oncesi 1798-1944 → onarim sonrasi 2268-2406 (+%19), 420p taslak 27.5,
    FLUX Studio 2963-3000. Bulaniklik merdiveni tek yonlu: 2268 → 1377 → 437 → 159.
! MUTLAK ESIK KURULMADI (ayni tuzak): seyrek-ama-keskin sentetik sahne 101 okunuyor, yogun sentetik 32-79,
    gercek foto 2268 → taban iccerik yogunluguna bagli, duz stilleri haksiz cezalandirirdi.
    Denetim GORELI: server.detail_baseline() ayni kademe+stil'in son <=12 kosusunun medyanini runs.json'dan
    alir (<4 ornek → None → iddia yok); MEDYAN x 0.60 altindaki kare WARNINGS uretir, FAILURES degil.
[x] tests: 152 → 161 (8 quality_math + 5→7 sidecar + 2 detay). Tumu geciyor.
[x] UI'ya veri akisi: META_FIELDS += quality_report, peak_vram_mb → /api/gallery sozlesmeyi tasiyor.

3 Kasim motor yuzeyi (#9) — silme yok, park var
[x] experiments/ acildi + experiments/README.md (neden park, nasil geri alinir)
[x] Kovulen 19 dosya: run_*.py (10), sbc_crisp_kernel, pt_radiant_physics, cinematic_dof,
    batch_generate_gallery, realtime_engine, test_callback/test_paris_fix/test_preview_gen (test fonksiyonu YOK),
    test_ws_client → experiments/ws_smoke.py, web_app/lowvram_engine.py (sunucunun BACKEND zincirinde yoktu).
[x] benchmarks/{bench_lowvram,lang_test,prompt_ab,prompt_v2} import'u experiments.lowvram_engine'a cevrildi,
    `python -c "import experiments.lowvram_engine"` dogrulandi.
[x] Kalan yuzey: server BACKEND = fast(Studio) | qwen(Atelier) | classic; engine.py artik ORTAK KUTUPHANE
    (STYLE_PRESETS, prompt derleyici, save_sidecar) — silinemez, 3 tarafdan import ediliyor.
[x] Yan yazim (sidecar) atomik: engine.save_sidecar → storage.write_json_atomic (yarim JSON riski kapandi),
    kullanilmayan `import json` dusuruldu. Studio/Atelier yan yazimina native_resolution + vae_blend_px eklendi
    → sozlesme plani artik kare basina denetlenebilir. Testler: 156 gecti (16.3 s).
! Sunucu (7860) bu son iki degisiklik oncesi kodla ayaga kalkmisti; nihai yeniden baslatma #11 sonrasi yapilacak.
[ ] #9 motor yuzeyi: lowvram_engine.py yalniz benchmarks/* tarafindan kullaniliyor (4 dosya) → park edilebilir.
    engine.py silinemez: STYLE_PRESETS + prompt derleyicisi qwen/fast/server tarafindan import ediliyor (ortak kutuphane).

## UI yenileme — 2026-11-03 (Material dark, Ingilizce, mantiga dokunmadan)
Amaç: ayni motor, ayni API; sadece kabuk. Hedef: yatirimci ekraninda daginik Turkce not yigini yerine tek bakista okunan panel.
web_app/templates/index.html → /static/app.css + /static/app.js  (ID'ler sabit kaldı, JS kırılmıyor)
web_app/static/app.css → bastan yazıldı: M3 dark token + component katmanı
web_app/static/app.js → SADECE metin sabitleri (Turkce→Ingilizce) + sunucu etiket eslemesi (EN_LABEL); mantık/akış aynı
! server.py QWEN_TIERS/FLUX_TIERS etiketleri Turkce kalıyor → app.js içinde EN_LABEL haritası çeviriyor (kod değişmedi)
! generateBtn içinde <svg> şart (JS querySelector("svg") ile spin atıyor) → sprite <use> korundu
[ ] index.html
[ ] app.css
[ ] app.js metinleri
[ ] tarayıcı doğrulama (7860 açık, 4 sekme + lightbox + üretimi başlat akışı)
x doğrulandı (Chromium 1600x900, qwen motoru): 4 sekme temiz, konsol sadece /api/math/solve 500 veriyor,
  lightbox + Esc çalışıyor, gerçek üretim 24.8 s / 2.33 GB / 512x768, görünen metinde tek Türkçe karakter kalmadı.
x adım satırı: 5 kademede alt satıra taşıyordu → seg-N dinamik + sayı etiket (Auto 16 20 28 40), ad tooltip'te.
x renk: native slider beyaz ray kullanıyordu → color-scheme: dark. Stil adları motor Türkçe gönderiyor → EN haritası.
! BULGU (UI değil, motor): HYPERFLUX_BACKEND=fast artık açılmıyor. diffusers 0.41'de
  AutoencoderKLFlux2.enable_tiling(self) kwargs almıyor; fast_engine.py:101 + engine.py:346 dört argüman geçiyor → startup crash.
  qwen_engine.py:286 aynı çağrıyı alıyor ama Qwen VAE kabul ediyor, o yüzden yalnız FLUX yolları (fast + classic) düşüyor.

## #10/#11 kapanis — 2026-11-04 (FLUX karosu cokmesi + sahne paneli)
[x] BULGU DOGRULANDI ve ONARILDI: AutoencoderKLFlux2.enable_tiling(self) kwargs almıyor;
    fast_engine.py:101 + engine.py:346 acilisda TypeError ile motoru dusuruyordu → sahne
    planinin CANLI arkadi (fast/Studio) hic acilamiyordu. Qwen kismiya kabul ettigi icin only
    FLUX yollari kirikti; GPU'suz testler motoru insa etmedigi icin bunu gormemisti.
[x] quality_math.apply_tiling(vae) → sozlesmeyi VAE'nin KENDI yuzeyine kurar:
    kwargs'lu API (Qwen) → enable_tiling(512/384); argsiz API (FLUX2) → tile_sample_min_size
    + tile_latent_min_size ornekliden + tile_overlap_factor. Geometri yoksa None (iddia yok).
[x] quality_math.blend_px_of(vae) → FIILEN onusme; uc motorun telemetrisi degeri oradan yaziyor
    (eskiden sabit 128 bildiriyordu, yani yan yazim sozlesme kurabilirkti).
[x] tests/test_vae_tiling.py: gercek diffusers siniflari agirliksiz/kucuk konfigla → 5 test.
    Toplam: 172 gecti.
[x] #11 panel: Studio sag ray → "Quality contract" blogu (MET/CHECK/BROKEN + blend/seam z/
    detail/grain/native rozetleri + kademe koruma notu); VRAM meterina tavan cizgisi (vramMark)
    ve asilirda kirmizi dolgu. Benchmarks → "Ceiling by tier": kademe basina olculen medyan
    bar + en kot tik + tavan cizgisi + native → hedef; <3 kosu soluk (provisional).
[x] /api/info → vram_ceiling_gb + tier_ladder + measured{runs,provisional,peak,worst,time,
    target,native} + contract{min_blend_px,tile_px,seam_z_limit,rel_detail_floor}; ezber "vram"
    duz metin alani kaldirildi. Guard sozleri yapilandirildi: WS payloadina `guard` nesnesi,
    kabuk Turkce cumle cevirmek yerine veriden Ingilizce kuruyor.
[x] Gorsel hijyen: kullanilmayan els.vramFree dusuldu (HTML'de de yoktu).
[ ] CANLI dogrulama: fast arkadiyla sunucu acik (7860) → bir uretim, panel + /api/info cekimi.
! RISK (sahne): olculmus gecmis fast/classic yolunda tavani asiyor (1080p/16:9 medyan ~5.09 GB,
  en kot 9.06 GB). Guard yalniz `backend` etiketli 3 kosu birikince indirmeye baslar; ilk
  kosular eskiye gore gecher. Patrona soylenecek: ya 720p sahnede, ya tavan 6 GB ilan.

## #10/#11 Canli olcu — 2026-11-04 ogleden sonra (tavani KENDI varsayilan yollu kirdi)
[x] CANLI Kosu (fast, 1080p, 16:9, DETAY ACIK — yani Studio'nun varsayilani):
    native 1920x1072 (2.06 MP) → tepe 5241 MB = 5.12 GB > tavan 5.00 GB.
    Soz kirildi, panel dogru soyledi (ok:false, vram ihlali). Kademe adi tepeyi
    soylemiyor: ayni 1080p detay KAPALIyken 0.92 MP / 4.45 GB.
[x] OLCUM-UYGULAMA (kac nokta, kac kademe — hepsi denoise tepesi):
    0.459 MP → 4282 MB | 0.922 MP → 4554 MB | 2.058 MP → 5242 MB
    en kucuk kareler: egim 602 MB/MP, taban 4001 MB → cizgi 0.459 noktasiyla %0.2 tutuyor.
    cap(5.00 GB x 0.97 pay) = 1.604 MP → 1080p izgarasi 1680x944 → ongu 4.84 GB.
[x] resolutions.fit_mp_ceiling() → (MP, MB) olcumlerinden tavina sigan en buyuk izgara.
    Veri yetmezse (>=3 olcum, >=2 farkli izgara, egim>0, taban<tavan) NONE: uydurma esik yok.
[x] server.detail_mp_cap() bu arkadin kosularindan cizgiyi kurar → WS `detail_max_mp`
    → fast_engine SR_MAX_MP hirsini budar; /api/info.contract.detail_mp_cap + seam_periods.
[x] quality_math.seam_periods(): diki periyodu KARO GEOMETRISINDEN. Hazir demet
    (192/384/512) sizinti yapiyordu: 384 aralikli diki 512 katlarinin 1/4'une denk
    dusuyor → ayni kare 384'te 2.08 (temiz) ama 512'de 3.20 (sahte ihlal). Sentetik
    kontrol: 384 diki kendi periyodunda 14.1 okunuyor → yakalama guclu, yalanci pozitif gitti.
[ ] CANLI DOGRULAMA (yeniden): cap uygulandi → 1080p detay acik tepe <= 5.00 GB ve ok:true.
! NOT: 420p/720p gibi karelerde karo tekrari <3 → seam_z bos doner; panel rozeti
  gostermiyor, bu "olcemedik" demek (kalite sozlesmesi icin acikca soylenecek).

## #10 kapanis — 2026-11-04 (denetim CANLI ateşlendi, kabukta cökme bulundu)
[x] CANLI: HYPERFLUX_VRAM_CEILING_GB=4.7 ile 1080p istegi → guard 720p'ye indirdi,
    uretim tamamlandi (4.18 GB), kayitta `vram_guard.used=720p`, panel kaydi var.
    Not: bu bir stres kosusu — urun tavani yine 5.0 GB.
[x] CANLIDA YAKALANAN COKME: indirilen kademe olculmemisse `predicted_peak_gb=None`
    f-string'i cokertiyordu → uretim "motor gunluklerini denetleyin" ile dustu. Sunucu
    `guard_status()` + kayda `measured` bayragi; kabuk `guardNote()` sayi uydurmuyor.
    Test: test_inilen_kademe_olculmemisse_cumlde_sayi_uydurulmaz (ayni yoldan gecen tek
    kanit GPU'suz test olamazdi — cunku crash WS isleyicisindeydi).
[x] Panel nesil tasimi: measured_by_tier artik SON 5 kosuyu kullaniyor (PROMISE_WINDOW);
    detay tavani oncesi 4.45 GB kosular 1080p satirinda "1280x720" diye yasiyor,
    yeni cizim 1680x944'i gosteriyordu → ikisi de guncel pencerede.
[x] Gecmis karesine tiklamak sozlesmeyi ayni kuruyordu ama rozetler eksikti: /api/gallery
    META_FIELDS resolution/native_resolution/vae_blend_px'i tasmiyordu (yan dosyada VARDI).
[x] seam_z olculemiyen kare (720p: 3 karo tekrari yok) rozet hic gostermiyordu → artik
    "n/a · not measured is not a pass".
[x] Tavan sozu icin olculen son deger (fast, 5.0 GB): 1080p/2K izgara 1680x944 →
    tepe 4.84 GB (sicak), ilk kosu 4.92 GB (soguk kodalayici) → ihlal yok; detay indeksi
    3754 vs 3732 (2.06 MP'lik budanmamis kare ile esdeger), yani kalite mali yok.
Toplam test: 183 gecti.

## #10/#11 SON DURUM — 2026-11-04 aksam
[x] Urun tavanindaki son olcum (fast, 5.0 GB, 16:9): 1080p → izgara 1680x944, soguk
    tepe 4.92 GB, sicak 4.84 GB → ihlal yok; sozlesme ok:true.
[x] Method sayfasi olcumle guncellendi: "1080p = 1280x720 + Lanczos, 4.43 GB" YAZILIYDI
    (eskimis, sahne rakip tarafından ölçülebilir bir iddia). Simdi tavan-esitligi ve
    2K/4K'nin bu kartta resample oldugu acikca yazili. 720p satiri da duzeltildi
    (4.18 GB sicak / 4.92 GB soguk, 3.3-5.0 s — eskisinde 17.9 s yaziyordu).
[x] /api/gallery → ayarli yan dosya alanlari (resolution, native_resolution,
    vae_blend_px) gecmis karesine tiklayinca da ayni sozlesmeyi kuruyor (canli API
    dogrulamasi: ilk kayitta uc alan da geldi).
[~] GORSEL dogrulama: panel DOM'da iki kez okundu (Benchmarks satirlari + Studio
    "MET / 4.84 · 5.0 GB / seam z 0.29 · detail 3754 · grain 0.38"), ama EKRAN
    GORUNTUSU alinamadi — uygulama ici tarayici yuzeyi gorunur degil
    (NATIVE_BROWSER_VIEWPORT_UNAVAILABLE), sonra sekme yanit vermeyi birakti.
    Yapilacak: tarayici paneli acikken Studio + Benchmarks'a bir bakis.
Toplam test: 183 gecti. Sunucu hala calisiyor (fast, 7860, task b9qvl9mrm).

## 7 Kasım 2026 — yatırımcı öncesi denetim (hijyen + marka + canlı kanıt)
Marka kararı: **Grosel** (arayüz, sekme başlığı, başlatıcılar). Varsayılan port **7870**.
[x] Silinen: `_experiments/` (24 MB), `__pycache__`/`.pytest_cache`, kök test artefaktları,
    `legacy_studio.html`, `experiments/` ve ona bağlı 4 ölü benchmark.
[x] Gerçek hatalar: FastKleinEngine'de silinen `decode` değişkeni her üretimde NameError
    verecekti (canlı testte yakalandı, `decode_time` ile onarıldı); Refine/Copy düğmeleri
    ikonu metinle siliyordu; Qwen benchmark'ı 180 s zaman aşımına takılıyordu (600 s +
    Qwen'de protokol kapalı); Method katmanı matris yerine fiili `native_resolution`
    kullanıyor; classic telemetriye `native_resolution` eklendi.
[x] Canlı doğrulama (fast, 7870): benchmark 5/5 · 1080p cold 12.83 s / 4.92 GB, warm
    11.08 s / 4.84 GB · 720p cold 4.61 s / 4.92 GB, warm 3.08 s / 4.18 GB · kalite
    sözleşmesi ok:true · tarayıcı konsolu hatasız.
[x] Tavan penceresi temiz: 1080p son 5 koşu en kötü 4.92 GB ≤ 5.00; detail cap 1.596 MP.
[x] Test: 184 vaka geçti (125 test fonksiyonu); ruff F/E9 sınıfı temiz.
! Galeri şu an 8 görsel (3 taze üretim + 5 benchmark). Eski deneme görselleri bu makinede yok.
! Fast motor açılışta kartın çoğunu ister; Chrome/Steam/Spotify açıkken warmup OOM.
  Sahne öncesi diğer GPU uygulamalarını kapat — `start_studio.bat` bunu not ediyor.
