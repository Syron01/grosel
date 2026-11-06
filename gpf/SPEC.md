# GPF — Grosel Pack Format (v0.1 draft)

Bir görüntüyü **tek bir piksel sabiti** olarak değil, **çözünürlüğü alıcı tarafından seçilen bir yeniden-kurulum fonksiyonu** olarak tanımlayan container.

## 1. Tasarım tezi

Mevcut tüm formatlarda (PNG/JPEG/WebP/AVIF/JXL) dosyanın byte bütçesi **encode anında** kilitlenir.
16K master'ı 1080p'de izleyen bir tarayıcı, ya hazır türev (derivatif) dosya indirir ya da master'ı
indirir; ya da özel bir sunucu kırparak yeniden kodlar. Yani "bir görüntü = bir dosya = bir çözünürlük".

GPF bunu tersine çevirir:

> **Bir dosya, çözünürlük-parametreli bir yeniden-kurulum planıdır.
> Ödediğin byte, göstermek istediğin çözünürlükle orantılıdır; master çözünürlüğüyle değil.**

## 2. Katman modeli (Laplacian pyramid)

```
band[0]  anchor   : düşük çözünürlüklü taban (kayıplı, self-contained)
band[1..k] detail : band[i-1] 'in 2x büyütülmüş halinin, gerçek görüntüden farkı
meta           : band/offset/tile indeksi (JSON)
```

Yeniden kurulum:

```
R(0) = anchor
R(i) = upsample2x( R(i-1) ) + detail[i]          i = 1..k
```

`upsample2x` **formatın bir parçasıdır** (spec-maddesi, §5) — uygulamanın keyfi seçimi değil.
Bu yüzden `detail[i]` gerçek rezidualdir ve bit bütçesi verimli harcanır.

Bir GPF dosyası `0..k` hedef çözümlülüklerinden **herhangi birini** eksiksiz üretir:
tile-mapsiz tek-request'te tüm band'lar sırayla okunur; tile-mapsiz range-request'te
hedef band'a kadar olan prefix indirilir.

## 3. Range-elastic okuma

Header JSON'daki `bands[i].len` + `data_off` ile hedef band'ın **bitiş offset'i** hesaplanır:

```
GET /poster.gpf       Range: bytes=0-{data_off + Σ len(bands[0..i]) - 1}
```

Sunucu tarafı değişiklik gerekmez: statik dosya + standart HTTP Range.
Kırma (crop) ve region-of-interest, tile-indeksi eklendiğinde (§7) yine tek range ile çözülür.

## 4. Paketleme (packfile) — corpus boyutu

GPF ayrıca bir **içerik-adresli zincir** tanımlar (git packfile'in görsel karşılığı):

```
{ "parent": "<sha256-of-parent-gpf>", "tiles": [ ...sadeği değişen tile'lar... ] }
```

Aynı sahnenin yeniden-aydınlatılmış / inpaint edilmiş / upscale edilmiş türevleri, parent'ın
band'larını yeniden ödemez; sadece değişen tile'ları öder. **Farklı seed'li üretimler için
bu zincir işe yaramaz** — ölçüldü: aynı prompt'un seed varyantları arasında PSNR ≈ 8–10 dB,
yani pikseller tamamen yabancı; delta = görüntünün kendisi. Zincir, **aile içindeki türevlerde**
(quantization / step / CFG karşılaştırması, rötuş, re-light) çalışır: aynı dosyalarda ölçülen
base_40 ↔ fp8_40 PSNR ≈ 17 dB, base_40 ↔ 4bit_stream PSNR ≈ 20 dB.

## 5. Spec-maddesi: upsample2x

`bilinear` + `sharp-bilinear` karışımı, sabit tam-sayı ağırlıklarıyla:

```
out(2x+0.5, 2y+0.5) = ( 3*a + 5*b + 5*c + 3*d ) / 16      # a=TL b=TR c=BL d=BR
out(2x,   2y)       = a
out(2x+1, 2y)       = ( 5*a + 11*b + 1*c ) / 17 ... (tam tablo ek'te)
```

Decoder ve encoder **aynı** tabloyu kullanmak zorundadır; farklısa rezidual bozulur.
Amaç: GPU'dan bağımsız, 200 satır JS ile birebir üretilebilir olması.

## 6. Header

```
0   4B  "GPF\x01"
4   2B  version (u16 LE)
6   4B  header_len (u32 LE)
10  4B  data_off (u32 LE)
14      header_len bayt UTF-8 JSON
        { w, h, colorspace, bands:[{name,scale,w,h,codec,quant,len,off}], parent?, tiles? }
data_off band baytları (band sırasıyla, concat)
```

Band codec'leri (v0.1): `webp-lossy` (anchor), `brotli-delta-u8` (detail).

## 7. V1'de eklenecekler (tasarımı açan yerler)

- **Tile-indeksi**: band'ı 512×512 tile'lara böl, `tiles[]` offset tablosu → ROI ve
  viewport-sıralı yükleme. 16K'da bu olmazsa format işe yaramaz (133 MP decode = 400 MB RAM).
- **LoD-sıralı byte dizilimi**: `prefetch hint` ile görünür viewport'un tile'ları önce.
- **Banded quantization**: detayı bölge-duyarlı kuantala (lokal kontrast enerjisi ≈ bedava
  saliency); aynı total byte'ta algısal kalite artar.
- **Semantic band**: `prompt + seed + model-hash` göm, `detail`'ı sadece latent ile üret.
  Master'ı yeniden-infer etmeden, arşiv kalitesinde 50–100 KB. Kayıpsız değil, **eşdeğer**.

## 8. Bilinen sınırlar (dürüst)

- Entropi tabanı kalır: `Σ band` ≈ o çözünürlükteki tek bir kayıplı dosyanın boyu.
  GPF tek bir hedef çözünürlük için PNG'den dramatik küçük **değildir**; kazancı
  *çoklu hedef çözünürlüğü tek dosyada* ve *range-elastic* sunmasından gelir.
- Pyramid reziduali, kayıplı anchor yüzünden detayı "düzeltmek" için byte harcar;
  optimum değil, ama ölçülebilir.
- Browser desteği: kendi decoder'ı ister (WebCodecs Yok, canvas + JS yeterli).
