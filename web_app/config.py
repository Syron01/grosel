import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _directory(env_name: str, default: Path) -> str:
    value = Path(os.environ.get(env_name, str(default))).expanduser()
    value.mkdir(parents=True, exist_ok=True)
    return str(value)


DATA_DIR = _directory("HYPERFLUX_DATA_DIR", PROJECT_DIR / "web_app" / "data")
GALLERY_DIR = _directory("HYPERFLUX_GALLERY_DIR", PROJECT_DIR / "Çizimler")
META_DIR = str(Path(GALLERY_DIR) / ".meta")
RUNS_FILE = str(Path(DATA_DIR) / "runs.json")
QUALITY_FILE = str(Path(DATA_DIR) / "quality.json")
FLUX_MODEL = os.environ.get("HYPERFLUX_FLUX_MODEL", "black-forest-labs/FLUX.2-klein-4B")

# Detay geçişi: yüksek kademelerde model küçük ızgarada değil, alan tavanına
# sığan büyük ızgarada doğrudan üretiyor. Adım süresi ve VRAM alanla büyüdüğü
# için tavan megapiksel ile konuyor (ölçüm: 2.2 MP @ 3.4 s/adım, 5.24 GB).
SR_MAX_SCALE = float(os.environ.get("HYPERFLUX_SR_MAX_SCALE", "2.0"))
SR_MAX_MP = float(os.environ.get("HYPERFLUX_SR_MAX_MP", "2.2"))

# Ürün sözü: hangi model bağlanırsa bağlansın tepe VRAM bu tavanı aşamaz.
# Tavan aşılırsa kaliteyi düşürmeden önce faz sürekliliği devreye girer;
# bu değer kalite sözleşmesinin plan değişmezlerinden biridir.
VRAM_CEILING_GB = float(os.environ.get("HYPERFLUX_VRAM_CEILING_GB", "5.0"))
