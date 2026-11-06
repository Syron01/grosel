"""GPF v0.1 — reference decoder + honest metrics.

    python verify.py IN.png OUT.gpf [--level 4]
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from pathlib import Path

import brotli
import numpy as np
from PIL import Image


def parse(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    assert raw[:4] == b"GPF\x01", "not a GPF file"
    hlen = struct.unpack_from("<I", raw, 6)[0]
    doff = struct.unpack_from("<I", raw, 10)[0]
    return json.loads(raw[14 : 14 + hlen]), raw[doff:]


def read_band(data: bytes, b: dict) -> np.ndarray:
    buf = data[b["off"] : b["off"] + b["len"]]
    if b["kind"] == "anchor":
        v = np.frombuffer(buf, dtype=">u2").reshape(b["h"], b["w"])
        r = (v >> 11) & 0x1F
        g = (v >> 5) & 0x3F
        bb = v & 0x1F
        return (np.stack([(r.astype(np.float32) * 255 + 16) / 32,
                           (g.astype(np.float32) * 255 + 32) / 64,
                           (bb.astype(np.float32) * 255 + 16) / 32], axis=-1))
    t = b["tile"]
    out = np.zeros((b["h"], b["w"], 3), dtype=np.float32)
    for i, (off, ln, tw, th) in enumerate(t["index"]):
        sub = np.frombuffer(brotli.decompress(buf[off : off + ln]), dtype=np.int8).reshape(th, tw, 3)
        y, x = (i // t["cols"]) * t["size"], (i % t["cols"]) * t["size"]
        out[y : y + th, x : x + tw] = sub
    return out * b["quant"]


def rebuild(path: Path, level: int | None = None) -> np.ndarray:
    header, data = parse(path)
    bands = header["bands"]
    level = len(bands) - 1 if level is None else level
    img = read_band(data, bands[0])
    for b in bands[1 : level + 1]:
        img = np.asarray(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((b["w"], b["h"]), Image.BILINEAR), dtype=np.float32)
        img = img + read_band(data, b)
    return np.clip(img, 0, 255)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 99.0 if mse <= 1e-9 else float(10 * np.log10(255.0**2 / mse))


def size_of(img: np.ndarray, fmt: str, **kw) -> int:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, fmt, **kw)
    return buf.tell()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("gpf")
    ap.add_argument("--level", type=int, default=None)
    a = ap.parse_args()

    src, gp = Path(a.src), Path(a.gpf)
    header, _ = parse(gp)
    orig = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32)
    rows = []
    for k in range(len(header["bands"])):
        rec = rebuild(gp, k)
        pay = sum(b["len"] for b in header["bands"][: k + 1])
        rows.append({
            "level": k, "w": rec.shape[1], "h": rec.shape[0],
            "gpfKB": round(pay / 1024, 1),
            "psnr": round(psnr(orig[: rec.shape[0], : rec.shape[1]], rec), 2),
            "webp82KB": round(size_of(orig[: rec.shape[0], : rec.shape[1]].astype(np.uint8), "WEBP", quality=82, method=6) / 1024, 1),
            "jpeg82KB": round(size_of(orig[: rec.shape[0], : rec.shape[1]].astype(np.uint8), "JPEG", quality=82) / 1024, 1),
        })
    full = rebuild(gp)
    print(json.dumps({
        "file": gp.name, "fileKB": round(gp.stat().st_size / 1024, 1),
        "sourceKB": round(src.stat().st_size / 1024, 1),
        "exactLosslessAtSource": bool(np.array_equal(np.rint(full), np.rint(orig))) if full.shape == orig.shape[:2] + (3,) else "shape-mismatch",
        "levels": rows,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
