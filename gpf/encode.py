"""GPF v0.1 — encoder.

    python encode.py IN.png OUT.gpf [--levels 4] [--quant auto] [--tile 256]

Model: Laplacian pyramid. band[0] = anchor (raw RGB at 1/2^L), band[i] = tiled
quantized residual that lifts band[i-1] to the next resolution. The decoder is
the same function, so any prefix of the file yields a complete image at that level.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import brotli
import numpy as np
from PIL import Image

MAGIC = b"GPF\x01"
VERSION = 1


def down2(a: np.ndarray) -> np.ndarray:
    h, w, _ = a.shape
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR), dtype=np.float32)


def up2(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32)


def pick_quant(res: np.ndarray, cap: float = 126.0) -> int:
    q = 1
    while np.abs(res).max() / q > cap and q < 64:
        q *= 2
    return q


def pack_delta(res: np.ndarray, quant: int, tile: int) -> tuple[bytes, dict]:
    q = np.clip(np.rint(res / quant), -127, 127).astype(np.int8)
    h, w, _ = q.shape
    rows, cols = (h + tile - 1) // tile, (w + tile - 1) // tile
    index: list[list[int]] = []
    blob = bytearray()
    for ty in range(rows):
        for tx in range(cols):
            sub = q[ty * tile : min(ty + tile, h), tx * tile : min(tx + tile, w)]
            packed = brotli.compress(sub.tobytes(), quality=11)
            index.append([len(blob), len(packed), sub.shape[1], sub.shape[0]])
            blob += packed
    return bytes(blob), {"size": tile, "rows": rows, "cols": cols, "index": index}


def encode(src: Path, dst: Path, levels: int, tile: int, quant_mode: str) -> dict:
    img = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32)
    H, W, _ = img.shape

    pyr = [img]
    for _ in range(levels):
        pyr.append(down2(pyr[-1]))
    pyr = pyr[::-1]  # anchor first

    bands: list[dict] = []
    blobs: list[bytes] = []

    anchor = pyr[0].astype(np.uint8)
    a565 = (anchor.astype(np.uint16) >> np.array([3, 2, 3], dtype=np.uint16)).astype(np.uint16)
    packed565 = ((a565[..., 0] << 11) | (a565[..., 1] << 5) | a565[..., 2]).astype(">u2")
    bands.append(
        {"name": "anchor", "w": int(anchor.shape[1]), "h": int(anchor.shape[0]), "codec": "raw-rgb565",
         "kind": "anchor", "len": int(packed565.nbytes)}
    )
    blobs.append(packed565.tobytes())

    for i in range(1, len(pyr)):
        big, small = pyr[i], pyr[i - 1]
        res = big - up2(small, big.shape[:2])
        quant = pick_quant(res) if quant_mode == "auto" else int(quant_mode)
        blob, tinfo = pack_delta(res, quant, tile)
        bands.append(
            {"name": f"detail{i}", "w": int(big.shape[1]), "h": int(big.shape[0]), "codec": "br-delta-i8",
             "kind": "delta", "quant": quant, "clip": float(np.percentile(np.abs(res), 99.99) / quant),
             "len": len(blob), "tile": tinfo}
        )
        blobs.append(blob)

    off, body = 0, bytearray()
    for b, blob in zip(bands, blobs):
        b["off"] = off
        off += len(blob)
        body += blob

    header = {
        "w": W, "h": H, "colorspace": "srgb", "levels": levels, "tile": tile, "bands": bands,
        "targets": [{"level": k, "w": bands[k]["w"], "h": bands[k]["h"],
                     "bytes": sum(b["len"] for b in bands[: k + 1])} for k in range(len(bands))],
    }
    hj = json.dumps(header, separators=(",", ":")).encode()
    data_off = 14 + len(hj)
    dst.write_bytes(MAGIC + VERSION.to_bytes(2, "little") + len(hj).to_bytes(4, "little") + data_off.to_bytes(4, "little") + hj + bytes(body))
    return header


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--quant", default="auto")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    header = encode(src, dst, a.levels, a.tile, a.quant)
    print(json.dumps({
        "src": src.name, "srcKB": round(src.stat().st_size / 1024, 1),
        "gpfKB": round(dst.stat().st_size / 1024, 1),
        "targets": header["targets"],
        "quants": [b.get("quant") for b in header["bands"]],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
