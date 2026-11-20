"""Contact sheet of the colour choices of paintings: original, then the picture on the hex
grid with each choice of n colours (present, contrast, accent) and its swatches.

    python tools/palette_sheet.py [--folder assets/paintings] [--out runs/palettes] [--n 5]

One PNG per page of `--rows` paintings; the smallest pairwise Delta E (CIE76) of each palette
is printed, and written to palettes.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))

import pictures  # noqa: E402
from mosaic_model.raster import fit_size, hex_pixel_map, render_rgb  # noqa: E402

BG = (28, 28, 34)


def variants(img: Image.Image, n: int, cells: int, smooth: float, seed: int, cp: dict):
    g = pictures.grid_for(img.size[0] / img.size[1], cells)
    x = pictures.sample(img, g, smooth)
    out = []
    for opt in pictures.PAINTING_OPTIONS:
        if opt == "present":
            pal, lab = pictures.kmeans(x, n, seed)
        else:
            pal, lab, _ = pictures.contrast_palette(x, n, seed, cp["clusters"], cp[f"{opt}_share"])
        out.append((opt, pal, lab))
    return g, out


def main() -> None:
    cfg = tomllib.loads((ROOT / "toy" / "toy.toml").read_text())
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", type=Path, default=ROOT / "assets" / "paintings")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "palettes")
    ap.add_argument("--n", type=int, default=cfg["sim"]["colours"])
    ap.add_argument("--cells", type=int, default=6000)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--tile", type=int, default=300, help="tile width and height, pixels")
    a = ap.parse_args()
    files = sorted(a.folder.glob("*.jpg"))
    a.out.mkdir(parents=True, exist_ok=True)
    T, pad, sw = a.tile, 10, 22
    ncol = 1 + len(pictures.PAINTING_OPTIONS)
    rows_csv = []
    for p0 in range(0, len(files), a.rows):
        page = files[p0 : p0 + a.rows]
        sheet = Image.new(
            "RGB", (ncol * (T + pad) + pad, len(page) * (T + sw + 26 + pad) + pad), BG
        )
        dr = ImageDraw.Draw(sheet)
        for k, f in enumerate(page):
            y = pad + k * (T + sw + 26 + pad)
            img = Image.open(f).convert("RGB")
            th = img.copy()
            th.thumbnail((T, T))
            sheet.paste(th, (pad + (T - th.width) // 2, y + 18 + (T - th.height) // 2))
            dr.text((pad, y), f"{p0 + k + 1}. {f.stem}", fill=(236, 233, 224))
            g, var = variants(
                img, a.n, a.cells, cfg["sim"]["smooth"], cfg["sim"]["seed"], cfg["palettes"]
            )
            pm = hex_pixel_map(g, fit_size(g, T, T))
            for j, (opt, pal, lab) in enumerate(var):
                x = pad + (j + 1) * (T + pad)
                rgb = Image.fromarray(render_rgb(pm, lab, pal, BG))
                sheet.paste(rgb, (x + (T - rgb.width) // 2, y + 18 + (T - rgb.height) // 2))
                de = pictures.min_delta_e(pal)
                dr.text((x, y), f"{opt}  dE {de:.0f}", fill=(150, 148, 160))
                for c in range(a.n):
                    dr.rectangle(
                        (x + c * (sw + 4), y + 22 + T, x + c * (sw + 4) + sw, y + 22 + T + sw),
                        fill=tuple(int(v) for v in pal[c]),
                    )
                share = np.bincount(lab, minlength=a.n) / len(lab)
                rows_csv.append([f.stem, opt, f"{de:.1f}", *[f"{s:.3f}" for s in share]])
        out = a.out / f"sheet_{p0 // a.rows + 1:02d}.png"
        sheet.save(out)
        print(out)
    with open(a.out / "palettes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["picture", "choice", "min_delta_e", *[f"share_{c}" for c in range(a.n)]])
        w.writerows(rows_csv)


if __name__ == "__main__":
    main()
