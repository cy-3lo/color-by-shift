"""Cover and banner for the itch.io page.

    python tools/make_cover.py [--level P04] [--out docs/itch] [--cells 450]

One design for both: the painting as the game draws it (cells in their picture colours,
thin dark lines between cells) at the top right, turning into the scan toward the bottom
left. The title "Color by Shift" and a tagline are written flush right in one big cell at
the top right: the cells under the text merged into one, in the palette's lightest colour,
the text in its darkest colour (--ink, or ink in banner.toml, to choose another). On the
cover the scan shows on --share of the picture.

Cover: a level of the game, from the exported data (game/data/levels/<id>.json,
<id>.labels.gz, game/data/paintings/<id>.jpg), 630 x 500 (itch.io's recommended cover).
Banner: the painting of docs/itch/banner.toml (not a level), cut into blocks and a foam by
tools/make_painting_level.build (files in runs/cover/banner_foam), 1920 px wide at the
painting's own shape (twice the 960 px width of an itch page banner).
Font: assets/fonts/OpenSans.ttf (SIL OFL). Writes <out>/cover_<id>.png and
<out>/banner_<key>.png.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = ROOT / "game" / "data"
FONT = ROOT / "assets" / "fonts" / "OpenSans.ttf"
LINE = (16, 16, 20)
TITLE = "Color by Shift"
TAGLINE = "Drag one cell: every like pair shifts."
TAG = 0.42  # tagline size / title size


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[k : k + 2], 16) for k in (0, 2, 4))


def luminance(c) -> float:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def font(size: int, weight: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT), size)
    axes = [ax["name"] for ax in f.get_variation_axes()]
    f.set_variation_by_axes([weight if n == b"Weight" else 100 for n in axes])
    return f


def text_size(size: int) -> tuple[float, float]:
    """Width and height of the title and tagline block at title size `size`."""
    w = max(font(size, 800).getlength(TITLE), font(round(size * TAG), 650).getlength(TAGLINE))
    return w, size * 1.22 + size * TAG * 1.25


def level_picture(level_id: str) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Colours, target colour per cell, raster of cells and scan of a level of the game."""
    level = json.loads((DATA / "levels" / f"{level_id}.json").read_text())
    lab = level["labels"]
    raw = gzip.decompress((DATA / "levels" / lab["file"]).read_bytes())
    labels = np.frombuffer(raw, "<u2").reshape(lab["height"], lab["width"]).astype(np.int64)
    scan = Image.open(DATA / "paintings" / f"{level_id}.jpg").convert("RGB")
    scan = np.asarray(scan.resize((lab["width"], lab["height"]), Image.LANCZOS))
    colours = [hex_rgb(c) for c in level["colours"]]
    return colours, np.asarray(level["target"]), labels, scan


def banner_picture(width: int, cells: int, seed: int):
    """The same for the banner's painting (blocks and foam made here, raster `width` wide),
    and the painting's key."""
    import make_painting_level as mpl

    entry = tomllib.loads((ROOT / "docs" / "itch" / "banner.toml").read_text())["pictures"][0]
    cfg = tomllib.loads((ROOT / "toy" / "toy.toml").read_text())
    cfg["pictures"] = [entry]
    work = ROOT / "runs" / "cover" / "banner_foam"
    foam, _, pal, target, *_ = mpl.build(
        cfg, entry["key"], "banner", cells=cells, moves=1, seed=seed, grids=work, levels=work,
        smooth_width=width,
    )  # fmt: skip
    height = round(width / entry["aspect"])
    labels = foam.raster(width, height).astype(np.int64)
    scan = mpl.open_cut(entry, entry["aspect"]).resize((width, height), Image.LANCZOS)
    g = float(entry.get("lighten", 1.0))  # a gamma below 1 lightens the scan and the colours

    def light(x):
        return np.clip(255.0 * (np.asarray(x, float) / 255.0) ** g, 0, 255).round().astype(np.uint8)

    colours = [tuple(int(v) for v in light(c)) for c in pal]
    if entry.get("white"):  # the lightest colour shown whiter (cells and the title's cell)
        i = max(range(len(colours)), key=lambda k: luminance(colours[k]))
        colours[i] = hex_rgb(entry["white"])
    ink = hex_rgb(entry["ink"]) if entry.get("ink") else None
    shown = light(scan).astype(float)
    if entry.get("highlights") and entry.get("white"):
        # the scan's grey highlights drawn toward the white: weight rises from the 60th to
        # the 99th percentile of luminance, up to `highlights`, and falls with chroma
        lum = shown @ np.array([0.2126, 0.7152, 0.0722])
        lo, hi = np.percentile(lum, [60, 99])
        u = np.clip((lum - lo) / (hi - lo), 0, 1)
        chroma = shown.max(-1) - shown.min(-1)  # greys and whites only, not the tan floor
        grey = np.clip(1 - (chroma - 8) / 22, 0, 1)
        wgt = float(entry["highlights"]) * (u * u * (3 - 2 * u)) * grey
        shown += wgt[..., None] * (np.array(hex_rgb(entry["white"]), float) - shown)
    shown = shown.round().astype(np.uint8)
    return colours, np.asarray(target), labels, shown, entry["key"], ink


def compose(
    colours: list,
    target: np.ndarray,
    labels: np.ndarray,
    scan: np.ndarray,
    size: tuple[int, int],
    title_px: int,
    margin: int,
    ink: tuple[int, int, int] | None,
    anchor: float = 0.5,
    share: float = 0.35,
    soft: float = 0.35,
) -> Image.Image:
    """The picture cut to `size`, cells at the top right, the title in one big cell. The
    scan shows on about `share` of the picture (share <= 0.5), at the bottom left; `soft`
    is the width of the passage from cells to scan."""
    mid = 1.0 - (share / 2.0) ** 0.5  # s = mean of two uniforms: area(s > mid) = 2 (1 - mid)^2
    out_w, out_h = size
    h, w = labels.shape
    aspect = out_w / out_h
    cw, ch = (w, round(w / aspect)) if w / h < aspect else (round(h * aspect), h)
    x0, y0 = round((w - cw) * anchor), (h - ch) // 2
    lab = labels[y0 : y0 + ch, x0 : x0 + cw].copy()
    sc = scan[y0 : y0 + ch, x0 : x0 + cw].astype(float)
    k = cw / out_w  # raster pixels per output pixel
    # the text block, flush right at the top right; the cells it touches become one cell
    tw, th = text_size(title_px)
    pad = 0.35 * title_px
    box = (out_w - margin - tw - pad, margin - pad, out_w - margin + pad, margin + th + pad)
    bx0, by0, bx1, by1 = (max(0, round(v * k)) for v in box)
    big = np.isin(lab, np.unique(lab[by0:by1, bx0:bx1]))
    light = max(range(len(colours)), key=lambda i: luminance(colours[i]))
    dark = min(range(len(colours)), key=lambda i: luminance(colours[i]))
    pal = np.array(colours + [colours[light]], dtype=np.uint8)
    col = np.append(target, len(colours))
    lab[big] = len(target)
    img = pal[col[lab]].astype(float)
    edge = np.zeros(lab.shape, bool)
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    edge[:, :-1] |= edge[:, 1:].copy()  # lines about 2 px wide on the raster
    img[edge] = LINE
    # cells at the top right, scan at the bottom left
    yy, xx = np.mgrid[0:ch, 0:cw]
    s = 0.5 * ((1 - xx / cw) + yy / ch)
    t = np.clip((s - mid) / soft + 0.5, 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    t[big] = 0.0
    t = t[..., None]
    pic = Image.fromarray((img * (1 - t) + sc * t).astype(np.uint8))
    pic = pic.resize((out_w, out_h), Image.LANCZOS)
    d = ImageDraw.Draw(pic)
    ink = ink or colours[dark]
    x = out_w - margin
    d.text((x, margin), TITLE, font=font(title_px, 800), fill=ink, anchor="ra")
    small = font(round(title_px * TAG), 650)
    d.text((x - 2, margin + round(title_px * 1.22)), TAGLINE, font=small, fill=ink, anchor="ra")
    return pic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--level", default="P04", help="the level of the cover")
    ap.add_argument("--out", default=str(ROOT / "docs" / "itch"))
    ap.add_argument("--anchor", type=float, default=0.5, help="horizontal place of the cover crop")
    ap.add_argument("--share", type=float, default=0.2, help="share of the cover showing the scan")
    ap.add_argument("--ink", default="", help="colour of the text (default: darkest colour)")
    ap.add_argument("--cells", type=int, default=450, help="cells of the banner's foam")
    ap.add_argument("--seed", type=int, default=1, help="seed of the banner's foam")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ink = hex_rgb(a.ink) if a.ink else None

    cover = compose(*level_picture(a.level), (630, 500), 52, 26, ink, a.anchor, a.share, 0.2)
    cover.save(out / f"cover_{a.level}.png")

    *pic, key, banner_ink = banner_picture(1920, a.cells, a.seed)
    banner = compose(*pic, (1920, pic[2].shape[0]), 92, 44, ink or banner_ink)
    banner.save(out / f"banner_{key}.png")
    print("wrote", out / f"cover_{a.level}.png", out / f"banner_{key}.png")


if __name__ == "__main__":
    main()
