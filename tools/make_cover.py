"""Cover and banner for the itch.io page, drawn from a level of the game.

    python tools/make_cover.py [--level P04] [--out docs/itch] [--banner P02,P04]

The picture is the level's foam (cells in their picture colours, thin dark lines between
cells) on the left, turning into the painting's scan on the right along a soft diagonal: the
reward of the game. The title "Color by Shift" and a tagline sit on a dark gradient, in
the orange of the game (--ink; --palette adds the level's colours as small squares).

Reads the exported game data (game/data/levels/<id>.json, <id>.labels.gz,
game/data/paintings/<id>.jpg) and the font assets/fonts/OpenSans.ttf (SIL OFL). Writes
<out>/cover_<id>.png (630 x 500, itch.io's recommended cover) and <out>/banner_<ids>.png
(1920 x 480, twice the 960 px width of an itch page banner: the title on the left, then
the paintings of --banner, each from cells to scan).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "game" / "data"
FONT = ROOT / "assets" / "fonts" / "OpenSans.ttf"
LINE = (16, 16, 20)
BACK = (28, 28, 34)
TEXT = (236, 233, 224)
TITLE = "Color by Shift"
TAGLINE = "Drag one cell: every like pair shifts."


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[k : k + 2], 16) for k in (0, 2, 4))


def foam_image(level: dict, labels: np.ndarray, which: str = "target") -> np.ndarray:
    """Cells in their colours (target or start) with lines 2 px wide between them."""
    pal = np.array([hex_rgb(c) for c in level["colours"]], dtype=np.uint8)
    col = np.asarray(level[which], dtype=np.int64)
    img = pal[col[labels]]
    edge = np.zeros(labels.shape, bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:, :-1] |= edge[:, 1:].copy()  # two pixels wide
    img[edge] = LINE
    return img


def load(level_id: str) -> tuple[dict, np.ndarray, np.ndarray]:
    level = json.loads((DATA / "levels" / f"{level_id}.json").read_text())
    lab = level["labels"]
    raw = gzip.decompress((DATA / "levels" / lab["file"]).read_bytes())
    labels = np.frombuffer(raw, "<u2").reshape(lab["height"], lab["width"]).astype(np.int64)
    scan = Image.open(DATA / "paintings" / f"{level_id}.jpg").convert("RGB")
    scan = np.asarray(scan.resize((lab["width"], lab["height"]), Image.LANCZOS))
    return level, labels, scan


def blend(foam: np.ndarray, scan: np.ndarray, mid: float, width: float = 0.22) -> np.ndarray:
    """Foam on the left, scan on the right, along a soft diagonal centred at x = mid."""
    h, w = foam.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    s = x / w + 0.25 * (y / h - 0.5)
    t = np.clip((s - mid) / width + 0.5, 0.0, 1.0)
    t = (t * t * (3 - 2 * t))[..., None]
    return (foam * (1 - t) + scan * t).astype(np.uint8)


def crop(img: np.ndarray, aspect: float, ax: float, ay: float) -> Image.Image:
    """Largest window of the given width/height ratio, placed by the anchors (0..1)."""
    h, w = img.shape[:2]
    cw, ch = (w, round(w / aspect)) if w / h < aspect else (round(h * aspect), h)
    x0, y0 = round((w - cw) * ax), round((h - ch) * ay)
    return Image.fromarray(img[y0 : y0 + ch, x0 : x0 + cw])


def shade(img: Image.Image, side: str, reach: float, strength: float = 0.82) -> Image.Image:
    """Darkens toward one side (top, bottom or left), so that the title reads."""
    w, h = img.size
    a = np.asarray(img).astype(float)
    if side == "bottom":
        t = np.clip((np.arange(h) / h - (1 - reach)) / reach, 0, 1)[:, None, None]
    elif side == "top":
        t = np.clip((reach - np.arange(h) / h) / reach, 0, 1)[:, None, None]
    else:
        t = np.clip((reach - np.arange(w) / w) / reach, 0, 1)[None, :, None]
    t = strength * t**1.3
    out = a * (1 - t) + np.array(BACK) * t
    return Image.fromarray(out.astype(np.uint8))


def font(size: int, weight: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT), size)
    axes = [ax["name"] for ax in f.get_variation_axes()]
    f.set_variation_by_axes([weight if n == b"Weight" else 100 for n in axes])
    return f


def fit(size: int, width: int) -> int:
    """The largest title size up to `size` whose title fits in `width` pixels."""
    while size > 10 and font(size, 800).getlength(TITLE) > width:
        size -= 2
    return size


def draw_title(
    img: Image.Image,
    level: dict | None,
    x: int,
    y: int,
    size: int,
    tagline: bool,
    ink: tuple[int, int, int] = TEXT,
    right: bool = False,
) -> None:
    """Title, tagline under it, and the level's colours as small squares (if a level). With
    right, x is the right edge and both lines are flush right."""
    d = ImageDraw.Draw(img)
    big = font(size, 800)
    small = font(round(size * 0.3), 600)
    ty = y + round(size * 1.22)
    if right:
        # a soft dark glow behind the letters only, so that the painting stays as it is
        mask = Image.new("L", img.size, 0)
        m = ImageDraw.Draw(mask)
        m.text((x, y), TITLE, font=big, fill=255, anchor="ra")
        if tagline:
            m.text((x - 3, ty), TAGLINE, font=small, fill=255, anchor="ra")
        glow = mask.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(size * 0.22))
        glow = glow.point(lambda v: min(255, round(v * 1.6 * 0.8)))
        img.paste(Image.new("RGB", img.size, BACK), (0, 0), glow)
        d.text((x, y), TITLE, font=big, fill=ink, anchor="ra")
        if tagline:
            d.text((x - 3, ty), TAGLINE, font=small, fill=ink, anchor="ra")
        return
    d.text((x, y), TITLE, font=big, fill=ink)
    if tagline:
        d.text((x + 3, ty), TAGLINE, font=small, fill=ink)
    if level is None:
        return
    side = round(size * 0.22)
    sy = ty + round(size * 0.48)
    for k, c in enumerate(level["colours"]):
        x0 = x + 4 + k * round(side * 1.35)
        box = (x0, sy, x0 + side, sy + side)
        d.rounded_rectangle(box, radius=side // 5, fill=hex_rgb(c), outline=LINE, width=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--level", default="P04")
    ap.add_argument("--out", default=str(ROOT / "docs" / "itch"))
    ap.add_argument("--anchor", type=float, default=0.5, help="horizontal place of the cover crop")
    ap.add_argument("--banner", default="P02,P04", help="paintings beside the banner's title")
    ap.add_argument("--mid", type=float, default=0.5, help="where foam turns into the scan")
    ap.add_argument("--ink", default="#ff8a1f", help="colour of the title and tagline")
    ap.add_argument("--palette", action="store_true", help="the level's colours under the title")
    ap.add_argument("--title-at", choices=["top-right", "bottom-left"], default="top-right")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    level, labels, scan = load(a.level)
    pic = blend(foam_image(level, labels), scan, a.mid)

    cover = crop(pic, 630 / 500, a.anchor, 0.5).resize((630, 500), Image.LANCZOS)
    ink = hex_rgb(a.ink)
    if a.title_at == "top-right":  # over the sky, flush right
        draw_title(cover, None, 630 - 28, 22, 60, tagline=True, ink=ink, right=True)
    else:
        cover = shade(cover, "bottom", 0.5, 0.9)
        squares = level if a.palette else None
        y = 500 - 34 - round(60 * (2.05 if a.palette else 1.62))
        draw_title(cover, squares, 30, y, 60, tagline=True, ink=ink)
    cover.save(out / f"cover_{a.level}.png")

    # banner: the title on the left, then whole paintings, each from cells to scan
    banner = Image.new("RGB", (1920, 480), BACK)
    x = 1920
    ids = a.banner.split(",")
    if len(ids) > 2:
        raise SystemExit("--banner: at most two paintings, the title needs the first third")
    for lid in reversed(ids):
        lv, lab, sc = load(lid)
        tile = crop(blend(foam_image(lv, lab), sc, 0.5, 0.35), 4 / 3, 0.5, 0.5)
        tile = tile.resize((640, 480), Image.LANCZOS)
        x -= 640
        banner.paste(tile, (x, 0))
    size = fit(110, x - 2 * 64)
    y = 480 // 2 - round(size * 0.95)
    draw_title(banner, None, 64, y, size, tagline=True, ink=hex_rgb(a.ink))
    name = f"banner_{a.banner.replace(',', '_')}.png"
    banner.save(out / name)
    print("wrote", out / f"cover_{a.level}.png", out / name)


if __name__ == "__main__":
    main()
