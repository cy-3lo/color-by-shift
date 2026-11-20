"""Page to pick paintings for the toy by clicking: index.html + cards.json (for an Artifact).

    python tools/pick_page.py FOLDER [FOLDER ...] [--out DIR] [--aspect 1.15 1.65]

Each FOLDER holds candidates.json (objectID, artistDisplayName, title, objectDate,
objectURL, optional source = museum label, default Met) and img/<objectID>.jpg. Only
pictures whose width / height lies in --aspect are kept (landscape, about 4:3). Every card
shows the original and the picture on the hex grid with several choices of n colours:
"present" (the most present colours) and "contrast" with decreasing minimum share of the
picture (toy/pictures.py contrast_palette), each with its smallest pairwise Delta E.
A click on a version picks the painting with these colours; the page stores the picks in
the artifact's database (collection "picks"), or in the browser when that is unavailable.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import sys
import tomllib
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))

import pictures  # noqa: E402
from mosaic_model.raster import fit_size, hex_pixel_map, render_rgb  # noqa: E402

BG = (28, 28, 34)
SHARES = [0.05, 0.02, 0.004]  # minimum share of the picture for a "contrast" colour


def data_uri(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, **({"quality": 78} if fmt == "JPEG" else {"optimize": True}))
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def main() -> None:
    cfg = tomllib.loads((ROOT / "toy" / "toy.toml").read_text())
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n", type=int, default=cfg["sim"]["colours"])
    ap.add_argument("--cells", type=int, default=8000)
    ap.add_argument("--tile", type=int, default=250)
    ap.add_argument("--aspect", type=float, nargs=2, default=[1.15, 1.65])
    a = ap.parse_args()
    sc, cp = cfg["sim"], cfg["palettes"]
    cards = []
    for folder in a.folders:
        for o in json.loads((folder / "candidates.json").read_text()):
            try:
                img = Image.open(folder / "img" / f"{o['objectID']}.jpg").convert("RGB")
            except OSError:
                continue
            if not a.aspect[0] <= img.size[0] / img.size[1] <= a.aspect[1]:
                continue
            g = pictures.grid_for(img.size[0] / img.size[1], a.cells)
            x = pictures.sample(img, g, sc["smooth"])
            pm = hex_pixel_map(g, fit_size(g, a.tile, a.tile))
            var = [("present", None, *pictures.kmeans(x, a.n, sc["seed"]))]
            for sh in SHARES:
                pal, lab, _ = pictures.contrast_palette(x, a.n, sc["seed"], cp["clusters"], sh)
                if not any((pal == v[2]).all() for v in var):  # skip identical colour sets
                    var.append((f"contrast {sh * 100:g} %", sh, pal, lab))
            th = img.copy()
            th.thumbnail((a.tile, a.tile))
            src = o.get("source", "Met")
            cards.append(
                {
                    "key": f"{src}_{o['objectID']}",
                    "source": src,
                    "id": o["objectID"],
                    "artist": o.get("artistDisplayName") or "unknown",
                    "title": o.get("title") or "",
                    "date": o.get("objectDate") or "",
                    "url": o.get("objectURL") or "",
                    "orig": data_uri(th, "JPEG"),
                    "versions": [
                        {
                            "name": name,
                            "share": sh,
                            "de": round(pictures.min_delta_e(pal)),
                            "colours": ["#" + bytes(int(v) for v in c).hex() for c in pal],
                            "img": data_uri(
                                Image.fromarray(render_rgb(pm, lab, pal, BG)).quantize(a.n + 1),
                                "PNG",
                            ),
                        }
                        for name, sh, pal, lab in var
                    ],
                }
            )
            print(len(cards), cards[-1]["artist"][:30], cards[-1]["title"][:50], flush=True)
    cards.sort(key=lambda c: (c["artist"], c["title"]))
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = a.out or ROOT / "runs" / "pickpage" / stamp
    out.mkdir(parents=True, exist_ok=True)
    body = "[\n" + ",\n".join(json.dumps(c) for c in cards) + "\n]\n"
    (out / "cards.json").write_text(body)
    (out / "index.html").write_text((ROOT / "tools" / "pick_page.html").read_text())
    print(f"{len(cards)} paintings -> {out} (cards.json {len(body) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
