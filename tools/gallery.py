"""Gallery to choose paintings and their colours: one self-contained HTML page.

    python tools/gallery.py FOLDER [--out FILE.html] [--n 5]

FOLDER holds candidates.json (a list of Met objects: objectID, artistDisplayName, title,
objectDate, objectURL, optional source = museum label, default Met) and img/<objectID>.jpg
(small images). Each card shows the original
and the picture on the hex grid with each choice of n colours (toy/pictures.py); tick a
choice to pick the painting with these colours. The box at the top lists the picks as
"museum objectID choice | title" lines, to paste back.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
import tomllib
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))
sys.path.insert(0, str(ROOT / "tools"))

import pictures  # noqa: E402
from mosaic_model.raster import fit_size, hex_pixel_map, render_rgb  # noqa: E402
from palette_sheet import BG, variants  # noqa: E402

CSS = """
body{background:#1c1c22;color:#ece9e0;font:14px system-ui,sans-serif;margin:16px}
#picks{position:sticky;top:0;background:#26262e;padding:8px;z-index:2;border-radius:6px}
#picks textarea{width:100%;height:5em;background:#1c1c22;color:#ece9e0;border:1px solid #555}
.card{display:flex;gap:10px;align-items:flex-end;margin:22px 0;flex-wrap:wrap}
.card h3{flex-basis:100%;margin:0;font-weight:500;font-size:15px}
.card h3 a{color:#9690a0}
label{display:block;text-align:center;cursor:pointer;color:#9690a0}
label img{display:block;border:3px solid transparent;border-radius:4px}
input:checked+img{border-color:#f0b45a}
input{display:none}
.sw span{display:inline-block;width:18px;height:18px;margin:3px 1px 0;border-radius:3px}
"""
JS = """
function upd(){const t=[...document.querySelectorAll('input:checked')].map(e=>e.value);
document.querySelector('#picks textarea').value=t.join('\\n');
document.querySelector('#count').textContent=t.length;}
document.querySelectorAll('input').forEach(e=>e.addEventListener('click',ev=>{
 if(e.dataset.on==='1'){e.checked=false;e.dataset.on='0';}
 else{document.querySelectorAll('input[name="'+e.name+'"]').forEach(o=>o.dataset.on='0');e.dataset.on='1';}
 upd();}));
"""


def data_uri(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, **({"quality": 80} if fmt == "JPEG" else {"optimize": True}))
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def main() -> None:
    cfg = tomllib.loads((ROOT / "toy" / "toy.toml").read_text())
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n", type=int, default=cfg["sim"]["colours"])
    ap.add_argument("--cells", type=int, default=5000)
    ap.add_argument("--tile", type=int, default=250)
    a = ap.parse_args()
    cands = json.loads((a.folder / "candidates.json").read_text())
    cands.sort(key=lambda o: (o.get("artistDisplayName") or "~", o.get("title") or ""))
    T = a.tile
    cards = []
    for o in cands:
        f = a.folder / "img" / f"{o['objectID']}.jpg"
        try:
            img = Image.open(f).convert("RGB")
        except OSError:
            continue
        th = img.copy()
        th.thumbnail((T, T))
        head = f"{o.get('artistDisplayName') or 'unknown'}, {o.get('title', '')}"
        head += f" ({o.get('objectDate', '')})"
        src = o.get("source", "Met")
        card = f'<div class="card"><h3>{html.escape(head)} '
        card += f'<a href="{html.escape(o.get("objectURL") or "")}">{src} {o["objectID"]}</a></h3>'
        card += f'<img src="{data_uri(th, "JPEG")}">'
        sc = cfg["sim"]
        g, var = variants(img, a.n, a.cells, sc["smooth"], sc["seed"], cfg["palettes"])
        pm = hex_pixel_map(g, fit_size(g, T, T))
        for opt, pal, lab in var:
            im = Image.fromarray(render_rgb(pm, lab, pal, BG)).quantize(a.n + 1)
            sw = "".join(f'<span style="background:rgb({c[0]},{c[1]},{c[2]})"></span>' for c in pal)
            card += f'<label><input type="radio" name="p{src}{o["objectID"]}" '
            pick = f"{src} {o['objectID']} {opt} | {html.escape(head[:60])}"
            card += f'value="{pick}"><img src="{data_uri(im, "PNG")}">'
            card += f'<div class="sw">{sw}</div>{opt} (dE {pictures.min_delta_e(pal):.0f})</label>'
        cards.append(card + "</div>")
        print(len(cards), head[:70], flush=True)
    page = "<!doctype html><meta charset='utf-8'><title>Mosaic: choose paintings</title>"
    page += f"<style>{CSS}</style><div id='picks'>Picks (<span id='count'>0</span>), "
    page += "click a version to pick it, click again to drop it; paste this list back:"
    page += "<textarea readonly></textarea></div>"
    page += "\n".join(cards) + f"<script>{JS}</script>"
    out = a.out or a.folder / "gallery.html"
    out.write_text(page)
    print(f"{len(cards)} paintings -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
