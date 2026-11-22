"""Screenshots of the web build for the itch.io page.

    python tools/itch_shots.py [--out docs/itch/screenshots] [--only drag,reveal]

Runs tools/web_shot.py (headless Chrome, no window) on game/export/web, with drags computed
from the exported levels the way the game reads a drag (main.gd: drag, _toward): press the
site of one cell of a pair of the move, move toward the side it shares with its partner.
The picture's place repeats the landscape layout of game/scenes/main.gd (_layout) at
1280 x 720 with unit 1; check it again if the bar changes.

Shots (1280 x 720):
  drag    P04 scrambled, first solution move dragged halfway (pairs in motion, preview)
  hint    P06 scrambled, untouched for 7 s: the orange hint edge
  reveal  P02 solved by its one move: lines fading, then the scan half in with the name box
  levels  P01, P02, P03 solved in a row, then the level grid
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "game" / "data"
SIZE = (1280, 720)


def load(level_id: str) -> dict:
    return json.loads((DATA / "levels" / f"{level_id}.json").read_text())


def picture_rect(level: dict) -> tuple[float, float, float]:
    """Left, top and height of the picture in page pixels (landscape 1280 x 720, unit 1)."""
    cfg = json.loads((DATA / "config.json").read_text())
    m = float(cfg["picture"]["margin"])
    room_w, room_h = SIZE[0] - float(cfg["strip"]["width"]), SIZE[1]
    aspect = float(level["aspect"])
    w = min(room_w - 2 * m, (room_h - 2 * m) * aspect)
    h = w / aspect
    return round((room_w - w) / 2), round((room_h - h) / 2), h


def apply(level: dict, col: list[int], move: list[int]) -> list[int]:
    """Colours after the move (group, a, b): every pair of the group holding a and b swaps."""
    g, a, b = move
    out = list(col)
    for (u, v), gu in zip(level["edges"], level["group"], strict=True):
        if gu == g and {col[u], col[v]} == {a, b} and a != b:
            out[u], out[v] = col[v], col[u]
    return out


def labels(level: dict) -> list[list[int]] | None:
    """Raster of cell indices (rows of the 1280-wide label image)."""
    lab = level["labels"]
    raw = gzip.decompress((DATA / "levels" / lab["file"]).read_bytes())
    w = lab["width"]
    flat = struct.unpack(f"<{len(raw) // 2}H", raw)
    return [flat[r * w : (r + 1) * w] for r in range(lab["height"])]


def drag_for(level: dict, col: list[int], move: list[int], f: float, release: bool) -> list[str]:
    """Actions that drag the move a fraction f of the way (main.gd drag and _toward): press
    the site of a cell u of a pair, move along the direction of its shared side with v. The
    pair is the one with the longest travel whose press lands on u and picks v."""
    x0, y0, h = picture_rect(level)
    aspect = float(level["aspect"])
    ras = labels(level)
    nbr: dict[int, list[tuple[int, int]]] = {}
    for e, (u, v) in enumerate(level["edges"]):
        nbr.setdefault(u, []).append((v, e))
        nbr.setdefault(v, []).append((u, e))

    def px(pt) -> tuple[float, float]:
        return x0 + pt[0] * h, y0 + pt[1] * h

    def mid(e: int) -> tuple[float, float]:
        pts = level["sides"][e]
        xs, ys = pts[0::2], pts[1::2]
        return px((sum(xs) / len(xs), sum(ys) / len(ys)))

    def unit(dx: float, dy: float) -> tuple[float, float]:
        n = math.hypot(dx, dy) or 1.0
        return dx / n, dy / n

    def dot(p: tuple[float, float], q: tuple[float, float]) -> float:
        return p[0] * q[0] + p[1] * q[1]

    g, a, b = move
    best = None
    for e, ((u0, v0), gu) in enumerate(zip(level["edges"], level["group"], strict=True)):
        if gu != g or {col[u0], col[v0]} != {a, b}:
            continue
        for u, v in ((u0, v0), (v0, u0)):
            su, sv = px(level["sites"][u]), px(level["sites"][v])
            sx, sy = level["sites"][u]
            row = ras[min(int(sy * len(ras)), len(ras) - 1)]
            if row[min(int(sx / aspect * len(row)), len(row) - 1)] != u:
                continue
            m = mid(e)
            aim = unit(m[0] - su[0], m[1] - su[1])
            chosen = max(
                nbr[u], key=lambda we: dot(aim, unit(mid(we[1])[0] - su[0], mid(we[1])[1] - su[1]))
            )[0]
            if chosen != v:
                continue
            span = math.hypot(sv[0] - su[0], sv[1] - su[1])
            travel = max(span, 40.0)
            along = dot(aim, unit(sv[0] - su[0], sv[1] - su[1]))
            if along <= 0.3:
                continue
            if best is None or travel > best[0]:
                best = (travel, su, aim, along)
    if best is None:
        raise SystemExit(f"{level['id']}: no drag gives move {move}")
    travel, su, aim, along = best
    dist = (1.3 if release else f) * travel / along  # beyond the end when letting go
    acts = [f"press:{su[0]:.0f},{su[1]:.0f}"]
    for t in [12.0] + [dist * k / 8 for k in range(1, 9)]:
        acts += [f"drag:{su[0] + t * aim[0]:.0f},{su[1] + t * aim[1]:.0f}"]
    if release:
        acts += [f"release:{su[0] + dist * aim[0]:.0f},{su[1] + dist * aim[1]:.0f}"]
    return acts


def solve(level_id: str) -> list[str]:
    """Drags that play the level's solution from its scrambled start."""
    level = load(level_id)
    col, acts = list(level["start"]), []
    for move in level["solution"]:
        acts += drag_for(level, col, move, 1.0, True) + ["wait:0.6"]
        col = apply(level, col, move)
    if col != level["target"]:
        raise SystemExit(f"{level_id}: the solution does not recover the picture")
    return acts


def shots() -> dict[str, tuple[str, str, list[str]]]:
    """name -> (level, size, actions)."""
    p04 = load("P04")
    half = drag_for(p04, p04["start"], p04["solution"][0], 0.5, False)
    return {
        "drag": ("P04", "1280x720", half + ["wait:0.4", "shot:01_drag"]),
        "hint": ("P06", "1280x720", ["wait:7.5", "shot:02_hint"]),
        "reveal": (
            "P02", "1280x720",
            solve("P02")[:-1] + ["wait:1.0", "shot:03_lines", "wait:2.0", "shot:04_reveal"],
        ),
        "levels": (
            "P01", "1280x720",
            solve("P01") + ["key:ArrowDown", "wait:1"] + solve("P02")
            + ["key:ArrowDown", "wait:1"] + solve("P03")
            + ["key:ArrowDown", "wait:1", "click:1194,67", "wait:0.6", "shot:05_levels"],
        ),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "itch" / "screenshots"))
    ap.add_argument("--only", default="", help="comma-separated shot names")
    ap.add_argument("--wait", default="20", help="seconds for the game to load")
    a = ap.parse_args()
    todo = shots()
    names = a.only.split(",") if a.only else list(todo)
    for name in names:
        level, size, acts = todo[name]
        cmd = [sys.executable, str(ROOT / "tools" / "web_shot.py"), "--out", a.out]
        cmd += ["--level", level, "--size", size, "--wait", a.wait, "--do", *acts]
        print(name, "...", flush=True)
        subprocess.run(cmd, check=True)
        (Path(a.out) / "end.png").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
