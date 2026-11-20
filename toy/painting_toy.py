"""Painting toy: pick a picture and some reactions, watch the mosaic dissolve, then play it
backwards and save the reversed movie as a GIF.

    .venv/bin/python toy/painting_toy.py [--picture great_wave] [--rules voter]

The panel has no text apart from the picture's title:
  < title >        picture (Left / Right)
  < colours >      the five colours of the picture, dark to light; the arrows change the
                   choice of colours (most present, most contrasted, ...) (key C)
  < . . . >        rule preset (Up / Down); its reactions are listed below and are yours
                   to edit, live, all with k = 1:
                     click a block          next colour (right click: previous)
                     click an arrow         next shape: cell, edge, vertex (clockwise
                                            only, curved arrow), vertex (either hand)
                     right click an arrow   remove the reaction
                     +                      add a reaction
                   An edge is drawn as two neighbouring cells, a vertex as the three cells
                   around a corner (top, lower right, lower left = clockwise).
  play / pause     (Space) every picture starts paused
  fast-forward     (F) on / off
  backward         (B) plays the run backwards down to the picture; pressed again, jumps
                   to the picture. Play then starts again from the picture.
  save             (S) the run backwards as a GIF in gifs/ (path printed in the terminal)
Esc quits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import threading
import time
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))

from mosaic_model import Reaction, ReactionSet, Sim  # noqa: E402
from mosaic_model.observables import pace_frames  # noqa: E402
from mosaic_model.raster import fit_size, hex_pixel_map, render_rgb  # noqa: E402
from pictures import PRESETS, load_picture, options, preset  # noqa: E402

BG = (28, 28, 34)
PANEL = (38, 38, 46)
FG = (236, 233, 224)
MID = (150, 148, 160)
BTN = (58, 58, 70)
BTN_ON = (96, 92, 120)
ACCENT = (240, 180, 90)
ARITY = {"cell": 1, "edge": 2, "vertex": 3}
# an arrow click cycles through these (shape, chiral)
KINDS = [("cell", True), ("edge", True), ("vertex", True), ("vertex", False)]
SQRT3 = math.sqrt(3.0)
GAP = 1.08  # spacing of neighbouring blocks (1 = touching)


def load_cfg() -> dict:
    return tomllib.loads((ROOT / "toy" / "toy.toml").read_text())


def write_gif(path: Path, pm: np.ndarray, frames: list[np.ndarray], pal: np.ndarray, g: dict):
    """GIF of the frames (cell colour arrays) drawn with the pixel map pm; exact palette."""
    from PIL import Image

    n = len(pal)
    flat = [int(v) for c in pal for v in c] + list(BG)
    imgs = []
    for f in frames:
        idx = np.where(pm >= 0, np.asarray(f)[np.maximum(pm, 0)], n).astype(np.uint8)
        im = Image.fromarray(idx, mode="P")
        im.putpalette(flat + [0] * (768 - len(flat)))
        imgs.append(im)
    durs = [g["frame_ms"]] * len(imgs)
    durs[0] += g["hold_start_ms"]
    durs[-1] += g["hold_end_ms"]
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durs, loop=0)


def hex_points(c, rad: float) -> list[tuple[float, float]]:
    """Pointy-top hexagon around c."""
    return [
        (
            c[0] + rad * math.cos(math.radians(60 * k - 90)),
            c[1] + rad * math.sin(math.radians(60 * k - 90)),
        )
        for k in range(6)
    ]


def block_offsets(shape: str, rb: float) -> list[tuple[float, float]]:
    """Block centres of one side of a reaction, relative to the top-left corner of its box
    (box_size): blocks sit like cells on the grid. A vertex is three hexes around their
    shared corner, clockwise 0 (top), 1 (lower right), 2 (lower left)."""
    w = SQRT3 * rb * GAP
    bw, bh = box_size(rb)
    x0 = SQRT3 * rb / 2
    if shape == "vertex":
        return [(x0 + w / 2, rb), (x0 + w, rb + 1.5 * rb * GAP), (x0, rb + 1.5 * rb * GAP)]
    if shape == "edge":
        return [(x0, bh / 2), (x0 + w, bh / 2)]
    return [(bw / 2, bh / 2)]


def box_size(rb: float) -> tuple[float, float]:
    return SQRT3 * rb * (1 + GAP), 2 * rb + 1.5 * rb * GAP


class Toy:
    def __init__(self, cfg: dict, picture: str | None = None, rules: str | None = None):
        import pygame

        self.pg = pygame
        self.cfg = cfg
        w, h = cfg["window"]["width"], cfg["window"]["height"]
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Mosaic")
        self.font = pygame.font.Font(None, 26)
        self.canvas = pygame.Rect(0, 0, cfg["window"]["canvas"], h)
        self.px = self.canvas.right + 18
        self.n = cfg["sim"]["colours"]
        keys = [p["key"] for p in cfg["pictures"]]
        self.pic_i = keys.index(picture) if picture else 0
        self.opt_i = 0
        self.preset_i = PRESETS.index(rules) if rules else 0
        self.fast = False
        self.buttons: dict[str, object] = {}
        self.blocks: dict[tuple[int, str, int], tuple[float, float]] = {}
        self.rb = 12.0
        self.saving: threading.Thread | None = None
        self.saved_at = 0.0
        self.cache: dict = {}
        self.load_preset()
        self.load()

    # ---- picture and rules -------------------------------------------------------------------
    def entry(self) -> dict:
        return self.cfg["pictures"][self.pic_i]

    def options(self) -> list[str]:
        return options(self.entry(), self.cfg["palettes"])

    def load(self) -> None:
        sc = self.cfg["sim"]
        entry = self.entry()
        opt = self.options()[self.opt_i]
        key = (entry["key"], opt)
        if key not in self.cache:
            self.cache[key] = load_picture(
                entry, self.n, sc["cells"], sc["smooth"], sc["seed"], self.cfg["palettes"], opt
            )
        self.pic = self.cache[key]
        g = self.pic.graph
        size = fit_size(g, self.canvas.w - 24, self.canvas.h - 24)
        self.pixmap = hex_pixel_map(g, size)
        ph, pw = self.pixmap.shape
        self.offset = ((self.canvas.w - pw) // 2, (self.canvas.h - ph) // 2)
        self.surf = self.pg.Surface((pw, ph))
        self.restart()

    def load_preset(self) -> None:
        """The preset's reactions become the editable list."""
        self.rx = [
            {"shape": r.shape, "chiral": r.chiral, "inp": list(r.inp), "out": list(r.out)}
            for r in preset(PRESETS[self.preset_i], self.n)
        ]
        self.edited = False

    def reactions(self) -> list[Reaction]:
        return [
            Reaction(s["shape"], list(s["inp"]), list(s["out"]), 1.0, chiral=s["chiral"])
            for s in self.rx
        ]

    def apply_rules(self) -> None:
        """Live: the running mosaic continues with the new rules."""
        rs = ReactionSet(self.reactions(), self.n)
        self.sim.reactions = rs
        self.sim.fires = [0] * len(rs.reactions)
        self.sim.rebuild()

    def restart(self) -> None:
        """Back to the picture, paused."""
        rs = ReactionSet(self.reactions(), self.n)
        self.sim = Sim(self.pic.graph, rs, self.pic.colours, self.cfg["sim"]["seed"])
        self.paused = True
        self.due = 0.0
        self.rec_t = [0.0]
        self.rec = [self.sim.colours.copy()]
        self.rec_next = self.cfg["rewind"]["t0"]
        self.replay: dict | None = None

    # ---- simulation and recording ------------------------------------------------------------
    def _record(self) -> None:
        rw = self.cfg["rewind"]
        t = self.sim.t
        self.rec_t.append(t)
        self.rec.append(self.sim.colours.copy())
        self.rec_next = t + min(max(t * (rw["ratio"] - 1.0), rw["t0"]), rw["max_dt"])
        if len(self.rec) > rw["max_frames"]:
            keep = sorted(set(range(0, len(self.rec), 2)) | {len(self.rec) - 1})
            self.rec = [self.rec[i] for i in keep]
            self.rec_t = [self.rec_t[i] for i in keep]

    def step(self, dt_wall: float) -> None:
        sim, sc = self.sim, self.cfg["sim"]
        if self.replay is not None:
            r = self.replay
            r["pos"] = min(r["pos"] + dt_wall * self.cfg["window"]["fps"], r["n"] - 1)
            return
        if self.paused or sim.n_elig == 0 or sim.K <= 0:
            return
        rate = sc["fast"] if self.fast else sc["speed"]
        self.due = min(self.due + dt_wall * rate, 0.25 * rate + 0.1)
        t_end = time.perf_counter() + sc["frame_budget_ms"] * 1e-3
        while self.due > 0 and time.perf_counter() < t_end:
            n = min(sc["chunk_attempts"], max(1, int(self.due * sim.K * sim.n_elig)))
            t0 = sim.t
            sim.attempts(n)
            self.due -= sim.t - t0
            if sim.t >= self.rec_next:
                self._record()

    def paced(self, m: int, how: str) -> np.ndarray:
        """m indices into the recording, start to now, paced by time or by change."""
        f = np.array(self.rec)
        e = self.sim.edge_arr
        acc = (f == f[0][None]).mean(axis=1)
        rho = (f[:, e[:, 0]] != f[:, e[:, 1]]).mean(axis=1)
        return pace_frames(np.array(self.rec_t), [acc, rho], how, m)

    def rewound(self) -> bool:
        return self.replay is not None and self.replay["pos"] >= self.replay["n"] - 1

    def start_rewind(self) -> bool:
        if self.sim.t > self.rec_t[-1]:
            self._record()
        if len(self.rec) < 3:
            return False
        rw = self.cfg["rewind"]
        n = max(2, int(rw["duration_s"] * self.cfg["window"]["fps"]))
        self.replay = {"idx": self.paced(n, rw["pacing"])[::-1], "pos": 0.0, "n": n}
        return True

    def shown(self) -> np.ndarray:
        if self.replay is None:
            return self.sim.colours
        r = self.replay
        return self.rec[int(r["idx"][int(r["pos"])])]

    def save_gif(self, folder: Path | None = None, wait: bool = False) -> Path | None:
        if self.saving is not None and self.saving.is_alive():
            return None
        if self.sim.t > self.rec_t[-1]:
            self._record()
        if len(self.rec) < 3:
            return None
        gc = self.cfg["gif"]
        idx = self.paced(gc["frames"], gc["pacing"])[::-1]
        frames = [self.rec[i] for i in idx]
        pm = hex_pixel_map(self.pic.graph, gc["hex_px"])
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = "custom" if self.edited else PRESETS[self.preset_i].replace(" ", "_")
        opt = self.options()[self.opt_i]
        path = (folder or ROOT / gc["folder"]) / f"{self.pic.key}_{opt}_{name}_{stamp}.gif"
        pal = self.pic.palette.copy()

        def work() -> None:
            write_gif(path, pm, frames, pal, gc)
            self.saved_at = time.time()
            print(f"saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")

        self.saving = threading.Thread(target=work, daemon=True)
        self.saving.start()
        if wait:
            self.saving.join()
        return path

    # ---- input -------------------------------------------------------------------------------
    def button(self, key: str, d: int = 1) -> None:
        if key in ("pic-", "pic+"):
            self.pic_i = (self.pic_i + (1 if key == "pic+" else -1)) % len(self.cfg["pictures"])
            self.opt_i = 0
            self.load()
        elif key in ("opt-", "opt+"):
            self.opt_i = (self.opt_i + (1 if key == "opt+" else -1)) % len(self.options())
            self.load()
        elif key in ("rule-", "rule+"):
            self.preset_i = (self.preset_i + (1 if key == "rule+" else -1)) % len(PRESETS)
            self.load_preset()
            self.apply_rules()
        elif key.startswith("arrow"):
            i = int(key[5:])
            if d < 0:
                del self.rx[i]
            else:
                s = self.rx[i]
                k = (KINDS.index((s["shape"], s["chiral"] or s["shape"] != "vertex")) + 1) % 4
                s["shape"], s["chiral"] = KINDS[k]
                m = ARITY[s["shape"]]
                s["inp"] = (s["inp"] + s["inp"][-1:] * m)[:m]
                s["out"] = (s["out"] + s["out"][-1:] * m)[:m]
            self.edited = True
            self.apply_rules()
        elif key == "add":
            if len(self.rx) < self.cfg["sim"]["max_reactions"]:
                self.rx.append({"shape": "edge", "chiral": True, "inp": [0, 1], "out": [0, 0]})
                self.edited = True
                self.apply_rules()
        elif key == "play":
            if self.rewound():  # the movie is back at the picture: a new run
                self.restart()
                self.paused = False
            elif self.replay is not None:  # mid-rewind: back to where the mosaic was
                self.replay = None
                self.paused = False
            else:
                self.paused = not self.paused
        elif key == "fast":
            self.fast = not self.fast
        elif key == "back":
            if self.replay is None:
                self.start_rewind()
            else:
                self.replay["pos"] = self.replay["n"] - 1
        elif key == "save":
            self.save_gif()

    def block_click(self, b: tuple[int, str, int], d: int) -> None:
        i, side, j = b
        self.rx[i][side][j] = (self.rx[i][side][j] + d) % self.n
        self.edited = True
        self.apply_rules()

    def click(self, pos, b: int) -> None:
        d = 1 if b == 1 else -1
        for blk, c in self.blocks.items():
            if math.hypot(pos[0] - c[0], pos[1] - c[1]) <= self.rb:
                self.block_click(blk, d)
                return
        for key, r in self.buttons.items():
            if r.collidepoint(pos):
                self.button(key, d)
                return

    # ---- drawing -----------------------------------------------------------------------------
    def colour(self, c: int) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.pic.palette[c])

    def swatch(self, c, rad: float, colour: int) -> None:
        pg = self.pg
        pg.draw.polygon(self.screen, self.colour(colour), hex_points(c, rad))
        pg.draw.polygon(self.screen, MID, hex_points(c, rad), 1)

    def icon(self, key: str, kind: str, r, on: bool = False, col=FG) -> None:
        """Button with a drawn icon: left, right, play, pause, fast, back, save, plus."""
        pg = self.pg
        r = pg.Rect(r)
        pg.draw.rect(self.screen, BTN_ON if on else BTN, r, border_radius=6)
        cx, cy = r.center
        s = min(r.w, r.h) * 0.26

        def tri(x: float, sign: int) -> None:  # triangle pointing right (sign 1) or left
            pts = [
                (x - sign * 0.8 * s, cy - s),
                (x + sign * 0.8 * s, cy),
                (x - sign * 0.8 * s, cy + s),
            ]
            pg.draw.polygon(self.screen, col, pts)

        if kind in ("left", "right"):
            sg = 1 if kind == "right" else -1
            pts = [
                (cx - sg * 0.5 * s, cy - s),
                (cx + sg * 0.5 * s, cy),
                (cx - sg * 0.5 * s, cy + s),
            ]
            pg.draw.lines(self.screen, col, False, pts, 2)
        elif kind == "play":
            tri(cx + 0.1 * s, 1)
        elif kind == "pause":
            for dx in (-0.6, 0.6):
                pg.draw.rect(
                    self.screen, col, pg.Rect(cx + dx * s - 0.3 * s, cy - s, 0.6 * s, 2 * s)
                )
        elif kind in ("fast", "back"):
            sg = 1 if kind == "fast" else -1
            tri(cx - 0.8 * s, sg)
            tri(cx + 0.8 * s, sg)
        elif kind == "save":  # arrow down into a tray
            pg.draw.line(self.screen, col, (cx, cy - 1.1 * s), (cx, cy + 0.3 * s), 3)
            pts = [(cx - 0.7 * s, cy - 0.3 * s), (cx, cy + 0.5 * s), (cx + 0.7 * s, cy - 0.3 * s)]
            pg.draw.polygon(self.screen, col, pts)
            tray = [(cx - 1.1 * s, cy + 0.5 * s), (cx - 1.1 * s, cy + 1.1 * s)]
            tray += [(cx + 1.1 * s, cy + 1.1 * s), (cx + 1.1 * s, cy + 0.5 * s)]
            pg.draw.lines(self.screen, col, False, tray, 3)
        elif kind == "plus":
            pg.draw.line(self.screen, col, (cx - s, cy), (cx + s, cy), 2)
            pg.draw.line(self.screen, col, (cx, cy - s), (cx, cy + s), 2)
        self.buttons[key] = r

    def dots(self, cx: float, y: float, n: int, cur: int) -> None:
        for k in range(n):
            x = cx + (k - (n - 1) / 2) * 14
            self.pg.draw.circle(
                self.screen, FG if k == cur else BTN_ON, (x, y), 4 if k == cur else 3
            )

    def title(self, x: int, y: int, w: int) -> int:
        """Title wrapped and centred between the two picture arrows; returns the bottom y."""
        words, lines, line = self.pic.title.split(), [], ""
        for word in words:
            test = f"{line} {word}".strip()
            if self.font.size(test)[0] > w - 84 and line:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
        lh = self.font.get_linesize()
        h = max(34, lh * len(lines))
        for k, s in enumerate(lines):
            t = self.font.render(s, True, FG)
            self.screen.blit(
                t, t.get_rect(midtop=(x + w // 2, y + (h - lh * len(lines)) // 2 + k * lh))
            )
        self.icon("pic-", "left", (x, y + h // 2 - 17, 34, 34))
        self.icon("pic+", "right", (x + w - 34, y + h // 2 - 17, 34, 34))
        return y + h

    def arrow(self, i: int, x: float, cy: float, length: float, s: dict) -> None:
        """Reaction arrow (a button). A clockwise-only vertex reaction carries a curved
        arrow; a vertex reaction valid for either hand has none."""
        pg = self.pg
        tip = x + length
        pg.draw.line(self.screen, FG, (x, cy), (tip - 4, cy), 2)
        pg.draw.polygon(self.screen, FG, [(tip - 6, cy - 4), (tip, cy), (tip - 6, cy + 4)])
        if s["shape"] == "vertex" and s["chiral"]:
            rad = max(4.0, 0.3 * length)
            c = (x + length / 2 - 1, cy - rad - 5)
            box = pg.Rect(c[0] - rad, c[1] - rad, 2 * rad, 2 * rad)
            pg.draw.arc(self.screen, MID, box, math.radians(-60), math.radians(200), 2)
            ex, ey = (
                c[0] + rad * math.cos(math.radians(60)),
                c[1] + rad * math.sin(math.radians(60)),
            )
            pg.draw.polygon(self.screen, MID, [(ex - 5, ey - 3), (ex + 2, ey - 4), (ex, ey + 3)])
        h = box_size(self.rb)[1]
        self.buttons[f"arrow{i}"] = pg.Rect(x - 2, cy - h / 2, length + 4, h)

    def draw_reactions(self, x: int, y: int, w: int, bottom: int) -> None:
        """The reaction list as a grid of tiles: columns and block size follow the count."""
        n = len(self.rx) + 1  # + the add button
        gap = 10
        for cols in (2, 3, 4, 5):
            tw = (w - (cols - 1) * 14) / cols
            # tile = box, arrow, box with box width sqrt3 rb (1 + GAP) and arrow 1.9 rb
            rb = min(15.0, tw / (2 * SQRT3 * (1 + GAP) + 2.3))
            bh = box_size(rb)[1]
            if math.ceil(n / cols) * (bh + gap) <= bottom - y:
                break
        self.rb = rb
        bw, bh = box_size(rb)
        al = min(tw - 2 * bw - 0.4 * rb, 2.4 * rb)  # arrow length
        self.blocks = {}
        for i, s in enumerate(self.rx):
            tx = x + (i % cols) * (tw + 14)
            ty = y + (i // cols) * (bh + gap)
            for side, bx in (("inp", tx), ("out", tx + bw + al + 0.4 * rb)):
                for j, (dx, dy) in enumerate(block_offsets(s["shape"], rb)):
                    c = (bx + dx, ty + dy)
                    self.swatch(c, rb, s[side][j])
                    self.blocks[(i, side, j)] = c
            self.arrow(i, tx + bw + 0.2 * rb, ty + bh / 2, al, s)
        if len(self.rx) < self.cfg["sim"]["max_reactions"]:
            i = len(self.rx)
            tx = x + (i % cols) * (tw + 14)
            ty = y + (i // cols) * (bh + gap)
            side = min(bh, 30)
            self.icon("add", "plus", (tx, ty + (bh - side) / 2, side, side))

    def draw(self) -> None:
        pg = self.pg
        self.screen.fill(BG)
        rgb = render_rgb(self.pixmap, self.shown(), self.pic.palette, BG)
        pg.surfarray.blit_array(self.surf, rgb.swapaxes(0, 1))
        self.screen.blit(self.surf, self.offset)
        pg.draw.rect(self.screen, PANEL, (self.canvas.right, 0, 9999, self.canvas.h))
        self.buttons = {}
        x, w = self.px, self.screen.get_width() - self.px - 18
        H = self.screen.get_height()
        y = self.title(x, 18, w) + 18
        # colours of the picture and the choice of colours
        self.icon("opt-", "left", (x, y, 28, 28))
        self.icon("opt+", "right", (x + w - 28, y, 28, 28))
        for c in range(self.n):
            self.swatch((x + w / 2 + (c - (self.n - 1) / 2) * 40, y + 14), 16, c)
        self.dots(x + w / 2, y + 42, len(self.options()), self.opt_i)
        y += 64
        # rule presets, then the reactions
        self.icon("rule-", "left", (x, y, 28, 28))
        self.icon("rule+", "right", (x + w - 28, y, 28, 28))
        self.dots(x + w / 2, y + 14, len(PRESETS), -1 if self.edited else self.preset_i)
        y += 44
        bar = 64
        self.draw_reactions(x, y, w, H - bar - 40)
        # transport
        ty = H - bar - 18
        bw = (w - 3 * 12) / 4
        playing = not self.paused and self.replay is None
        self.icon("play", "pause" if playing else "play", (x, ty, bw, bar))
        self.icon("fast", "fast", (x + (bw + 12), ty, bw, bar), on=self.fast)
        self.icon("back", "back", (x + 2 * (bw + 12), ty, bw, bar), on=self.replay is not None)
        busy = self.saving is not None and self.saving.is_alive()
        done = time.time() - self.saved_at < 1.5
        self.icon(
            "save", "save", (x + 3 * (bw + 12), ty, bw, bar), on=busy, col=ACCENT if done else FG
        )
        if self.replay is not None:  # progress of the movie played backwards
            f = self.replay["pos"] / (self.replay["n"] - 1)
            r = self.buttons["back"]
            pg.draw.rect(self.screen, ACCENT, (r.x, r.bottom + 5, r.w * f, 3))

    # ---- main loop ---------------------------------------------------------------------------
    def loop(self, smoke: int = 0) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        frame, running = 0, True
        keymap = {
            pg.K_SPACE: "play",
            pg.K_f: "fast",
            pg.K_b: "back",
            pg.K_s: "save",
            pg.K_LEFT: "pic-",
            pg.K_RIGHT: "pic+",
            pg.K_UP: "rule-",
            pg.K_DOWN: "rule+",
            pg.K_c: "opt+",
        }
        while running:
            dtw = clock.tick(self.cfg["window"]["fps"]) / 1000.0
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.MOUSEBUTTONDOWN and ev.button in (1, 3):
                    self.click(ev.pos, ev.button)
                elif ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_ESCAPE:
                        running = False
                    elif ev.key in keymap:
                        self.button(keymap[ev.key])
            if smoke:
                dtw = 1.0 / self.cfg["window"]["fps"]
                running = self.smoke_step(frame, smoke)
            self.step(dtw)
            self.draw()
            pg.display.flip()
            frame += 1

    def smoke_step(self, frame: int, total: int) -> bool:
        """Headless checks: every picture, colour choice and preset loads, starts paused and
        runs; editing by clicks; rewind returns to the picture; a GIF is written. Saves a
        screenshot of every preset's panel and one at the end."""
        out = ROOT / "runs" / "toy"
        out.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        if frame == 0:
            nopt = 0
            for _ in range(len(self.cfg["pictures"])):
                self.button("pic+")
                for _ in self.options():
                    self.button("opt+")
                    assert self.paused and self.sim.t == 0
                    self.sim.advance(0.3)
                    nopt += 1
            for _ in range(len(PRESETS)):
                self.button("rule+")
                self.sim.advance(0.3)
                self.draw()
                name = PRESETS[self.preset_i].replace(" ", "_")
                self.pg.image.save(self.screen, str(out / f"panel_{name}_{stamp}.png"))
            # edit by clicks: add a reaction, turn it into a vertex, change a block, remove
            n0 = len(self.rx)
            self.draw()
            self.click(self.buttons["add"].center, 1)
            assert len(self.rx) == n0 + 1 and self.edited
            self.draw()
            self.click(self.buttons[f"arrow{n0}"].center, 1)
            assert self.rx[n0]["shape"] == "vertex" and self.rx[n0]["chiral"]
            self.draw()
            self.click(self.blocks[(n0, "out", 2)], 1)
            assert self.reactions()[n0].out[2] == (self.rx[n0]["out"][1] + 1) % self.n
            self.click(self.buttons[f"arrow{n0}"].center, 3)
            assert len(self.rx) == n0
            print(f"smoke: {len(self.cfg['pictures'])} pictures, {nopt} colour choices,", end=" ")
            print(f"{len(PRESETS)} presets, editing OK")
            self.restart()
            assert self.paused
            self.button("play")
            self.button("fast")
        if frame == total - 2:
            self.button("back")
            assert self.replay is not None
            self.button("back")  # jump to the picture
            assert self.rewound() and (self.shown() == self.pic.colours).all()
            p = self.save_gif(out, wait=True)
            assert p is not None and p.exists() and p.stat().st_size > 0
            print(f"smoke: rewind OK ({len(self.rec)} frames), GIF {p.relative_to(ROOT)}")
            self.button("play")  # a new run from the picture
            assert self.replay is None and not self.paused and self.sim.t == 0
        if frame >= total:
            self.draw()
            p = out / f"screen_{stamp}.png"
            self.pg.image.save(self.screen, str(p))
            print(f"smoke: screenshot {p.relative_to(ROOT)}")
            return False
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picture", default=None, help="key of a [[pictures]] entry")
    ap.add_argument("--rules", default=None, choices=PRESETS)
    ap.add_argument("--smoke", type=int, default=0, help="headless: run N frames and checks")
    a = ap.parse_args()
    if a.smoke:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    pygame.init()
    toy = Toy(load_cfg(), a.picture, a.rules)
    toy.loop(a.smoke)
    pygame.quit()


if __name__ == "__main__":
    main()
