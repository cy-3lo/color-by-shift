"""Painting toy: pick a picture and some reactions, watch the mosaic dissolve, then play it
backwards and save the reversed movie as a GIF.

    .venv/bin/python toy/painting_toy.py [--picture great_wave] [--rules voter]

The panel, top to bottom (the only texts are the picture's title and the rules' name):
  < title > dice   picture (Left / Right; dice or P: a random one)
  colours          the five colours of the picture, dark to light
  < rules > dice   rule preset (Up / Down; dice or R: a random one); its reactions are
                   listed below and are yours to edit, live, all with k = 1:
                     click a block          next colour (right click: previous)
                     click an arrow         next shape: cell, edge, vertex
                     right click an arrow   remove the reaction
                     +                      add a reaction
                   An edge is drawn as two neighbouring cells, a vertex as the three cells
                   around a corner (top, lower right, lower left = clockwise). Every
                   vertex reaction is chiral: it does not match the mirror arrangement.
                   The name is dimmed once the preset has been edited.
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
import random
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
from pictures import PRESETS, choice, load_picture, preset  # noqa: E402

BG = (28, 28, 34)
PANEL = (38, 38, 46)
FG = (236, 233, 224)
MID = (150, 148, 160)
BTN = (58, 58, 70)
BTN_ON = (96, 92, 120)
ACCENT = (240, 180, 90)
ARITY = {"cell": 1, "edge": 2, "vertex": 3}
SHAPES = ["cell", "edge", "vertex"]  # an arrow click cycles through these
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
        self.preset_i = PRESETS.index(rules) if rules else 0
        self.fast = False
        self.dice = random.Random()  # for the two random buttons only
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

    def load(self) -> None:
        sc = self.cfg["sim"]
        entry = self.entry()
        key = entry["key"]
        if key not in self.cache:
            cp = self.cfg["palettes"]
            self.cache[key] = load_picture(
                entry, self.n, sc["cells"], sc["smooth"], sc["seed"], cp, choice(entry, cp)
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
            {"shape": r.shape, "inp": list(r.inp), "out": list(r.out)}
            for r in preset(PRESETS[self.preset_i], self.n)
        ]
        self.edited = False

    def reactions(self) -> list[Reaction]:
        return [Reaction(s["shape"], list(s["inp"]), list(s["out"]), 1.0) for s in self.rx]

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
        path = (folder or ROOT / gc["folder"]) / f"{self.pic.key}_{name}_{stamp}.gif"
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
            self.load()
        elif key == "pic?":  # a random other picture
            n = len(self.cfg["pictures"])
            self.pic_i = (self.pic_i + self.dice.randrange(1, n)) % n if n > 1 else 0
            self.load()
        elif key == "rule?":  # a random other preset
            self.preset_i = (self.preset_i + self.dice.randrange(1, len(PRESETS))) % len(PRESETS)
            self.load_preset()
            self.apply_rules()
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
                s["shape"] = SHAPES[(SHAPES.index(s["shape"]) + 1) % len(SHAPES)]
                m = ARITY[s["shape"]]
                s["inp"] = (s["inp"] + s["inp"][-1:] * m)[:m]
                s["out"] = (s["out"] + s["out"][-1:] * m)[:m]
            self.edited = True
            self.apply_rules()
        elif key == "add":
            if len(self.rx) < self.cfg["sim"]["max_reactions"]:
                self.rx.append({"shape": "edge", "inp": [0, 1], "out": [0, 0]})
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
        """Button with a drawn icon: left, right, dice, play, pause, fast, back, save, plus."""
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
        elif kind == "save":  # floppy disc: cut corner, shutter at the top, label below
            u = 1.15 * s
            body = [(cx - u, cy - u), (cx + 0.6 * u, cy - u), (cx + u, cy - 0.6 * u)]
            body += [(cx + u, cy + u), (cx - u, cy + u)]
            pg.draw.polygon(self.screen, col, body)
            bg = BTN_ON if on else BTN
            pg.draw.rect(self.screen, bg, pg.Rect(cx - 0.55 * u, cy - u, 0.95 * u, 0.7 * u))
            pg.draw.rect(
                self.screen, col, pg.Rect(cx + 0.02 * u, cy - 0.88 * u, 0.24 * u, 0.46 * u)
            )
            pg.draw.rect(self.screen, bg, pg.Rect(cx - 0.65 * u, cy + 0.1 * u, 1.3 * u, 0.9 * u))
        elif kind == "dice":
            u = 1.25 * s
            pg.draw.rect(
                self.screen, col, pg.Rect(cx - u, cy - u, 2 * u, 2 * u), 2, border_radius=4
            )
            for dx, dy in ((-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)):
                pg.draw.circle(self.screen, col, (cx + 0.5 * u * dx, cy + 0.5 * u * dy), 2)
        elif kind == "plus":
            pg.draw.line(self.screen, col, (cx - s, cy), (cx + s, cy), 2)
            pg.draw.line(self.screen, col, (cx, cy - s), (cx, cy + s), 2)
        self.buttons[key] = r

    def heading(self, text: str, key: str, x: int, y: int, w: int, col=FG) -> int:
        """< text > dice: the text wrapped and centred between the arrows (buttons key-,
        key+ and key?); returns the bottom y."""
        tw = w - 3 * 34 - 24  # room for the text
        words, lines, line = text.split(), [], ""
        for word in words:
            test = f"{line} {word}".strip()
            if self.font.size(test)[0] > tw and line:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
        lh = self.font.get_linesize()
        h = max(34, lh * len(lines))
        mid = x + 34 + 8 + tw // 2
        for k, s in enumerate(lines):
            t = self.font.render(s, True, col)
            self.screen.blit(t, t.get_rect(midtop=(mid, y + (h - lh * len(lines)) // 2 + k * lh)))
        by = y + h // 2 - 17
        self.icon(f"{key}-", "left", (x, by, 34, 34))
        self.icon(f"{key}+", "right", (x + w - 2 * 34 - 8, by, 34, 34))
        self.icon(f"{key}?", "dice", (x + w - 34, by, 34, 34))
        return y + h

    def arrow(self, i: int, x: float, cy: float, length: float) -> None:
        """Reaction arrow (a button)."""
        pg = self.pg
        tip = x + length
        pg.draw.line(self.screen, FG, (x, cy), (tip - 4, cy), 2)
        pg.draw.polygon(self.screen, FG, [(tip - 6, cy - 4), (tip, cy), (tip - 6, cy + 4)])
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
            self.arrow(i, tx + bw + 0.2 * rb, ty + bh / 2, al)
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
        y = self.heading(self.pic.title, "pic", x, 18, w) + 16
        for c in range(self.n):  # the colours of the picture, dark to light
            self.swatch((x + w / 2 + (c - (self.n - 1) / 2) * 40, y + 16), 16, c)
        y += 52
        name = PRESETS[self.preset_i]
        y = self.heading(name, "rule", x, y, w, MID if self.edited else FG) + 18
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
            pg.K_p: "pic?",
            pg.K_r: "rule?",
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
        """Headless checks: every picture and preset loads, starts paused and
        runs; editing by clicks; rewind returns to the picture; a GIF is written. Saves a
        screenshot of every preset's panel and one at the end."""
        out = ROOT / "runs" / "toy"
        out.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        if frame == 0:
            for _ in range(len(self.cfg["pictures"])):
                self.button("pic+")
                assert self.paused and self.sim.t == 0
                self.sim.advance(0.3)
                self.draw()
                self.pg.image.save(self.screen, str(out / f"picture_{self.pic.key}_{stamp}.png"))
            for key in ("pic?", "rule?"):
                was = (self.pic_i, self.preset_i)
                self.click(self.buttons[key].center, 1)
                assert (self.pic_i, self.preset_i) != was
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
            assert self.rx[n0]["shape"] == "vertex" and self.reactions()[n0].chiral
            self.draw()
            self.click(self.blocks[(n0, "out", 2)], 1)
            assert self.reactions()[n0].out[2] == (self.rx[n0]["out"][1] + 1) % self.n
            self.click(self.buttons[f"arrow{n0}"].center, 3)
            assert len(self.rx) == n0
            print(f"smoke: {len(self.cfg['pictures'])} pictures, {len(PRESETS)} presets,", end=" ")
            print("random buttons, editing OK")
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
