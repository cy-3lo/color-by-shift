"""Painting toy: pick a picture and some reactions, watch the mosaic dissolve, then play it
backwards and save the reversed movie as a GIF.

    .venv/bin/python toy/painting_toy.py [--picture great_wave] [--colours 4] [--rules voter]

Panel (right):
  Picture   < name >      pictures of toy/toy.toml (paintings, generated pavings)
  Colours   - n +         the picture is re-quantised (restarts)
  Rules     < preset >    built-in rule sets (all k = 1), applied live
  Your reactions          up to [sim] slots rules of your own, all k = 1, applied live:
                          shape button cycles off / cell / edge / vertex (right click back);
                          click a block for the next colour, right click for the previous
  Play / Pause, speed, Restart (back to the picture), Rewind (play the run backwards),
  Save GIF (the run backwards, from the dissolved state to the picture, in gifs/).
Keys: Space pause, 1-4 speed, R restart, B rewind, G save GIF, Left/Right picture,
Up/Down rules, -/+ colours, Esc quit.
"""

from __future__ import annotations

import argparse
import datetime as dt
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
from pictures import DESCRIPTIONS, PRESETS, load_picture, preset, rule_text  # noqa: E402

BG = (28, 28, 34)
PANEL = (38, 38, 46)
FG = (236, 233, 224)
MID = (150, 148, 160)
BTN = (58, 58, 70)
BTN_ON = (96, 92, 120)
ACCENT = (240, 180, 90)
ARITY = {"off": 0, "cell": 1, "edge": 2, "vertex": 3}
SHAPES = ["off", "cell", "edge", "vertex"]


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


class Toy:
    def __init__(self, cfg: dict, picture: str | None, colours: int | None, rules: str | None):
        import pygame

        self.pg = pygame
        self.cfg = cfg
        w, h = cfg["window"]["width"], cfg["window"]["height"]
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Mosaic: painting toy")
        self.font = pygame.font.Font(None, 21)
        self.small = pygame.font.Font(None, 17)
        self.big = pygame.font.Font(None, 30)
        self.canvas = pygame.Rect(0, 0, cfg["window"]["canvas"], h)
        self.px = self.canvas.right + 18
        keys = [p["key"] for p in cfg["pictures"]]
        self.pic_i = keys.index(picture) if picture else 0
        self.n = colours or cfg["sim"]["colours"]
        self.preset_i = PRESETS.index(rules) if rules else 0
        self.slots = [{"shape": "off", "inp": [], "out": []} for _ in range(cfg["sim"]["slots"])]
        self.speed_i = cfg["sim"]["speeds"].index(1) if 1 in cfg["sim"]["speeds"] else 0
        self.buttons: dict[str, object] = {}
        self.notice, self.notice_until = "", 0.0
        self.saving: threading.Thread | None = None
        self.cache: dict = {}
        self.load()

    # ---- picture and rules -------------------------------------------------------------------
    def load(self) -> None:
        sc = self.cfg["sim"]
        entry = self.cfg["pictures"][self.pic_i]
        key = (entry["key"], self.n)
        if key not in self.cache:
            gp = self.cfg["generated"]["palette"]
            self.cache[key] = load_picture(entry, self.n, sc["cells"], sc["smooth"], sc["seed"], gp)
        self.pic = self.cache[key]
        g = self.pic.graph
        size = fit_size(g, self.canvas.w - 24, self.canvas.h - 24)
        self.pixmap = hex_pixel_map(g, size)
        ph, pw = self.pixmap.shape
        self.offset = ((self.canvas.w - pw) // 2, (self.canvas.h - ph) // 2)
        self.surf = self.pg.Surface((pw, ph))
        for s in self.slots:  # keep your reactions inside the palette
            s["inp"] = [c % self.n for c in s["inp"]]
            s["out"] = [c % self.n for c in s["out"]]
        self.restart()

    def reactions(self) -> list[Reaction]:
        out = preset(PRESETS[self.preset_i], self.n)
        for s in self.slots:
            if s["shape"] != "off":
                out.append(Reaction(s["shape"], list(s["inp"]), list(s["out"]), 1.0))
        return out

    def apply_rules(self) -> None:
        """Live: the running mosaic continues with the new rules."""
        rs = ReactionSet(self.reactions(), self.n)
        self.sim.reactions = rs
        self.sim.fires = [0] * len(rs.reactions)
        self.sim.rebuild()

    def restart(self) -> None:
        g = self.pic.graph
        rs = ReactionSet(self.reactions(), self.n)
        self.sim = Sim(g, rs, self.pic.colours, self.cfg["sim"]["seed"])
        self.paused = False
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
        rate = sc["speeds"][self.speed_i]
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

    def start_rewind(self) -> None:
        if self.sim.t > self.rec_t[-1]:
            self._record()
        if len(self.rec) < 3:
            self.note("nothing to rewind yet: let it play first")
            return
        rw = self.cfg["rewind"]
        n = max(2, int(rw["duration_s"] * self.cfg["window"]["fps"]))
        self.replay = {"idx": self.paced(n, rw["pacing"])[::-1], "pos": 0.0, "n": n}

    def shown(self) -> np.ndarray:
        if self.replay is None:
            return self.sim.colours
        r = self.replay
        return self.rec[int(r["idx"][int(r["pos"])])]

    def save_gif(self, folder: Path | None = None, wait: bool = False) -> Path | None:
        if self.saving is not None and self.saving.is_alive():
            self.note("still saving the previous GIF")
            return None
        if self.sim.t > self.rec_t[-1]:
            self._record()
        if len(self.rec) < 3:
            self.note("nothing to save yet: let it play first")
            return None
        gc = self.cfg["gif"]
        idx = self.paced(gc["frames"], gc["pacing"])[::-1]
        frames = [self.rec[i] for i in idx]
        pm = hex_pixel_map(self.pic.graph, gc["hex_px"])
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = PRESETS[self.preset_i].replace(" ", "_")
        if any(s["shape"] != "off" for s in self.slots):
            name += "+own"
        path = (folder or ROOT / gc["folder"]) / f"{self.pic.key}_{self.n}c_{name}_{stamp}.gif"
        pal = self.pic.palette.copy()

        def work() -> None:
            write_gif(path, pm, frames, pal, gc)
            self.note(f"saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")

        self.note("saving GIF ...")
        self.saving = threading.Thread(target=work, daemon=True)
        self.saving.start()
        if wait:
            self.saving.join()
        return path

    def note(self, msg: str, secs: float = 5.0) -> None:
        self.notice, self.notice_until = msg, time.time() + secs

    # ---- input -------------------------------------------------------------------------------
    def button(self, key: str, d: int = 1) -> None:
        npic = len(self.cfg["pictures"])
        if key in ("pic-", "pic+"):
            self.pic_i = (self.pic_i + (1 if key == "pic+" else -1)) % npic
            self.load()
        elif key in ("col-", "col+"):
            n = min(8, max(2, self.n + (1 if key == "col+" else -1)))
            if n != self.n:
                self.n = n
                self.load()
        elif key in ("rule-", "rule+"):
            self.preset_i = (self.preset_i + (1 if key == "rule+" else -1)) % len(PRESETS)
            self.apply_rules()
        elif key.startswith("shape"):
            s = self.slots[int(key[5:])]
            s["shape"] = SHAPES[(SHAPES.index(s["shape"]) + d) % len(SHAPES)]
            k = ARITY[s["shape"]]
            s["inp"] = (s["inp"] + [0] * k)[:k]
            s["out"] = (s["out"] + [0] * k)[:k]
            self.apply_rules()
        elif key.startswith("blk"):
            i, side, j = key[3:].split(":")
            s = self.slots[int(i)]
            s[side][int(j)] = (s[side][int(j)] + d) % self.n
            self.apply_rules()
        elif key == "clear":
            for s in self.slots:
                s.update(shape="off", inp=[], out=[])
            self.apply_rules()
        elif key == "pause":
            if self.replay is not None:
                self.replay = None
                self.paused = False
            else:
                self.paused = not self.paused
        elif key.startswith("speed"):
            self.speed_i = int(key[5:])
        elif key == "restart":
            self.restart()
        elif key == "rewind":
            if self.replay is None:
                self.start_rewind()
            else:
                self.replay = None
        elif key == "gif":
            self.save_gif()

    def click(self, pos, b: int) -> None:
        for key, r in self.buttons.items():
            if r.collidepoint(pos):
                self.button(key, 1 if b == 1 else -1)
                return

    # ---- drawing -----------------------------------------------------------------------------
    def text(self, s: str, pos, font=None, col=FG) -> int:
        surf = (font or self.font).render(s, True, col)
        self.screen.blit(surf, pos)
        return surf.get_width()

    def btn(self, key: str, label: str, x: int, y: int, w: int, on: bool = False, h: int = 26):
        pg = self.pg
        r = pg.Rect(x, y, w, h)
        pg.draw.rect(self.screen, BTN_ON if on else BTN, r, border_radius=5)
        pg.draw.rect(self.screen, MID, r, 1, border_radius=5)
        t = self.font.render(label, True, FG)
        self.screen.blit(t, t.get_rect(center=r.center))
        self.buttons[key] = r
        return r

    def block(self, key: str, c: int, x: int, y: int, size: int = 22) -> None:
        pg = self.pg
        r = pg.Rect(x, y, size, size)
        col = tuple(int(v) for v in self.pic.palette[c])
        pg.draw.rect(self.screen, col, r, border_radius=4)
        pg.draw.rect(self.screen, MID, r, 1, border_radius=4)
        self.buttons[key] = r

    def wrap(self, s: str, x: int, y: int, width: int, font=None, col=MID) -> int:
        font = font or self.small
        line = ""
        for word in s.split():
            test = f"{line} {word}".strip()
            if font.size(test)[0] > width and line:
                self.text(line, (x, y), font, col)
                y += font.get_linesize()
                line = word
            else:
                line = test
        if line:
            self.text(line, (x, y), font, col)
            y += font.get_linesize()
        return y

    def draw(self) -> None:
        pg = self.pg
        self.screen.fill(BG)
        rgb = render_rgb(self.pixmap, self.shown(), self.pic.palette, BG)
        pg.surfarray.blit_array(self.surf, rgb.swapaxes(0, 1))
        self.screen.blit(self.surf, self.offset)
        pg.draw.rect(self.screen, PANEL, (self.canvas.right, 0, 9999, self.canvas.h))
        self.buttons = {}
        x, y, w = self.px, 16, self.screen.get_width() - self.px - 16
        y = self.wrap(self.pic.title, x, y, w, self.big, FG) + 10
        # picture and colours
        self.text("Picture", (x, y + 5), self.font, MID)
        self.btn("pic-", "<", x + 80, y, 28)
        self.text(self.pic.key.replace("_", " "), (x + 116, y + 5))
        self.btn("pic+", ">", x + w - 28, y, 28)
        y += 34
        self.text("Colours", (x, y + 5), self.font, MID)
        self.btn("col-", "-", x + 80, y, 28)
        self.text(str(self.n), (x + 118, y + 5))
        self.btn("col+", "+", x + 136, y, 28)
        for c in range(self.n):
            col = tuple(int(v) for v in self.pic.palette[c])
            pg.draw.rect(self.screen, col, (x + 180 + 26 * c, y + 3, 20, 20), border_radius=4)
            self.text(str(c), (x + 186 + 26 * c, y + 26), self.small, MID)
        y += 50
        # rules
        self.text("Rules", (x, y + 5), self.font, MID)
        self.btn("rule-", "<", x + 80, y, 28)
        self.text(PRESETS[self.preset_i], (x + 116, y + 5))
        self.btn("rule+", ">", x + w - 28, y, 28)
        y += 32
        y = self.wrap(DESCRIPTIONS[PRESETS[self.preset_i]], x, y, w)
        rt = rule_text(preset(PRESETS[self.preset_i], self.n))
        if rt:
            y = self.wrap(
                rt if len(rt) < 160 else rt[:157] + "...", x, y + 2, w, col=(120, 118, 130)
            )
        y += 12
        # your reactions
        self.text("Your reactions (k = 1)", (x, y), self.font, MID)
        self.btn("clear", "Clear", x + w - 60, y - 4, 60)
        y += 28
        for i, s in enumerate(self.slots):
            self.btn(f"shape{i}", s["shape"], x, y, 62)
            bx = x + 72
            for j, c in enumerate(s["inp"]):
                self.block(f"blk{i}:inp:{j}", c, bx, y + 2)
                bx += 26
            if s["shape"] != "off":
                pg.draw.line(self.screen, FG, (bx + 4, y + 13), (bx + 26, y + 13), 2)
                pg.draw.polygon(
                    self.screen, FG, [(bx + 26, y + 8), (bx + 33, y + 13), (bx + 26, y + 18)]
                )
                bx += 42
                for j, c in enumerate(s["out"]):
                    self.block(f"blk{i}:out:{j}", c, bx, y + 2)
                    bx += 26
                if s["shape"] == "vertex":
                    self.text("clockwise", (bx + 6, y + 7), self.small, MID)
            y += 32
        y += 8
        # transport
        bx = x
        playing = not self.paused and self.replay is None
        self.btn("pause", "Pause" if playing else "Play", bx, y, 64, on=not playing)
        bx += 72
        for k, m in enumerate(self.cfg["sim"]["speeds"]):
            lab = "x1/4" if m == 0.25 else f"x{m:g}"
            self.btn(f"speed{k}", lab, bx, y, 46, on=k == self.speed_i)
            bx += 52
        y += 34
        self.btn("restart", "Restart", x, y, 80)
        self.btn("rewind", "Rewind", x + 88, y, 80, on=self.replay is not None)
        self.btn("gif", "Save GIF", x + 176, y, 90)
        y += 40
        state = f"t = {self.sim.t:7.1f}   frames recorded {len(self.rec)}"
        self.text(state, (x, y), self.small, MID)
        y += 22
        if self.notice and time.time() < self.notice_until:
            self.wrap(self.notice, x, y, w, self.small, ACCENT)
        elif self.replay is not None:
            done = self.replay["pos"] >= self.replay["n"] - 1
            msg = "back at the picture: Play continues the mosaic" if done else "rewinding ..."
            self.text(msg, (x, y), self.small, ACCENT)
        foot = "Space play, 1-4 speed, R restart, B rewind, G GIF, arrows: picture and rules"
        self.text(foot, (x, self.screen.get_height() - 24), self.small, MID)

    # ---- main loop ---------------------------------------------------------------------------
    def loop(self, smoke: int = 0) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        frame, running = 0, True
        keymap = {
            pg.K_SPACE: "pause",
            pg.K_r: "restart",
            pg.K_b: "rewind",
            pg.K_g: "gif",
            pg.K_LEFT: "pic-",
            pg.K_RIGHT: "pic+",
            pg.K_UP: "rule-",
            pg.K_DOWN: "rule+",
            pg.K_MINUS: "col-",
            pg.K_EQUALS: "col+",
            pg.K_PLUS: "col+",
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
                    elif pg.K_1 <= ev.key <= pg.K_4:
                        self.button(f"speed{ev.key - pg.K_1}")
            if smoke:
                dtw = 1.0 / self.cfg["window"]["fps"]
                running = self.smoke_step(frame, smoke)
            self.step(dtw)
            self.draw()
            pg.display.flip()
            frame += 1

    def smoke_step(self, frame: int, total: int) -> bool:
        """Headless checks: every picture and preset loads and runs; own reactions; rewind
        returns to the picture; a GIF is written. Saves a screenshot at the end."""
        out = ROOT / "runs" / "toy"
        if frame == 0:
            self.speed_i = len(self.cfg["sim"]["speeds"]) - 1
            for _ in range(len(self.cfg["pictures"])):
                self.button("pic+")
                self.sim.advance(0.5)
            for _ in range(len(PRESETS)):
                self.button("rule+")
                self.sim.advance(0.5)
            self.button("col+")
            self.button("shape0")
            self.button("shape0")  # edge
            self.button("blk0:out:1")
            assert self.reactions()[-1].shape == "edge"
            print(f"smoke: {len(self.cfg['pictures'])} pictures, {len(PRESETS)} presets OK")
            self.restart()
        if frame == total - 2:
            self.button("rewind")
            assert self.replay is not None
            self.replay["pos"] = self.replay["n"] - 1
            assert (self.shown() == self.pic.colours).all()
            self.button("rewind")
            p = self.save_gif(out, wait=True)
            assert p is not None and p.exists() and p.stat().st_size > 0
            print(f"smoke: rewind OK ({len(self.rec)} frames), GIF {p.relative_to(ROOT)}")
        if frame >= total:
            self.draw()
            p = out / f"screen_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
            self.pg.image.save(self.screen, str(p))
            print(f"smoke: screenshot {p.relative_to(ROOT)}")
            return False
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picture", default=None, help="key of a [[pictures]] entry")
    ap.add_argument("--colours", type=int, default=None)
    ap.add_argument("--rules", default=None, choices=PRESETS)
    ap.add_argument("--smoke", type=int, default=0, help="headless: run N frames and checks")
    a = ap.parse_args()
    if a.smoke:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    pygame.init()
    cfg = load_cfg()
    toy = Toy(cfg, a.picture, a.colours, a.rules)
    toy.loop(a.smoke)
    pygame.quit()


if __name__ == "__main__":
    main()
