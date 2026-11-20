"""Shift toy: a picture was scrambled by a few directional swaps; list the reactions that
put it back and play them, one by one (model: mosaic_model.shift, exactly reversible).

    .venv/bin/python toy/shift_toy.py [--level S01]

Time only goes forward: your list is applied once, from top to bottom. Each reaction moves a
cell by one place at most and undoes itself when applied again, so the way back is the
scrambling list in reverse order.

Drawer (right), top to bottom:
  < level >     level (Up / Down)
  reactions     your list. Each row: a direction button (the side shared by the two cells:
                | / \\; click = next direction, right click = remove the row) and the two
                colours that swap, drawn as neighbours in that direction (click a cell =
                next colour, right click = previous). + adds a row.
                Click a row (beside the button and the cells) to see the picture just
                after that reaction: the row is marked, the rows above it are applied, the
                rows below are not yet. Changing any reaction updates the picture at the
                step you are looking at.
  buttons       the scrambled picture (R), one reaction forward (Right or Space; Left goes
                one back), the picture to recover while switched on (T)
On the picture, the pairs swapped by the marked reaction are linked by a short bar. The frame
lights up when the picture is recovered. H prints the scrambling list in the terminal. Esc
quits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))

from mosaic_model.shift import DIRECTION_NAMES, Swap, load_shift_level  # noqa: E402

BG = (28, 28, 34)
PANEL = (38, 38, 46)
ROW = (50, 50, 60)
FG = (236, 233, 224)
MID = (150, 148, 160)
BTN = (58, 58, 70)
BTN_ON = (96, 92, 120)
ACCENT = (240, 180, 90)


def load_cfg() -> dict:
    return tomllib.loads((ROOT / "toy" / "shift.toml").read_text())


def hex_points(c, rad: float) -> list[tuple[float, float]]:
    """Pointy-top hexagon around c."""
    return [
        (
            c[0] + rad * math.cos(math.radians(60 * k - 90)),
            c[1] + rad * math.sin(math.radians(60 * k - 90)),
        )
        for k in range(6)
    ]


def rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[k : k + 2], 16) for k in (0, 2, 4))


class ShiftToy:
    def __init__(self, cfg: dict, level: str | None = None):
        import pygame

        self.pg = pygame
        self.cfg = cfg
        w, h = cfg["window"]["width"], cfg["window"]["height"]
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Mosaic shift")
        self.font = pygame.font.Font(None, 26)
        self.canvas = pygame.Rect(0, 0, w - cfg["window"]["drawer"], h)
        self.px = self.canvas.right + 12
        self.files = sorted((ROOT / cfg["levels"]["folder"]).glob("*.json"))
        ids = [f.stem for f in self.files]
        self.level_i = ids.index(level) if level else 0
        self.buttons: dict[str, object] = {}
        self.rows: dict[int, object] = {}
        self.cells: dict[tuple[int, int], tuple[float, float]] = {}
        self.rb = 13.0
        self.peek = False
        self.load()

    # ---- level and moves ---------------------------------------------------------------------
    def new_row(self) -> dict:
        return {"dir": "|", "pair": [0, min(1, self.level.n - 1)]}

    def load(self) -> None:
        self.level = load_shift_level(self.files[self.level_i])
        self.pal = [rgb(c) for c in self.level.colours]
        self.start = self.level.scrambled()
        # your list starts with as many rows as the scrambling one, all the same
        self.program = [self.new_row() for _ in range(self.level.slots)]
        self.goto(0)

    def swaps(self) -> list[Swap]:
        return [Swap(r["dir"], r["pair"][0], r["pair"][1], self.level.alt) for r in self.program]

    def goto(self, k: int) -> None:
        """Show the picture after the first k reactions of your list (0 = scrambled)."""
        self.k = max(0, min(k, len(self.program)))
        col = self.start
        self.moved: tuple[np.ndarray, np.ndarray] | None = None
        for s in self.swaps()[: self.k]:
            new = self.level.grid.apply(col, s)
            alts = (s.alt,) if s.alt is not None else (0, 1)
            p = np.concatenate([self.level.grid.pairs[(s.direction, a)][0] for a in alts])
            q = np.concatenate([self.level.grid.pairs[(s.direction, a)][1] for a in alts])
            m = new[p] != col[p]
            self.moved = (p[m], q[m])  # kept for the last reaction applied
            col = new
        self.col = col

    def solved(self) -> bool:
        return bool((self.col == self.level.target).all())

    def button(self, key: str, d: int = 1) -> None:
        if key in ("lvl-", "lvl+"):
            self.level_i = (self.level_i + (1 if key == "lvl+" else -1)) % len(self.files)
            self.load()
        elif key.startswith("dir"):
            i = int(key[3:])
            if d < 0:
                del self.program[i]
                self.goto(self.k - 1 if i < self.k else self.k)
            else:
                r = self.program[i]
                k = (DIRECTION_NAMES.index(r["dir"]) + 1) % len(DIRECTION_NAMES)
                r["dir"] = DIRECTION_NAMES[k]
                self.goto(self.k)
        elif key == "add":
            if len(self.program) < self.cfg["levels"]["max_reactions"]:
                self.program.append(self.new_row())
        elif key == "reset":
            self.goto(0)
        elif key == "step":
            self.goto(self.k + 1)
        elif key == "back":
            self.goto(self.k - 1)
        elif key == "peek":
            self.peek = not self.peek
        elif key == "hidden":
            print(self.level.id, "scrambled by:", end=" ")
            print(", ".join(f"{s.direction} {s.a}|{s.b}" for s in self.level.program))

    def click(self, pos, b: int) -> None:
        d = 1 if b == 1 else -1
        for (i, j), c in self.cells.items():
            if math.hypot(pos[0] - c[0], pos[1] - c[1]) <= self.rb:
                self.program[i]["pair"][j] = (self.program[i]["pair"][j] + d) % self.level.n
                self.goto(self.k)
                return
        for key, r in self.buttons.items():
            if r.collidepoint(pos):
                self.button(key, d)
                return
        for i, r in self.rows.items():  # a row: the picture just after this reaction
            if r.collidepoint(pos):
                self.goto(i + 1)
                return

    # ---- drawing -----------------------------------------------------------------------------
    def icon(self, key: str, kind: str, r, on: bool = False, col=FG) -> None:
        """Button with a drawn icon: left, right, plus, reset, step, peek, or a direction
        (| / \\)."""
        pg = self.pg
        r = pg.Rect(r)
        pg.draw.rect(self.screen, BTN_ON if on else BTN, r, border_radius=6)
        cx, cy = r.center
        s = min(r.w, r.h) * 0.26

        def tri(x: float, sg: int) -> None:
            pts = [(x - sg * 0.8 * s, cy - s), (x + sg * 0.8 * s, cy), (x - sg * 0.8 * s, cy + s)]
            pg.draw.polygon(self.screen, col, pts)

        if kind in ("left", "right"):
            sg = 1 if kind == "right" else -1
            pts = [
                (cx - sg * 0.5 * s, cy - s),
                (cx + sg * 0.5 * s, cy),
                (cx - sg * 0.5 * s, cy + s),
            ]
            pg.draw.lines(self.screen, col, False, pts, 2)
        elif kind == "plus":
            pg.draw.line(self.screen, col, (cx - s, cy), (cx + s, cy), 2)
            pg.draw.line(self.screen, col, (cx, cy - s), (cx, cy + s), 2)
        elif kind == "step":  # one reaction: a triangle against a bar
            tri(cx - 0.2 * s, 1)
            pg.draw.rect(self.screen, col, pg.Rect(cx + 0.7 * s, cy - s, 0.4 * s, 2 * s))
        elif kind == "reset":  # back to the start: bar, then two triangles pointing left
            pg.draw.rect(self.screen, col, pg.Rect(cx - 1.9 * s, cy - s, 0.4 * s, 2 * s))
            tri(cx - 0.5 * s, -1)
            tri(cx + 1.1 * s, -1)
        elif kind == "peek":  # an eye
            box = pg.Rect(cx - 1.5 * s, cy - 0.85 * s, 3 * s, 1.7 * s)
            pg.draw.ellipse(self.screen, col, box, 2)
            pg.draw.circle(self.screen, col, (cx, cy), 0.45 * s)
        else:  # direction of the shared side
            a = {"|": 90, "/": 60, "\\": 120}[kind]
            dx, dy = 1.3 * s * math.cos(math.radians(a)), -1.3 * s * math.sin(math.radians(a))
            pg.draw.line(self.screen, col, (cx - dx, cy - dy), (cx + dx, cy + dy), 4)
        self.buttons[key] = r

    def draw_picture(self) -> None:
        pg = self.pg
        g = self.level.graph
        x, y = np.asarray(g.centers)[:, 0], np.asarray(g.centers)[:, 1]
        m = 40
        k = min(
            (self.canvas.w - 2 * m) / (x.max() - x.min() + math.sqrt(3)),
            (self.canvas.h - 2 * m) / (y.max() - y.min() + 2),
        )
        ox = self.canvas.centerx - k * (x.max() + x.min()) / 2
        oy = self.canvas.centery - k * (y.max() + y.min()) / 2
        col = self.level.target if self.peek else self.col
        for i in range(g.n_cells):
            c = (ox + k * x[i], oy + k * y[i])
            pg.draw.polygon(self.screen, self.pal[int(col[i])], hex_points(c, k * 0.96))
        if self.moved is not None and not self.peek:  # the pairs swapped by the marked row
            for p, q in zip(*self.moved, strict=True):
                a = (ox + k * x[p], oy + k * y[p])
                b = (ox + k * x[q], oy + k * y[q])
                pg.draw.line(self.screen, (20, 20, 24), a, b, 5)
                pg.draw.line(self.screen, FG, a, b, 2)
        if self.solved() or self.peek:
            box = pg.Rect(
                ox + k * x.min() - k - 8,
                oy + k * y.min() - k - 8,
                k * (x.max() - x.min()) + 2 * k + 16,
                k * (y.max() - y.min()) + 2 * k + 16,
            )
            pg.draw.rect(self.screen, MID if self.peek else ACCENT, box, 3, border_radius=8)

    def draw_program(self, x: int, y: int, w: int, bottom: int) -> None:
        pg = self.pg
        n = len(self.program)
        rh = min(58.0, (bottom - y) / max(1, n + 1))
        rb = self.rb = min(13.0, rh * 0.25)
        self.cells, self.rows = {}, {}
        for i, r in enumerate(self.program):
            top = y + i * rh
            cy = top + rh / 2
            box = pg.Rect(x - 6, top + 2, w + 12, rh - 4)
            self.rows[i] = box
            if i < self.k:  # applied; the last one applied is the step you are looking at
                pg.draw.rect(self.screen, ROW, box, border_radius=6)
            if i == self.k - 1:
                pg.draw.rect(self.screen, ACCENT, box, 2, border_radius=6)
            side = min(34, rh - 12)
            self.icon(f"dir{i}", r["dir"], (x, cy - side / 2, side, side))
            off = self.level.grid.offset(r["dir"]) * rb * 1.04
            cx = x + side + (w - side) / 2
            for j, sg in ((0, -0.5), (1, 0.5)):
                c = (cx + sg * off[0], cy + sg * off[1])
                pg.draw.polygon(self.screen, self.pal[r["pair"][j]], hex_points(c, rb))
                pg.draw.polygon(self.screen, MID, hex_points(c, rb), 1)
                self.cells[(i, j)] = c
        if n < self.cfg["levels"]["max_reactions"]:
            cy = y + n * rh + rh / 2
            side = min(28, rh - 14)
            self.icon("add", "plus", (x, cy - side / 2, side, side))

    def draw(self) -> None:
        pg = self.pg
        self.screen.fill(BG)
        self.draw_picture()
        pg.draw.rect(self.screen, PANEL, (self.canvas.right, 0, 9999, self.canvas.h))
        self.buttons = {}
        x, w = self.px, self.screen.get_width() - self.px - 12
        H = self.screen.get_height()
        y = 14
        t = self.font.render(self.level.title or self.level.id, True, FG)
        self.screen.blit(t, t.get_rect(center=(x + w // 2, y + 15)))
        self.icon("lvl-", "left", (x, y, 30, 30))
        self.icon("lvl+", "right", (x + w - 30, y, 30, 30))
        y += 46
        bar = 48
        self.draw_program(x, y, w, H - bar - 28)
        ty = H - bar - 14
        bw = (w - 2 * 8) / 3
        self.icon("reset", "reset", (x, ty, bw, bar))
        left = self.k < len(self.program)
        self.icon("step", "step", (x + (bw + 8), ty, bw, bar), col=FG if left else BTN_ON)
        self.icon("peek", "peek", (x + 2 * (bw + 8), ty, bw, bar), on=self.peek)

    # ---- main loop ---------------------------------------------------------------------------
    def loop(self, smoke: bool = False) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        keymap = {
            pg.K_RIGHT: "step",
            pg.K_SPACE: "step",
            pg.K_LEFT: "back",
            pg.K_r: "reset",
            pg.K_t: "peek",
            pg.K_h: "hidden",
            pg.K_UP: "lvl-",
            pg.K_DOWN: "lvl+",
        }
        running = True
        while running:
            clock.tick(self.cfg["window"]["fps"])
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
                self.smoke()
                running = False
            self.draw()
            pg.display.flip()

    def smoke(self) -> None:
        """Headless checks, by clicks: every level starts scrambled; the scrambling list
        written in reverse order recovers it once its last row is picked; picking an earlier
        row shows that step; an edit changes the picture at the step shown and undoing the
        edit restores it. Saves two screenshots per level (a middle step, recovered)."""
        out = ROOT / "runs" / "shift_toy"
        out.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        for _ in self.files:
            lv = self.level
            n = len(lv.program)
            assert not self.solved() and self.k == 0
            self.draw()
            for i, s in enumerate(reversed(lv.program)):  # write the way back with clicks
                while self.program[i]["dir"] != s.direction:
                    self.click(self.buttons[f"dir{i}"].center, 1)
                for j, c in enumerate((s.a, s.b)):
                    while self.program[i]["pair"][j] != c:
                        self.click(self.cells[(i, j)], 1)
                    self.draw()
            assert self.k == 0 and (self.col == self.start).all()
            self.click(self.buttons["step"].center, 1)
            assert self.k == 1 and self.moved is not None and len(self.moved[0]) > 0, lv.id
            mid = (n + 1) // 2
            self.click((self.rows[mid - 1].right - 4, self.rows[mid - 1].centery), 1)
            assert self.k == mid
            self.draw()
            self.pg.image.save(self.screen, str(out / f"{lv.id}_step{mid}_{stamp}.png"))
            seen = self.col.copy()
            self.click(self.cells[(mid - 1, 0)], 1)  # an edit acts on the step shown ...
            assert self.k == mid
            if lv.n > 2:
                assert (self.col != seen).any(), lv.id
            self.click(self.cells[(mid - 1, 0)], 3)  # ... and undoing it restores the picture
            assert (self.col == seen).all()
            self.click((self.rows[n - 1].right - 4, self.rows[n - 1].centery), 1)
            assert self.solved() and self.k == n, lv.id
            self.click(self.buttons["step"].center, 1)  # the list is done: nothing more
            assert self.solved() and self.k == n
            self.draw()
            self.pg.image.save(self.screen, str(out / f"{lv.id}_recovered_{stamp}.png"))
            self.click(self.buttons["add"].center, 1)
            self.draw()
            assert len(self.program) == n + 1 and self.k == n
            self.click(self.buttons[f"dir{n}"].center, 3)
            assert len(self.program) == n and self.solved()
            self.click(self.buttons["reset"].center, 1)
            assert self.k == 0 and not self.solved()
            self.button("lvl+")
            self.draw()
        print(f"smoke: {len(self.files)} levels recovered going forward; screenshots in", end=" ")
        print(out.relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", default=None, help="level id (file name in levels/shift)")
    ap.add_argument("--smoke", action="store_true", help="headless checks and screenshots")
    a = ap.parse_args()
    if a.smoke:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    pygame.init()
    toy = ShiftToy(load_cfg(), a.level)
    toy.loop(a.smoke)
    pygame.quit()


if __name__ == "__main__":
    main()
