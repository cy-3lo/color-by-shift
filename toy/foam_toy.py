"""Foam toy: click to shift. A picture on a foam was scrambled by a few moves; put it back.

    .venv/bin/python toy/foam_toy.py [--level F01]

Point between two neighbouring cells of different colours and click: they swap their
colours, and so does every pair of the same group that holds these two colours (model:
mosaic_model.foam). While you point, the pairs of that group are drawn as thin links and the
ones that would swap as bright bars. A move is its own inverse: the same click again takes
it back (and the count goes down).

The foam follows the shapes of the picture to recover (a pentagon, a star ...): their
outlines are made of cell sides, so the eye sees the shapes in the paving; which colour goes
where is not given. When the picture is recovered the lines between cells vanish and only
the shapes remain, with their outline ([picture] in toy/foam.toml: cell_lines, outline_px
while playing, win_outline_px once recovered, show_target for a small coloured copy).

Strip (right): < level > (Up / Down), the number of moves over the number the picture was
scrambled with, and three buttons: undo (Z), redo (Y), back to the scrambled picture (R).
A level made from a painting (tools/make_painting_level.py) ends with the painting itself
fading in, with its painter and title.
The frame lights up when the picture is recovered. H prints the scrambling moves in the
terminal. Esc quits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))

from mosaic_model.foam import load_foam_level  # noqa: E402

BG = (28, 28, 34)
LINE = (16, 16, 20)  # outline of the picture and lines between cells
PANEL = (38, 38, 46)
FG = (236, 233, 224)
MID = (150, 148, 160)
BTN = (58, 58, 70)
DIM = (92, 90, 104)  # icon of a button that has nothing to do
ACCENT = (240, 180, 90)


def load_cfg() -> dict:
    return tomllib.loads((ROOT / "toy" / "foam.toml").read_text())


def rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[k : k + 2], 16) for k in (0, 2, 4))


class FoamToy:
    def __init__(self, cfg: dict, level: str | None = None):
        import pygame

        self.pg = pygame
        self.cfg = cfg
        w, h = cfg["window"]["width"], cfg["window"]["height"]
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Mosaic shift")
        self.font = pygame.font.Font(None, 26)
        self.small = pygame.font.Font(None, 21)
        self.canvas = pygame.Rect(0, 0, w - cfg["window"]["strip"], h)
        self.px = self.canvas.right + 12
        self.files = sorted((ROOT / cfg["levels"]["folder"]).glob("*.json"))
        ids = [f.stem for f in self.files]
        self.level_i = ids.index(level) if level else 0
        self.buttons: dict[str, object] = {}
        self.rasters: dict = {}
        self.hover: tuple[int, int] | None = None
        self.load()

    # ---- level and moves ---------------------------------------------------------------------
    def load(self) -> None:
        self.level = load_foam_level(self.files[self.level_i], ROOT)
        foam = self.level.foam
        self.pal = np.array([rgb(c) for c in self.level.colours] + [LINE], dtype=np.uint8)
        m = self.cfg["window"]["margin"]
        pw = min(self.canvas.w - 2 * m, int((self.canvas.h - 2 * m) * foam.aspect))
        ph = round(pw / foam.aspect)
        self.rect = self.pg.Rect(0, 0, pw, ph)
        self.rect.center = self.canvas.center
        key = (foam.sites.tobytes(), pw, ph)
        if key not in self.rasters:
            lab = foam.raster(pw, ph)
            edge = np.zeros(lab.shape, bool)  # thin dark line between cells
            edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
            edge[:-1, :] |= lab[:-1, :] != lab[1:, :]
            self.rasters[key] = (lab, edge)
        self.labels, self.border = self.rasters[key]
        pic = self.cfg["picture"]
        self.outline = self.outline_mask(pic["outline_px"])  # while playing (0 = none)
        self.win_outline = self.outline_mask(pic["win_outline_px"])  # once recovered
        self.surf = self.pg.Surface((pw, ph))
        self.site_px = foam.sites / (foam.aspect, 1.0) * (pw, ph) + self.rect.topleft
        tw = self.cfg["window"]["strip"] - 24
        self.thumb = self.pg.Surface((tw, round(tw / foam.aspect)))
        self.thumb_lab = foam.raster(tw, round(tw / foam.aspect))
        self.pg.surfarray.blit_array(
            self.thumb, self.pal[self.level.target][self.thumb_lab].swapaxes(0, 1)
        )
        self.painting = None  # the reward of a painting level, at the size of the picture
        rw = self.level.reward
        if rw and rw.get("image"):
            im = self.pg.image.load(str(ROOT / rw["image"]))
            W, H = im.get_size()
            x0, y0, x1, y1 = rw.get("crop") or (0.0, 0.0, 1.0, 1.0)
            box = self.pg.Rect(
                round(x0 * W), round(y0 * H), round((x1 - x0) * W), round((y1 - y0) * H)
            )
            ax, ay = rw.get("anchor") or (0.5, 0.5)
            if box.w / box.h > foam.aspect:  # the largest window of the foam's format
                w2 = round(box.h * foam.aspect)
                box = self.pg.Rect(box.x + round((box.w - w2) * ax), box.y, w2, box.h)
            else:
                h2 = round(box.w / foam.aspect)
                box = self.pg.Rect(box.x, box.y + round((box.h - h2) * ay), box.w, h2)
            self.painting = self.pg.transform.smoothscale(im.subsurface(box), (pw, ph))
        self.flat = self.pg.Surface((pw, ph))
        self.start = self.level.scrambled()
        self.reset()

    def outline_mask(self, px: int) -> np.ndarray:
        """Pixels of the outline of the picture to recover (sides between cells of different
        target colours), about px pixels thick; empty if px <= 0."""
        lab, tgt = self.labels, self.level.target
        line = np.zeros(lab.shape, bool)
        if px <= 0:
            return line
        mh = tgt[lab[:, :-1]] != tgt[lab[:, 1:]]
        mv = tgt[lab[:-1, :]] != tgt[lab[1:, :]]
        line[:, :-1] |= mh
        line[:, 1:] |= mh
        line[:-1, :] |= mv
        line[1:, :] |= mv
        for _ in range(max(0, px // 2 - 1)):  # thicken
            grow = line.copy()
            grow[:, 1:] |= line[:, :-1]
            grow[:, :-1] |= line[:, 1:]
            grow[1:, :] |= line[:-1, :]
            grow[:-1, :] |= line[1:, :]
            line = grow
        return line

    def reset(self) -> None:
        self.col = self.start.copy()
        self.history: list[tuple[int, int, int]] = []  # moves played, in order
        self.future: list[tuple[int, int, int]] = []  # moves taken back, for redo
        self.solved_at: float | None = None

    def after_move(self) -> None:
        """Start (or cancel) the reward when the picture becomes (or stops being) recovered."""
        if not self.solved():
            self.solved_at = None
        elif self.solved_at is None:
            self.solved_at = time.time()

    def undo(self) -> None:
        if self.history:
            mv = self.history.pop()
            self.col = self.level.foam.move(self.col, *mv)  # a move is its own inverse
            self.future.append(mv)
            self.after_move()

    def redo(self) -> None:
        if self.future:
            mv = self.future.pop()
            self.col = self.level.foam.move(self.col, *mv)
            self.history.append(mv)
            self.after_move()

    def solved(self) -> bool:
        return bool((self.col == self.level.target).all())

    def pair_at(self, pos) -> tuple[int, int] | None:
        """The cell under the pointer and its neighbour nearest to the pointer."""
        if not self.rect.collidepoint(pos):
            return None
        d = np.hypot(self.site_px[:, 0] - pos[0], self.site_px[:, 1] - pos[1])
        u, v = (int(k) for k in np.argsort(d)[:2])
        return (u, v) if self.level.foam.group_of(u, v) is not None else None

    def move_at(self, pair: tuple[int, int]) -> tuple[int, int, int] | None:
        """The move a pair stands for: (group, colour, colour), None if nothing would move."""
        u, v = pair
        a, b = sorted((int(self.col[u]), int(self.col[v])))
        return (self.level.foam.group_of(u, v), a, b) if a != b else None

    def play(self, pair: tuple[int, int]) -> bool:
        mv = self.move_at(pair)
        if mv is None:
            return False
        if self.history and self.history[-1] == mv:
            self.undo()  # the same move again takes it back
            return True
        self.col = self.level.foam.move(self.col, *mv)
        self.history.append(mv)
        self.future = []
        self.after_move()
        return True

    def button(self, key: str) -> None:
        if key in ("lvl-", "lvl+"):
            self.level_i = (self.level_i + (1 if key == "lvl+" else -1)) % len(self.files)
            self.load()
        elif key == "reset":
            self.reset()
        elif key == "undo":
            self.undo()
        elif key == "redo":
            self.redo()
        elif key == "hidden":
            moves = ", ".join(f"group {g} {a}|{b}" for g, a, b in self.level.scramble)
            print(self.level.id, "scrambled by:", moves)

    def click(self, pos) -> None:
        for key, r in self.buttons.items():
            if r.collidepoint(pos):
                self.button(key)
                return
        pair = self.pair_at(pos)
        if pair is not None:
            self.play(pair)

    # ---- drawing -----------------------------------------------------------------------------
    def icon(self, key: str, kind: str, r, col=FG) -> None:
        pg = self.pg
        r = pg.Rect(r)
        pg.draw.rect(self.screen, BTN, r, border_radius=6)
        cx, cy = r.center
        s = min(r.w, r.h) * 0.26
        if kind in ("undo", "redo"):  # an arrow that turns back: left = undo, right = redo
            sg = -1 if kind == "undo" else 1
            box = pg.Rect(cx - 1.1 * s, cy - 0.7 * s, 2.2 * s, 2.0 * s)
            a0, a1 = (0.0, 3.6) if sg < 0 else (-0.45, 3.14)
            pg.draw.arc(self.screen, col, box, a0, a1, 3)
            tip = (cx + sg * 1.1 * s, cy + 0.3 * s)
            pts = [(tip[0] - 0.7 * s, tip[1] - 0.2 * s), (tip[0] + 0.7 * s, tip[1] - 0.2 * s)]
            pg.draw.polygon(self.screen, col, [*pts, (tip[0], tip[1] + 0.8 * s)])
        elif kind in ("left", "right"):
            sg = 1 if kind == "right" else -1
            pts = [
                (cx - sg * 0.5 * s, cy - s),
                (cx + sg * 0.5 * s, cy),
                (cx - sg * 0.5 * s, cy + s),
            ]
            pg.draw.lines(self.screen, FG, False, pts, 2)
        else:  # reset: a bar, then two triangles pointing left
            pg.draw.rect(self.screen, FG, pg.Rect(cx - 1.9 * s, cy - s, 0.4 * s, 2 * s))
            for x in (cx - 0.5 * s, cx + 1.1 * s):
                pts = [(x + 0.8 * s, cy - s), (x - 0.8 * s, cy), (x + 0.8 * s, cy + s)]
                pg.draw.polygon(self.screen, FG, pts)
        self.buttons[key] = r

    def draw(self) -> None:
        pg = self.pg
        self.screen.fill(BG)
        idx = self.col[self.labels]
        if self.cfg["picture"]["cell_lines"]:
            idx[self.border] = len(self.pal) - 1
        idx[self.outline] = len(self.pal) - 1
        pg.surfarray.blit_array(self.surf, self.pal[idx].swapaxes(0, 1))
        self.screen.blit(self.surf, self.rect.topleft)
        shown = 0.0  # how much of the painting is shown, 0..1
        if self.solved_at is not None:
            # the reward: the lines between cells fade and only the shapes remain; then, for
            # a painting, the painting itself appears with its painter and title
            rw = self.cfg["reward"]
            t = time.time() - self.solved_at
            flat = self.col[self.labels]
            flat[self.win_outline] = len(self.pal) - 1
            pg.surfarray.blit_array(self.flat, self.pal[flat].swapaxes(0, 1))
            self.flat.set_alpha(int(255 * min(1.0, t / rw["lines_s"])))
            self.screen.blit(self.flat, self.rect.topleft)
            if self.painting is not None:
                shown = min(1.0, max(0.0, (t - rw["lines_s"] - rw["hold_s"]) / rw["fade_s"]))
                self.painting.set_alpha(int(255 * shown))
                self.screen.blit(self.painting, self.rect.topleft)
        if self.hover is not None:  # the group of the pair pointed at, and what would swap
            u, v = self.hover
            foam = self.level.foam
            g = foam.group_of(u, v)
            for p, q in zip(*foam.pairs[g], strict=True):
                pg.draw.line(self.screen, (20, 20, 24), self.site_px[p], self.site_px[q], 3)
                pg.draw.line(self.screen, MID, self.site_px[p], self.site_px[q], 1)
            mv = self.move_at(self.hover)
            if mv is not None:
                for p, q in zip(*foam.swapping(self.col, *mv), strict=True):
                    pg.draw.line(self.screen, (20, 20, 24), self.site_px[p], self.site_px[q], 7)
                    pg.draw.line(self.screen, FG, self.site_px[p], self.site_px[q], 3)
            for k in (u, v):
                pg.draw.circle(self.screen, FG, self.site_px[k], 5)
                pg.draw.circle(self.screen, (20, 20, 24), self.site_px[k], 5, 1)
        if self.solved():
            pg.draw.rect(self.screen, ACCENT, self.rect.inflate(14, 14), 3, border_radius=8)
        pg.draw.rect(self.screen, PANEL, (self.canvas.right, 0, 9999, self.canvas.h))
        self.buttons = {}
        x, w = self.px, self.screen.get_width() - self.px - 12
        H = self.screen.get_height()
        t = self.font.render(self.level.title or self.level.id, True, FG)
        self.screen.blit(t, t.get_rect(center=(x + w // 2, 29)))
        self.icon("lvl-", "left", (x, 14, 30, 30))
        self.icon("lvl+", "right", (x + w - 30, 14, 30, 30))
        y = 64
        if self.cfg["picture"]["show_target"]:
            self.screen.blit(self.thumb, (x, y))
            pg.draw.rect(self.screen, MID, self.thumb.get_rect(topleft=(x, y)), 1)
            y += self.thumb.get_height() + 16
        if shown > 0 and self.level.reward:  # painter and title, with the painting
            for text, col in (
                (self.level.reward.get("artist", ""), FG),
                (self.level.reward.get("title", ""), MID),
            ):
                line = ""
                for word in [*text.split(), None]:
                    test = f"{line} {word}".strip() if word else line
                    if word is None or self.small.size(test)[0] > w:
                        if line:
                            t = self.small.render(line, True, col)
                            t.set_alpha(int(255 * shown))
                            self.screen.blit(t, (x, y))
                            y += self.small.get_linesize()
                        line = word or ""
                    else:
                        line = test
                y += 8
        n, par = len(self.history), self.level.par
        good = self.solved() and n <= par
        num = self.font.render(f"{n} / {par}", True, ACCENT if good else FG)
        self.screen.blit(num, num.get_rect(center=(x + w // 2, H - 148)))
        half = (w - 8) // 2
        self.icon("undo", "undo", (x, H - 118, half, 48), FG if self.history else DIM)
        self.icon("redo", "redo", (x + half + 8, H - 118, half, 48), FG if self.future else DIM)
        self.icon("reset", "reset", (x, H - 62, w, 48))

    # ---- main loop ---------------------------------------------------------------------------
    def loop(self, smoke: bool = False) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        keymap = {
            pg.K_r: "reset",
            pg.K_z: "undo",
            pg.K_y: "redo",
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
                elif ev.type == pg.MOUSEMOTION:
                    self.hover = self.pair_at(ev.pos)
                elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                    self.click(ev.pos)
                    self.hover = self.pair_at(ev.pos)
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

    def point_for(self, mv: tuple[int, int, int]) -> tuple[float, float]:
        """A pixel whose pair stands for the move mv (for the headless run)."""
        for u, v in zip(*self.level.foam.swapping(self.col, *mv), strict=True):
            for f in (0.5, 0.45, 0.55, 0.4, 0.6):
                pos = tuple(self.site_px[u] * f + self.site_px[v] * (1 - f))
                pair = self.pair_at(pos)
                if pair is not None and self.move_at(pair) == mv:
                    return pos
        raise AssertionError(f"no pixel found for move {mv}")

    def smoke(self) -> None:
        """Headless checks, by clicks: every level starts scrambled; the same click twice
        takes a move back; the scrambling moves clicked in reverse order recover the picture
        in par moves; reset brings the scrambled picture back. Two screenshots per level
        (pointing at the first move, recovered)."""
        out = ROOT / "runs" / "foam_toy"
        out.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        for _ in self.files:
            lv = self.level
            assert not self.solved() and not self.history
            way_back = lv.scramble[::-1]
            pos = self.point_for(way_back[0])
            self.hover = self.pair_at(pos)
            self.draw()
            self.pg.image.save(self.screen, str(out / f"{lv.id}_point_{stamp}.png"))
            self.click(pos)
            assert len(self.history) == 1 and (self.col != self.start).any()
            self.click(pos)  # the same click again: taken back
            assert not self.history and (self.col == self.start).all()
            for mv in way_back:
                self.click(self.point_for(mv))
            assert self.solved() and len(self.history) == lv.par, lv.id
            self.click(self.buttons["undo"].center)
            assert not self.solved() and len(self.history) == lv.par - 1 and self.solved_at is None
            self.click(self.buttons["redo"].center)
            assert self.solved() and len(self.history) == lv.par and not self.future
            self.solved_at -= 60  # the end of the reward
            self.hover = None
            self.draw()
            self.pg.image.save(self.screen, str(out / f"{lv.id}_recovered_{stamp}.png"))
            self.click(self.buttons["reset"].center)
            assert not self.history and (self.col == self.start).all()
            self.button("lvl+")
            self.draw()
        print(
            f"smoke: {len(self.files)} levels recovered at par by clicks; screenshots in", end=" "
        )
        print(out.relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", default=None, help="level id (file name in levels/foam)")
    ap.add_argument("--smoke", action="store_true", help="headless checks and screenshots")
    a = ap.parse_args()
    if a.smoke:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    pygame.init()
    toy = FoamToy(load_cfg(), a.level)
    toy.loop(a.smoke)
    pygame.quit()


if __name__ == "__main__":
    main()
