"""Minimal pygame-ce viewer for a level (Phase 1).

    python toy/viewer.py levels/T02_rps.json [--seed S] [--shuffled]
    python toy/viewer.py levels/T02_rps.json --smoke 120   # headless test, saves a PNG

Click two unlocked blocks to swap them (only while running, not while paused).
Buttons or keys: Space pause, 1/2/3/4 speed x1/4, x1, x4, x16, R restart, S screenshot, Esc quit.
Zoom: mouse wheel or +/- (about the cursor), drag the grid to pan, 0 to see the whole world.
--n N plays on an N x N rectangular torus instead of the level's world.
--world torus:64 | open:64 | hextorus:37 | hexagon:37 overrides the world shape
(hextorus = hexagon with opposite sides glued; hexagon = open hexagon).
Swaps are logged to runs/viewer/<level>_<time>/session.json.

Sandbox (design tool): python toy/viewer.py --sandbox [level.json] [--colours 5] [--start 3]
  Mode: edit   click a block to paint it with the chosen colour, right-click to lock it,
               click the shape / k buttons of a slot to change it (right-click: backwards;
               k = 0 switches the slot off). Works while running or paused.
  Mode: play   the normal rules (swap two unlocked blocks while running).
  Random start draws a new uniform random mosaic; Save level writes levels/sandbox/S*.json.
  Terrain (edit mode): choose Wall, Fixed (in the paint colour) or Erase, then click or drag
  on the grid (right button erases); Brush sets the radius; Blocks gives the grid back to
  panning. Walls are holes in the world; fixed cells keep their colour (dot) forever.
The lower chart is the patch size S (area-weighted mean size of same-colour connected
patches, cells, log scale): total in the text colour, each colour in its own.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "toy"))

from mosaic_model import GoalTracker, Level, Sim, load_level, make_sim  # noqa: E402
from mosaic_model.graph import SQRT3  # noqa: E402
from mosaic_model.level import terrain_to_json  # noqa: E402
from mosaic_model.observables import sim_patch_sizes  # noqa: E402
from mosaic_model.raster import fit_size, render_rgb, view_pixel_map, world_centre  # noqa: E402
from mosaic_model.reactions import ARITY, Block, Reaction  # noqa: E402
from palette import (  # noqa: E402
    colour_names,
    glyphs,
    luminance,
    menu_names,
    palette_rgb,
    ui_colours,
)

RB = 16.0  # circumradius of an equation block (hexagon), px
GAP = 1.08  # spacing factor between neighbouring blocks (1 = touching)
DARK = (25, 25, 30)  # outlines and glyphs drawn on top of cell colours
EQ_DX = 64  # x offset of the first equation block from the panel edge (room for the meter)
SHAPE_CYCLE = [("cell", True), ("edge", True), ("vertex", True), ("vertex", False)]
SHAPE_LABEL = {("cell", True): "cell", ("edge", True): "edge", ("vertex", True): "vertex cw"}
SHAPE_LABEL[("vertex", False)] = "vertex +-"


def load_cfg() -> dict:
    return tomllib.loads((ROOT / "toy" / "viewer.toml").read_text())


class Viewer:
    def __init__(
        self, level: Level, cfg: dict, seed: int | None, shuffled: bool, sandbox: bool = False
    ) -> None:
        import pygame

        self.pg = pygame
        self.level = level
        self.sandbox = sandbox
        self.edit = sandbox  # sandbox starts in edit mode
        self.paint = 0  # colour painted in edit mode
        self.tool: str | None = None  # terrain tool in edit mode: wall | fix | erase | None
        self.brush = 0  # terrain brush radius (cells)
        self.painting = 0  # mouse button held while painting terrain (0 = none)
        self.terrain_dirty = False
        self.arrow_style = cfg["arrows"]["style"]
        self.cfg = cfg
        self.shuffled = shuffled
        self.seed = level.start.get("seed", 0) if seed is None else seed
        w, h = cfg["window"]["width"], cfg["window"]["height"]
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(f"Mosaic Shift viewer: {level.id}")
        # pygame's bundled font: no system font lookup (fc-list can hang on macOS)
        self.font = pygame.font.Font(None, 20)
        self.small = pygame.font.Font(None, 17)
        self.big = pygame.font.Font(None, 36)
        pal = cfg["palette"]
        self.palette_name = pal["default"]
        self.pal = palette_rgb(cfg, level.palette, self.palette_name)
        self.cnames = colour_names(cfg, level.palette, self.palette_name)
        self.ui = ui_colours(cfg, self.palette_name, level.palette)  # from the palette
        self.bg = self.ui["bg"]
        self.glyphs = glyphs(cfg, level.palette)
        self.show_glyphs = bool(pal.get("show_glyphs", False))
        self.menu_open = False
        fr = level.start.get("fractions", [1.0] * level.palette)
        self.start_n = sum(1 for f in fr if f > 0)  # random start: colours 0 .. start_n - 1
        self.rb = RB  # block radius, shrunk by _fit_rows when many slots do not fit
        self.row_gap = 10.0
        self.row_scale = 1.0

        t0 = time.perf_counter()
        print(f"burn-in: {level.start.get('burn_in_sweeps', 0)} sweeps ...", flush=True)
        self.sim0 = make_sim(level, seed=self.seed, shuffled=shuffled)
        print(f"burn-in done in {time.perf_counter() - t0:.1f} s", flush=True)
        self.start_cols = list(self.sim0.col)
        self.hole = self.sim0.hole.copy()
        self.fixed = self.sim0.fixed.copy()
        self.fixed_col = self.sim0.colours.astype(np.int64)  # colour of each fixed cell
        g = self.sim0.graph
        self._nb = g.neighbours()
        gw = cfg["window"]["grid_width"]
        self.fit = fit_size(g, gw - 20, h - 20)  # whole world visible; periodic worlds tile out
        self.zoom_reset()
        self.grid_rect = pygame.Rect(0, 0, gw, h)
        self.dragging = False
        self.grid_surf = pygame.Surface(self.grid_rect.size)
        self.panel_x = self.grid_rect.right + 20
        self.restarts = 0
        self.session: list[dict] = []
        self.run_dir: Path | None = None
        self.buttons: dict[str, object] = {}
        self.notice = ""
        self.notice_until = 0.0
        self.reset(first=True)

    # ---- state ------------------------------------------------------------------------------
    def reset(self, first: bool = False) -> None:
        if not first:
            self.restarts += 1
            self.session.append({"event": "restart", "wall": time.time()})
        keep = self.sandbox and hasattr(self, "sim")  # sandbox restarts keep the edits
        cols = np.array(self.start_cols)
        cols[self.fixed] = self.fixed_col[self.fixed]
        self.sim = Sim(
            self.sim0.graph,
            self.sim.reactions.copy() if keep else self.level.reaction_set(),
            cols,
            seed=self.seed + 1 + 7919 * self.restarts,
            shuffled=self.shuffled,
            holes=self.hole,
            fixed=self.fixed,
        )
        self.tracker = GoalTracker(self.level.goals, self.level.fail)
        self.hist_t: list[float] = [0.0]
        self.hist_x: list[np.ndarray] = [self.sim.fractions()]
        self.measure_size()
        self.hist_st: list[float] = [0.0]
        self.hist_s: list[tuple[float, np.ndarray]] = [(self.obs["size"], self.obs["size_c"])]
        self.lamps = np.zeros(len(self.sim.fires))
        self.needles = np.zeros(len(self.sim.fires))  # dial position in [0, 1]
        self.last_fires = list(self.sim.fires)
        self.last_t = 0.0
        self.selected: Block | None = None
        self.paused = False
        mults = self.cfg["speed"]["multipliers"]
        self.speed_i = mults.index(1) if 1 in mults else 0
        self.due = 0.0
        self.rate_meas = 0.0

    def swap(self, a: Block, b: Block) -> None:
        self.sim.swap(a, b)
        ev = {
            "event": "swap",
            "sweep": self.sim.sweeps,
            "t": self.sim.t,
            "a": a,
            "b": b,
            "restart": self.restarts,
        }
        self.session.append(ev)
        print(f"swap at sweep {self.sim.sweeps} (t={self.sim.t:.2f}): {a} <-> {b}", flush=True)

    # ---- zoom and pan -------------------------------------------------------------------------
    def zoom_reset(self) -> None:
        self.size = self.fit
        self.centre = world_centre(self.sim0.graph)
        self.map_dirty = True

    def zoom(self, factor: float, at: tuple[int, int] | None = None) -> None:
        """Zoom by `factor` keeping the world point under pixel `at` fixed."""
        gw, h = self.cfg["window"]["grid_width"], self.cfg["window"]["height"]
        new = min(max(self.size * factor, self.fit), self.cfg["grid"]["max_cell_px"])
        if at is not None:
            wx = self.centre[0] + (at[0] - gw / 2) / self.size
            wy = self.centre[1] + (at[1] - h / 2) / self.size
            self.centre = (wx - (at[0] - gw / 2) / new, wy - (at[1] - h / 2) / new)
        self.size = new
        if self.size <= self.fit * 1.0001:
            self.centre = world_centre(self.sim0.graph)
        self.map_dirty = True

    def pan(self, dx: float, dy: float) -> None:
        self.centre = (self.centre[0] - dx / self.size, self.centre[1] - dy / self.size)
        g = self.sim0.graph
        if not g.periodic:  # keep some of an open world in view
            lo, hi = g.centers.min(axis=0), g.centers.max(axis=0)
            self.centre = (
                float(np.clip(self.centre[0], lo[0], hi[0])),
                float(np.clip(self.centre[1], lo[1], hi[1])),
            )
        self.map_dirty = True

    def _remap(self) -> None:
        if self.map_dirty:
            gw, h = self.cfg["window"]["grid_width"], self.cfg["window"]["height"]
            g = self.sim0.graph
            self.pixmap = view_pixel_map(
                g, self.size, gw, h, self.cfg["grid"]["border"], self.centre
            )
            # inner hexagon of each cell (fixed-cell marker) and wall hatching
            dot = self.cfg["terrain"]["fixed_dot"]
            self.inner = view_pixel_map(g, self.size, gw, h, 1.0 - dot, self.centre)
            ys, xs = np.mgrid[0:h, 0:gw]
            self.stripe = ((xs + ys) // self.cfg["terrain"]["wall_stripe_px"]) % 2 == 0
            self.map_dirty = False

    # ---- terrain and patch size ---------------------------------------------------------------
    def measure_size(self) -> None:
        s, sc = sim_patch_sizes(self.sim)
        self.obs = {"size": s, "size_c": sc}

    def paint_terrain(self, pos, erase: bool = False) -> None:
        """Apply the terrain tool to the cells under the brush at screen position pos."""
        x, y = pos[0] - self.grid_rect.x, pos[1] - self.grid_rect.y
        if not (0 <= y < self.pixmap.shape[0] and 0 <= x < self.pixmap.shape[1]):
            return
        i = int(self.pixmap[y, x])
        if i < 0:
            return
        cells, front = {i}, [i]
        for _ in range(self.brush):
            front = [j for k in front for j in self._nb[k] if j not in cells]
            cells.update(front)
        idx = np.fromiter(cells, dtype=np.int64)
        tool = "erase" if erase else self.tool
        if tool == "erase":
            self.hole[idx] = False
            self.fixed[idx] = False
        elif tool == "wall":
            self.hole[idx] = True
            self.fixed[idx] = False
        elif tool == "fix":
            self.hole[idx] = False
            self.fixed[idx] = True
            self.fixed_col[idx] = self.paint
            for j in idx.tolist():
                self.sim.col[j] = self.paint
        self.terrain_dirty = True

    def apply_terrain(self) -> None:
        """Push painted terrain to the running sim (once per frame)."""
        if not self.terrain_dirty:
            return
        self.sim.set_terrain(self.hole, self.fixed)
        self.terrain_dirty = False
        self.measure_size()
        self.session.append(
            {
                "event": "terrain",
                "t": self.sim.t,
                "holes": int(self.hole.sum()),
                "fixed": int(self.fixed.sum()),
            }
        )

    def move_needles(self, dt_wall: float) -> None:
        """Needles swing toward the smoothed firing rate (log dial), with a short lag."""
        lc = self.cfg["lamps"]
        lo, hi = math.log10(lc["lo"]), math.log10(lc["hi"])
        target = np.clip((np.log10(np.maximum(self.lamps, 1e-12)) - lo) / (hi - lo), 0.0, 1.0)
        a = 1.0 - math.exp(-dt_wall / lc["needle_tau"]) if dt_wall > 0 else 1.0
        self.needles += a * (target - self.needles)

    def step(self, dt_wall: float) -> None:
        """Advance base_time_per_s * multiplier sim time units per wall second, in chunks of
        attempts (partial sweeps), within a per-frame time budget."""
        if self.paused or self.sim.n_elig == 0 or self.sim.K <= 0:
            return  # paused, or nothing can fire (every reaction off)
        sp = self.cfg["speed"]
        sim = self.sim
        rate = sp["base_time_per_s"] * sp["multipliers"][self.speed_i]
        self.due = min(self.due + dt_wall * rate, 0.25 * rate + 0.1)  # cap the backlog
        t_end = time.perf_counter() + sp["frame_budget_ms"] * 1e-3
        t_start = sim.t
        chunk = sp["chunk_attempts"]
        while self.due > 0 and time.perf_counter() < t_end:
            n = min(chunk, max(1, int(self.due * sim.K * sim.n_elig)))
            t0 = sim.t
            sim.attempts(n)
            self.due -= sim.t - t0
            self.hist_t.append(sim.t)
            self.hist_x.append(sim.fractions())
            self.tracker.update(sim.t, sim.counts(), self.obs)
        if sim.t > t_start:  # patch size: once per frame (about 1 ms on 16 000 cells)
            self.measure_size()
            self.hist_st.append(sim.t)
            self.hist_s.append((self.obs["size"], self.obs["size_c"]))
            self.tracker.update(sim.t, sim.counts(), self.obs)
        if dt_wall > 0:
            self.rate_meas = 0.9 * self.rate_meas + 0.1 * (sim.t - t_start) / dt_wall
        # lamps: firing rate per cell per unit time, exponentially smoothed
        dt = self.sim.t - self.last_t
        if dt > 0:
            df = np.array(self.sim.fires) - np.array(self.last_fires)
            inst = df / (dt * self.sim.graph.n_cells)
            a = 1.0 - math.exp(-dt / self.cfg["lamps"]["tau"])
            self.lamps += a * (inst - self.lamps)
            self.last_fires = list(self.sim.fires)
            self.last_t = self.sim.t
        # trim history
        w = self.cfg["chart"]["window"] * 1.2
        while len(self.hist_t) > 2 and self.hist_t[1] < self.sim.t - w:
            self.hist_t.pop(0)
            self.hist_x.pop(0)
        while len(self.hist_st) > 2 and self.hist_st[1] < self.sim.t - w:
            self.hist_st.pop(0)
            self.hist_s.pop(0)

    # ---- drawing ------------------------------------------------------------------------------
    def draw(self) -> None:
        pg = self.pg
        self.screen.fill(self.bg)
        self._remap()
        rgb = render_rgb(self.pixmap, self.sim.colours, self.pal, self.bg)
        self._draw_terrain(rgb)
        pg.surfarray.blit_array(self.grid_surf, rgb.swapaxes(0, 1))
        self.screen.blit(self.grid_surf, self.grid_rect)
        y = self._draw_header(10)
        y = self._draw_equations(y + 10)
        self._draw_chart(y + 10)
        self._draw_status()
        self._draw_menu()

    def _draw_terrain(self, rgb: np.ndarray) -> None:
        """Walls hatched in two interface tones; a dot on fixed cells, dark on light colours
        and light on dark ones."""
        if self.hole.any():
            h = np.append(self.hole, False)[self.pixmap]
            ta, tb = (self.ui[k] for k in self.cfg["terrain"]["wall_tones"])
            rgb[h & self.stripe] = ta
            rgb[h & ~self.stripe] = tb
        if self.fixed.any():
            f = np.append(self.fixed, False)[self.inner]
            if f.any():
                mark = np.array(
                    [self.ui["bg"] if luminance(c) > 0.3 else self.ui["fg"] for c in self.pal],
                    dtype=np.uint8,
                )
                rgb[f] = mark[self.sim.colours[self.inner[f]]]

    def _text(self, s: str, pos, font=None, col=None) -> int:
        surf = (font or self.font).render(s, True, col or self.ui["fg"])
        self.screen.blit(surf, pos)
        return surf.get_height()

    def _draw_header(self, y: int) -> int:
        lv, sim = self.level, self.sim
        x = self.panel_x
        y += self._text(f"{lv.id}  {lv.title}", (x, y))
        mode = "SHUFFLED (well-mixed)" if self.shuffled else "grid"
        mult = self.cfg["speed"]["multipliers"][self.speed_i]
        state = "PAUSED" if self.paused else _mult_label(mult)
        zoom = f"zoom {self.size / self.fit:.1f}x   " if self.size > self.fit * 1.001 else ""
        y += self._text(
            f"t = {sim.t:7.1f}   sweep {sim.sweeps}   {state}   {zoom}"
            f"{self.rate_meas:4.2f} t/s   K={sim.K:g}   {mode}",
            (x, y + 2),
            self.small,
            self.ui["mid"],
        )
        return self._draw_buttons(y + 8)

    def _draw_buttons(self, y: int) -> int:
        pg = self.pg
        x = self.panel_x
        mults = self.cfg["speed"]["multipliers"]
        labels = [("pause", "Play" if self.paused else "Pause", 70)]
        labels += [(f"speed{i}", _mult_label(m), 48) for i, m in enumerate(mults)]
        labels += [("restart", "Restart", 70)]
        self.buttons = {}
        for key, text, w in labels:
            r = pg.Rect(x, y, w, 26)
            active = (key == "pause" and self.paused) or key == f"speed{self.speed_i}"
            pg.draw.rect(
                self.screen, self.ui["btn_on"] if active else self.ui["btn"], r, border_radius=5
            )
            pg.draw.rect(self.screen, self.ui["mid"], r, 1, border_radius=5)
            t = self.font.render(text, True, self.ui["fg"])
            self.screen.blit(t, t.get_rect(center=r.center))
            self.buttons[key] = r
            x += w + 8
        # second row: palette menu and symbols toggle
        x = self.panel_x
        y2 = y + 32
        for key, text, w in (
            ("palette", f"Palette: {self.palette_name}", 170),
            ("glyphs", f"Symbols: {'on' if self.show_glyphs else 'off'}", 110),
            ("arrows", f"Arrows: {self.arrow_style}", 120),
        ):
            r = pg.Rect(x, y2, w, 26)
            pg.draw.rect(self.screen, self.ui["btn"], r, border_radius=5)
            pg.draw.rect(self.screen, self.ui["mid"], r, 1, border_radius=5)
            t = self.font.render(text, True, self.ui["fg"])
            self.screen.blit(t, t.get_rect(center=r.center))
            self.buttons[key] = r
            x += w + 8
        if self.sandbox:
            y2 = self._draw_sandbox_bar(y2 + 32)
        x = self.panel_x
        if self.notice and time.time() < self.notice_until:
            self._text(self.notice, (x, y2 + 32), self.small, (240, 180, 90))
        elif self.paused and not self.edit:
            self._text("paused: no switching", (x, y2 + 32), self.small, self.ui["mid"])
        elif self.sim.n_elig == 0 or self.sim.K <= 0:
            self._text("every reaction is off (k = 0)", (x, y2 + 32), self.small, self.ui["mid"])
        return y2 + 50

    def _button(self, key: str, text: str, x: float, y: float, w: int, active=False, h=26):
        pg = self.pg
        r = pg.Rect(x, y, w, h)
        pg.draw.rect(
            self.screen, self.ui["btn_on"] if active else self.ui["btn"], r, border_radius=5
        )
        pg.draw.rect(self.screen, self.ui["mid"], r, 1, border_radius=5)
        t = self.font.render(text, True, self.ui["fg"])
        self.screen.blit(t, t.get_rect(center=r.center))
        self.buttons[key] = r
        return r

    def _draw_sandbox_bar(self, y: int) -> int:
        """Sandbox rows: mode, random start, save; in edit mode, the paint colours."""
        x = self.panel_x
        for key, text, w in (
            ("mode", f"Mode: {'edit' if self.edit else 'play'}", 100),
            ("random", "Random start", 112),
            ("startn", f"Start: {self.start_n} col.", 100),
            ("savelevel", "Save level", 92),
        ):
            self._button(key, text, x, y, w, active=(key == "mode" and self.edit))
            x += w + 8
        if not self.edit:
            return y
        y += 32
        x = self.panel_x
        self._text("paint:", (x, y + 6), self.font, self.ui["mid"])
        x += 50
        pg = self.pg
        for c in range(self.level.palette):
            cc = (x + 13, y + 13)
            if c == self.paint:
                pg.draw.polygon(self.screen, self.ui["fg"], self._hex_points(cc, 15))
                pg.draw.polygon(self.screen, self.bg, self._hex_points(cc, 13))
            self._swatch(cc, 11, c)
            self.buttons[f"paint{c}"] = pg.Rect(x, y, 26, 26)
            x += 30
        self._text("click: paint   right-click: lock", (x + 10, y + 7), self.small, self.ui["mid"])
        y += 32
        x = self.panel_x
        self._text("terrain:", (x, y + 6), self.font, self.ui["mid"])
        x += 62
        for key, text, w in (
            ("tool:none", "Blocks", 64),
            ("tool:wall", "Wall", 52),
            ("tool:fix", "Fixed", 56),
            ("tool:erase", "Erase", 56),
            ("brush", f"Brush r={self.brush}", 92),
        ):
            act = key.startswith("tool:") and (key[5:] if self.tool else "none") == (
                self.tool or "none"
            )
            self._button(key, text, x, y, w, active=act)
            x += w + 6
        nh, nf = int(self.hole.sum()), int(self.fixed.sum())
        self._text(f"{nh} wall  {nf} fixed", (x + 4, y + 7), self.small, self.ui["mid"])
        return y

    def _draw_menu(self) -> None:
        """Palette dropdown, drawn last (on top): the palettes of [palette] menu, each with
        its worst Delta E for this level's number of colours (smallest pairwise difference,
        worst of normal vision / deuteranopia / protanopia) and its colours."""
        if not self.menu_open:
            return
        pg = self.pg
        if getattr(self, "_menu", None) is None:
            self._menu = menu_names(self.cfg, self.level.palette)
        r0 = self.buttons["palette"]
        w = 330
        y = r0.bottom + 2
        for name, de in self._menu:
            r = pg.Rect(r0.x, y, w, 30)
            sel = name == self.palette_name
            pg.draw.rect(self.screen, self.ui["btn_on"] if sel else self.ui["btn"], r)
            pg.draw.rect(self.screen, self.ui["mid"], r, 1)
            self._text(name, (r.x + 8, r.y + 8), self.small)
            self._text(f"dE {de:2.0f}", (r.x + 120, r.y + 8), self.small, self.ui["mid"])
            sw = palette_rgb(self.cfg, self.level.palette, name)
            dk = ui_colours(self.cfg, name, self.level.palette)["dark"]
            for j, c in enumerate(sw[:8]):
                cc = (r.x + 180 + j * 24, r.centery)
                if j == dk:
                    pg.draw.polygon(self.screen, self.ui["fg"], self._hex_points(cc, 11))
                pg.draw.polygon(self.screen, c, self._hex_points(cc, 9))
            self.buttons[f"pal:{name}"] = r
            y += 30

    def _swatch(self, c: tuple[float, float], rad: float, colour: int) -> None:
        """Hex of a cell colour on the interface; the darkest colour (= the background) gets
        a light contour so that it stays visible."""
        pg = self.pg
        if colour == self.ui["dark"]:
            pg.draw.polygon(self.screen, self.ui["fg"], self._hex_points(c, rad + 2))
        pg.draw.polygon(self.screen, self.pal[colour], self._hex_points(c, rad))

    @staticmethod
    def _hex_points(c: tuple[float, float], rad: float) -> list[tuple[float, float]]:
        """Pointy-top hexagon around c."""
        return [
            (
                c[0] + rad * math.cos(math.radians(60 * k - 90)),
                c[1] + rad * math.sin(math.radians(60 * k - 90)),
            )
            for k in range(6)
        ]

    def _block_layout(self, y: int) -> tuple[dict, list[tuple[int, int, int]]]:
        """Centre of every block, and (reaction index, row top y, row height).

        Blocks sit like cells on the grid: an edge is two horizontal neighbours, a vertex
        is three hexes around their shared corner, clockwise 0 (top), 1 (lower right),
        2 (lower left)."""
        cents: dict[Block, tuple[float, float]] = {}
        rows = []
        rb = self.rb
        w = SQRT3 * rb * GAP  # centre spacing of neighbours
        x_in = self.panel_x + EQ_DX + rb
        x_out = x_in + 2 * w + 60
        for i, r in enumerate(self.sim.reactions.reactions):
            m = ARITY[r.shape]
            h = int(2 * rb + (1.5 * rb * GAP if r.shape == "vertex" else 0))
            cy0 = y + rb
            for side, xs in (("in", x_in), ("out", x_out)):
                for p in range(m):
                    if r.shape == "vertex":
                        dx, dy = [(w / 2, 0), (w, 1.5 * rb * GAP), (0, 1.5 * rb * GAP)][p]
                    else:
                        dx, dy = p * w, 0
                    cents[(i, side, p)] = (xs + dx, cy0 + dy)
            rows.append((i, y, h))
            y += h + int(self.row_gap)
        return cents, rows

    def _draw_glyph(self, kind: str, c: tuple[float, float], col) -> None:
        pg = self.pg
        s = self.rb * 0.42
        x, y = c
        if kind == "circle":
            pg.draw.circle(self.screen, col, c, s)
        elif kind == "ring":
            pg.draw.circle(self.screen, col, c, s * 1.1, 3)
        elif kind == "triangle":
            tri = [(x, y - s), (x + s, y + s * 0.8), (x - s, y + s * 0.8)]
            pg.draw.polygon(self.screen, col, tri)
        elif kind == "square":
            pg.draw.rect(self.screen, col, pg.Rect(x - s * 0.8, y - s * 0.8, 1.6 * s, 1.6 * s))
        elif kind == "diamond":
            pg.draw.polygon(
                self.screen, col, [(x, y - s * 1.2), (x + s, y), (x, y + s * 1.2), (x - s, y)]
            )
        elif kind == "cross":
            pg.draw.line(self.screen, col, (x - s, y), (x + s, y), 3)
            pg.draw.line(self.screen, col, (x, y - s), (x, y + s), 3)
        elif kind == "x":
            pg.draw.line(self.screen, col, (x - s, y - s), (x + s, y + s), 3)
            pg.draw.line(self.screen, col, (x - s, y + s), (x + s, y - s), 3)
        elif kind == "bar":
            pg.draw.rect(self.screen, col, pg.Rect(x - s, y - s * 0.3, 2 * s, 0.6 * s))
        else:  # colour index as text
            t = self.small.render(kind, True, col)
            self.screen.blit(t, t.get_rect(center=c))

    def _draw_block(self, b: Block, c: tuple[float, float]) -> None:
        pg = self.pg
        rs = self.sim.reactions
        col = rs.get(b)
        if b == self.selected:  # white ring with a gap: visible on any block colour
            pg.draw.polygon(self.screen, self.ui["fg"], self._hex_points(c, self.rb + 7))
            pg.draw.polygon(self.screen, self.bg, self._hex_points(c, self.rb + 4))
        locked = b in rs.locked
        if locked:  # a fixed block sits in a grey frame, like a tile set in the panel
            pg.draw.polygon(self.screen, self.ui["mid"], self._hex_points(c, self.rb + 3))
        self._swatch(c, self.rb, col)
        if locked:
            pg.draw.polygon(self.screen, DARK, self._hex_points(c, self.rb), 2)
        if self.show_glyphs:
            self._draw_glyph(self.glyphs[col], c, DARK)

    def _fit_rows(self, y: int) -> None:
        """Shrink the equation rows (block radius, gap, meter) when all the slots do not fit
        between y and the chart; never below half size."""
        reacts = self.sim.reactions.reactions
        nat = sum(2 * RB + (1.5 * RB * GAP if r.shape == "vertex" else 0) + 10 for r in reacts)
        n_lines = len(self.level.goals) + len(self.level.fail)
        ch = self.cfg["chart"]
        below = ch["height"] + ch["size_height"] + 26 + 22 * n_lines + 34
        avail = self.screen.get_height() - y - below
        s = min(1.0, max(0.5, avail / nat)) if nat > 0 else 1.0
        self.rb, self.row_gap, self.row_scale = RB * s, max(4.0, 10.0 * s), s

    def _draw_equations(self, y: int) -> int:
        self._fit_rows(y)
        cents, rows = self._block_layout(y)
        self.block_centres = cents
        s = self.row_scale
        w = SQRT3 * self.rb * GAP
        for i, ry, h in rows:
            r = self.sim.reactions.reactions[i]
            lv = float(self.lamps[i])
            cy = ry + h // 2
            self._draw_meter(self.panel_x + 2, cy, float(self.needles[i]), s)
            ax = self.panel_x + EQ_DX + 2 * self.rb + w + 8
            kmax = max(q.rate for q in self.sim.reactions.reactions) or 1.0
            self._draw_arrow(ax, cy, r.rate / kmax)
            for b_, c in cents.items():
                if b_[0] == i:
                    self._draw_block(b_, c)
            xi = self.panel_x + EQ_DX + 4 * w + 90
            if self.sandbox and self.edit:
                bh = int(min(26, 2 * self.rb + self.row_gap - 2))
                lab = SHAPE_LABEL[(r.shape, r.chiral)]
                self._button(f"shape{i}", lab, xi, cy - bh // 2, 76, h=bh)
                k = "off" if r.rate == 0 else f"k={r.rate:g}"
                self._button(f"rate{i}", k, xi + 82, cy - bh // 2, 58, h=bh)
                continue
            info = f"k={r.rate:g}  {lv:.3f}"
            if r.shape == "vertex":
                info += "  cw only" if r.chiral else "  either hand"
            self._text(info, (xi, cy - 7), self.small, self.ui["mid"])
        return rows[-1][1] + rows[-1][2] + int(1.4 * self.row_gap) if rows else y

    def _draw_arrow(self, ax: float, cy: float, frac: float) -> None:
        """Reaction arrow, 42 px; the head is the same for every reaction. Rate k / k_max:
        "width" style: line width 1-5 px; "grey" style: three greys, lighter = faster.
        A switched-off reaction (k = 0) gets a dotted dark line."""
        pg = self.pg
        tip = ax + 42
        if frac <= 0:
            for x in range(int(ax), int(tip), 6):
                pg.draw.line(self.screen, self.ui["line"], (x, cy), (x + 2, cy), 2)
            return
        if self.arrow_style == "grey":
            f = 1.0 if frac > 0.75 else 0.6 if frac > 0.4 else 0.3
            bg, fg = self.ui["bg"], self.ui["fg"]
            col = tuple(int(b + f * (t - b)) for b, t in zip(bg, fg, strict=True))
            width = 3
        else:
            col = self.ui["fg"]
            width = 1 + round(4 * frac)
        pg.draw.line(self.screen, col, (ax, cy), (tip - 8, cy), width)
        pg.draw.polygon(self.screen, col, [(tip, cy), (tip - 11, cy - 6), (tip - 11, cy + 6)])

    def _draw_meter(self, x: float, cy: float, f: float, s: float = 1.0) -> None:
        """Small VU-style dial centred on height cy, 52 x 30 px times s: log firing rate,
        one tick per decade, needle at fraction f of the scale."""
        pg = self.pg
        lc = self.cfg["lamps"]
        r = pg.Rect(x, cy - 15 * s, 52 * s, 30 * s)
        face, ink = self.ui["fg"], self.ui["bg"]
        pg.draw.rect(self.screen, face, r, border_radius=4)
        px, py = r.centerx, r.bottom - 3 * s
        rad = 22.0 * s
        a0, a1 = math.radians(150), math.radians(30)  # left and right ends of the scale
        decades = round(math.log10(lc["hi"] / lc["lo"]))
        for k in range(decades + 1):
            a = a0 + (a1 - a0) * k / decades
            ca, sa = math.cos(a), math.sin(a)
            pg.draw.line(
                self.screen,
                ink,
                (px + 0.80 * rad * ca, py - 0.80 * rad * sa),
                (px + rad * ca, py - rad * sa),
                1,
            )
        pts = [
            (
                px + rad * math.cos(a0 + (a1 - a0) * t / 12),
                py - rad * math.sin(a0 + (a1 - a0) * t / 12),
            )
            for t in range(13)
        ]
        pg.draw.aalines(self.screen, ink, False, pts)
        a = a0 + (a1 - a0) * f
        tip = (px + (rad - 1) * math.cos(a), py - (rad - 1) * math.sin(a))
        pg.draw.line(self.screen, self.ui["accent"], (px, py), tip, 2)
        pg.draw.circle(self.screen, ink, (px, py), 3)

    def _draw_chart(self, y: int) -> None:
        pg = self.pg
        x = self.panel_x
        w = self.screen.get_width() - x - 15
        h = self.cfg["chart"]["height"]
        rect = pg.Rect(x, y, w, h)
        pg.draw.rect(self.screen, self.ui["chart"], rect)
        win = self.cfg["chart"]["window"]
        t1 = max(self.sim.t, win)
        t0 = t1 - win

        def px(t: float, v: float) -> tuple[float, float]:
            return x + (t - t0) / win * w, y + h - v * h

        overlay = pg.Surface(rect.size, pg.SRCALPHA)
        for goal in self.level.goals:
            if goal["type"] == "hold":
                for c, (lo, hi) in enumerate(goal["bands"]):
                    band = pg.Rect(0, h - hi * h, w, (hi - lo) * h)
                    pg.draw.rect(overlay, (*self.pal[c], 45), band)
            elif goal["type"] == "reach":
                for c, tg in enumerate(goal["target"]):
                    lo, hi = tg - goal["tol"], tg + goal["tol"]
                    pg.draw.rect(
                        overlay, (*self.pal[c], 60), pg.Rect(0, h - hi * h, w, (hi - lo) * h)
                    )
            elif goal["type"] == "extinct":
                pg.draw.rect(overlay, (*self.pal[goal["color"]], 160), pg.Rect(0, h - 4, w, 4))
        self.screen.blit(overlay, rect)
        for v in (0.25, 0.5, 0.75):
            pg.draw.line(self.screen, self.ui["line"], px(t0, v), px(t1, v))
        ts = np.array(self.hist_t)
        xs = np.array(self.hist_x)
        sel = ts >= t0
        if sel.sum() >= 2:
            for c in range(xs.shape[1]):
                pts = [px(t, v) for t, v in zip(ts[sel], xs[sel, c], strict=True)]
                pg.draw.lines(self.screen, self.pal[c], False, pts, 2)
        self._text("fractions (free cells) vs time", (x + 4, y + 3), self.small, self.ui["fg"])
        yy = self._draw_size_chart(x, y + h + 6, w, t0, t1) + 10
        for k, goal in enumerate(self.level.goals):
            pr = self.tracker.progress(self.sim.t)[k]
            yy += self._draw_segments(self._goal_segments(goal, f"[{pr:4.0%}]  goal:"), x, yy)
        for f in self.level.fail:
            yy += self._draw_segments(
                self._goal_segments(f, "fail if:", fail=True), x, yy, self.ui["mid"]
            )
        help_ = "click 2 blocks to swap | wheel zoom, drag pan, 0 reset | spc 1-4 R S"
        if self.sandbox and self.edit:
            help_ = "edit: paint blocks, right-click lock, shape/k, terrain tools | Mode: play"
        self._text(help_, (x, self.screen.get_height() - 22), self.small, self.ui["mid"])

    def _draw_size_chart(self, x: float, y: float, w: float, t0: float, t1: float) -> float:
        """Patch size S on a log scale from 1 to the number of world cells, with size goal
        bands; returns the bottom y."""
        pg = self.pg
        h = self.cfg["chart"]["size_height"]
        rect = pg.Rect(x, y, w, h)
        pg.draw.rect(self.screen, self.ui["chart"], rect)
        top = math.log10(max(10.0, len(self.sim.open_idx)))
        win = t1 - t0

        def py(v: float) -> float:
            return y + h - min(1.0, max(0.0, math.log10(max(v, 1.0)) / top)) * h

        overlay = pg.Surface(rect.size, pg.SRCALPHA)
        for g in self.level.goals + self.level.fail:
            if g["type"] == "size":
                lo, hi = g["range"]
                c = g.get("color")
                col = self.ui["fg"] if c is None else self.pal[c]
                band = pg.Rect(0, py(hi) - y, w, py(lo) - py(hi))
                pg.draw.rect(overlay, (*col, 50), band)
        self.screen.blit(overlay, rect)
        for k in range(1, int(top) + 1):
            pg.draw.line(self.screen, self.ui["line"], (x, py(10**k)), (x + w, py(10**k)))
            self._text(f"{10**k:g}", (x + w - 40, py(10**k) + 1), self.small, self.ui["line"])
        ts = np.array(self.hist_st)
        sel = ts >= t0
        if sel.sum() >= 2:
            idx = np.flatnonzero(sel)
            xs = [x + (ts[i] - t0) / win * w for i in idx]
            for c in range(self.level.palette):
                pts = [(xx, py(self.hist_s[i][1][c])) for xx, i in zip(xs, idx, strict=True)]
                pg.draw.lines(self.screen, self.pal[c], False, pts, 1)
            pts = [(xx, py(self.hist_s[i][0])) for xx, i in zip(xs, idx, strict=True)]
            pg.draw.lines(self.screen, self.ui["fg"], False, pts, 2)
        self._text(
            f"patch size S (cells, log): {self.obs['size']:.1f}",
            (x + 4, y + 3),
            self.small,
            self.ui["fg"],
        )
        return y + h

    def _goal_segments(self, g: dict, prefix: str, fail: bool = False) -> list:
        """Goal as text pieces and colour swatches (ints), e.g. ["goal: make", 2, "coral",
        "extinct"]."""
        nm = self.cnames
        if g["type"] == "extinct":
            c = g["color"]
            return [prefix, *([] if fail else ["make"]), c, nm[c], "extinct"]
        if g["type"] == "hold":
            seg: list = [prefix, "hold"]
            for c, (lo, hi) in enumerate(g["bands"]):
                if lo > 0 or hi < 1:
                    seg += [c, f"{lo:g}-{hi:g}"]
            return seg + [f"for t = {g['duration']:g}"]
        if g["type"] == "reach":
            seg = [prefix, "reach"]
            for c, v in enumerate(g["target"]):
                seg += [c, f"{v:g}"]
            return seg + [f"(+/- {g['tol']:g}) for t = {g['duration']:g}"]
        if g["type"] == "size":
            lo, hi = g["range"]
            c = g.get("color")
            seg = [prefix, "patch size", *([] if c is None else [c, nm[c]])]
            rng = f">= {lo:g}" if hi >= len(self.sim.open_idx) else f"{lo:g}-{hi:g}"
            seg += [f"{rng} cells"]
            if g.get("duration"):
                seg += [f"for t = {g['duration']:g}"]
            return seg
        return [prefix, json.dumps(g)]

    def _draw_segments(self, segs: list, x: float, y: float, col=None) -> int:
        """Draw text pieces separated by spaces; an int is a hex swatch of that colour."""
        for s_ in segs:
            if isinstance(s_, int):
                self._swatch((x + 8, y + 7), 8, s_)
                x += 20
            else:
                surf = self.font.render(s_, True, col or self.ui["fg"])
                self.screen.blit(surf, (x, y))
                x += surf.get_width() + 6
        return 22

    def _draw_status(self) -> None:
        st = self.tracker.status
        if st == "running":
            return
        msg, col = ("GOAL REACHED", (120, 230, 140)) if st == "won" else ("FAILED", (240, 110, 90))
        s = self.big.render(f"{msg}  (R to restart)", True, col)
        r = s.get_rect(center=self.grid_rect.center)
        self.pg.draw.rect(self.screen, DARK, r.inflate(30, 20), border_radius=8)
        self.screen.blit(s, r)

    # ---- input --------------------------------------------------------------------------------
    def click(self, pos, button: int = 1) -> None:
        if self.menu_open:
            for key, r in self.buttons.items():
                if key.startswith("pal:") and r.collidepoint(pos):
                    self.set_palette(key[4:])
                    break
            self.menu_open = False
            return
        for key, r in self.buttons.items():
            if not key.startswith("pal:") and r.collidepoint(pos):
                self.button(key, -1 if button == 3 else 1)
                return
        if self.grid_rect.collidepoint(pos):
            if self.sandbox and self.edit and self.tool:
                self.painting = button
                self.paint_terrain(pos, erase=button == 3)
                return
            self.dragging = button == 1
            return
        hit = None
        for b, c in getattr(self, "block_centres", {}).items():
            if (pos[0] - c[0]) ** 2 + (pos[1] - c[1]) ** 2 <= (self.rb * SQRT3 / 2) ** 2:
                hit = b
                break
        if hit is None:
            self.selected = None
            return
        if self.sandbox and self.edit:
            if button == 3:
                self.toggle_lock(hit)
            else:
                self.edit_block(hit, self.paint)
            return
        if button != 1:
            return
        if self.paused:
            self._notify("paused: press Play to switch blocks")
            return
        if hit in self.sim.reactions.locked:
            self._notify("this block is locked")
            return
        if self.selected is None:
            self.selected = hit
        elif hit == self.selected:
            self.selected = None
        else:
            self.swap(self.selected, hit)
            self.selected = None

    # ---- sandbox edits ------------------------------------------------------------------------
    def _edited(self, what: str, **kw) -> None:
        self.sim.rebuild()
        self.selected = None
        self.session.append({"event": "edit", "what": what, "t": self.sim.t, **kw})

    def edit_block(self, b: Block, c: int) -> None:
        self.sim.reactions._set(b, c)
        self._edited("paint", block=b, colour=c)

    def toggle_lock(self, b: Block) -> None:
        lk = self.sim.reactions.locked
        lk.symmetric_difference_update({b})
        self._edited("lock", block=b, locked=b in lk)

    def cycle_shape(self, i: int, d: int) -> None:
        """Next (d = 1) or previous shape of slot i, keeping its colours where possible;
        the slot's locks are cleared."""
        rs = self.sim.reactions
        r = rs.reactions[i]
        k = (SHAPE_CYCLE.index((r.shape, r.chiral)) + d) % len(SHAPE_CYCLE)
        shape, chiral = SHAPE_CYCLE[k]
        m = ARITY[shape]

        def fit(cols: list[int]) -> list[int]:
            return (cols + [cols[-1]] * 3)[:m]

        rs.reactions[i] = Reaction(shape, fit(r.inp), fit(r.out), r.rate, chiral)
        rs.locked = {b for b in rs.locked if b[0] != i}
        self._edited("shape", slot=i, shape=shape, chiral=chiral)

    def cycle_rate(self, i: int, d: int) -> None:
        rates = self.cfg["sandbox"]["rates"]
        r = self.sim.reactions.reactions[i]
        k = min(range(len(rates)), key=lambda j: abs(rates[j] - r.rate))
        r.rate = float(rates[(k + d) % len(rates)])
        self._edited("rate", slot=i, rate=r.rate)

    def random_start(self) -> None:
        """New uniform random mosaic of colours 0 .. start_n - 1 (new seed), current reactions
        kept; time restarts."""
        self.seed += 1
        rng = np.random.default_rng(self.seed)
        self.start_cols = rng.integers(0, self.start_n, self.sim0.graph.n_cells).tolist()
        self.reset()

    def save_level(self, folder: Path | None = None) -> Path:
        """Write the current reactions (slots with k = 0 dropped) and locks as a level."""
        rs = self.sim.reactions
        keep = [i for i, r in enumerate(rs.reactions) if r.rate > 0]
        new_index = {i: j for j, i in enumerate(keep)}
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        n = self.level.palette
        d = {
            "id": f"S{stamp}",
            "title": "sandbox",
            "grid": self.level.grid,
            "palette": n,
            "reactions": [
                {
                    "shape": r.shape,
                    "in": r.inp,
                    "out": r.out,
                    "rate": r.rate,
                    **({"chiral": r.chiral} if r.shape == "vertex" else {}),
                }
                for r in (rs.reactions[i] for i in keep)
            ],
            "locked": sorted([new_index[b[0]], b[1], b[2]] for b in rs.locked if b[0] in new_index),
            "start": {
                "seed": self.seed,
                "fractions": [round(1 / self.start_n, 6)] * self.start_n
                + [0.0] * (n - self.start_n),
                "burn_in_sweeps": 300,
            },
            "goals": [],
            "fail": [],
            "meta": {"from": "sandbox", "source": str(self.level.path or "")},
        }
        ter = terrain_to_json(self.sim0.graph, self.hole, self.fixed, self.fixed_col)
        if ter:
            d["terrain"] = ter
        folder = folder or ROOT / "levels" / "sandbox"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{d['id']}.json"
        k = 1
        while path.exists():  # never overwrite (two saves in the same second)
            d["id"] = f"S{stamp}-{k}"
            path = folder / f"{d['id']}.json"
            k += 1
        path.write_text(json.dumps(d, indent=1))
        self._notify(f"saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        print(f"saved level {path}", flush=True)
        return path

    def set_palette(self, name: str) -> None:
        self.palette_name = name
        self.pal = palette_rgb(self.cfg, self.level.palette, name)
        self.cnames = colour_names(self.cfg, self.level.palette, name)
        self.ui = ui_colours(self.cfg, name, self.level.palette)
        self.bg = self.ui["bg"]

    def _notify(self, msg: str) -> None:
        self.notice = msg
        self.notice_until = time.time() + 2.0

    def button(self, key: str, d: int = 1) -> None:
        if key.startswith("shape"):
            self.cycle_shape(int(key[5:]), d)
        elif key.startswith("rate"):
            self.cycle_rate(int(key[4:]), d)
        elif key.startswith("paint"):
            self.paint = int(key[5:])
        elif key.startswith("tool:"):
            self.tool = None if key == "tool:none" else key[5:]
        elif key == "brush":
            radii = self.cfg["terrain"]["brush_radii"]
            k = radii.index(self.brush) if self.brush in radii else 0
            self.brush = radii[(k + d) % len(radii)]
        elif key == "mode":
            self.edit = not self.edit
            self.selected = None
        elif key == "random":
            self.random_start()
        elif key == "startn":  # left click: one more colour, right click: one fewer
            self.start_n = (self.start_n - 1 + d) % self.level.palette + 1
            self.random_start()
        elif key == "savelevel":
            self.save_level()
        elif key == "arrows":
            self.arrow_style = "grey" if self.arrow_style == "width" else "width"
        elif key == "pause":
            self.paused = not self.paused
            if self.paused:
                self.selected = None
        elif key.startswith("speed"):
            self.speed_i = int(key[5:])
        elif key == "restart":
            self.reset()
        elif key == "palette":
            self.menu_open = not self.menu_open
        elif key == "glyphs":
            self.show_glyphs = not self.show_glyphs

    def save(self, screenshot: bool = False) -> Path:
        if self.run_dir is None:
            from runlog import new_run_dir

            self.run_dir = new_run_dir(
                "viewer",
                self.level.id,
                {"level": str(self.level.path), "seed": self.seed, "shuffled": self.shuffled},
            )
        (self.run_dir / "session.json").write_text(json.dumps(self.session, indent=1))
        if screenshot:
            p = self.run_dir / f"screen_{self.sim.sweeps:06d}_{self.restarts}.png"
            self.pg.image.save(self.screen, str(p))
            print(f"saved {p}", flush=True)
            return p
        return self.run_dir

    def _smoke_sandbox(self) -> None:
        """Headless checks of the sandbox edits (run once by --smoke)."""
        rs = self.sim.reactions
        self.draw()
        self.click(self.buttons["paint2"].center)
        b = (0, "out", 0)
        self.click(self.block_centres[b])
        assert rs.get(b) == 2 and self.paint == 2
        self.click(self.block_centres[b], 3)
        assert b in rs.locked
        self.click(self.buttons["shape0"].center)  # edge -> vertex cw: lock cleared
        assert rs.reactions[0].shape == "vertex" and b not in rs.locked
        self.click(self.buttons["shape0"].center, 3)  # back to edge
        self.click(self.buttons["rate3"].center)  # off -> first positive rate
        assert rs.reactions[3].rate > 0
        self.click(self.block_centres[(0, "in", 1)], 3)
        p = self.save_level(self.save())  # into the run folder, not levels/
        lv = load_level(p)
        assert len(lv.reactions) == sum(r.rate > 0 for r in rs.reactions)
        assert lv.locked == [[0, "in", 1]]
        # terrain: a wall line and a fixed blob, saved and reloaded
        self.click(self.buttons["tool:wall"].center)
        self.brush = 1
        cx, cy = self.grid_rect.center
        for k in range(-120, 121, 6):
            self.click((cx + k, cy - 80))
        self.painting = 0
        self.click(self.buttons["paint1"].center)
        self.click(self.buttons["tool:fix"].center)
        self.click((cx - 60, cy + 90))
        self.painting = 0
        self.apply_terrain()
        assert self.sim.hole.any() and self.sim.fixed.any()
        fx = np.flatnonzero(self.sim.fixed)
        assert (self.sim.colours[fx] == 1).all()
        self.sim.sweep(3)
        assert (self.sim.colours[fx] == 1).all()
        p2 = self.save_level(self.run_dir)
        lv2 = load_level(p2)
        assert len(lv2.terrain["holes"]) == int(self.hole.sum())
        assert len(lv2.terrain["fixed"]) == int(self.fixed.sum())
        self.click(self.buttons["tool:none"].center)
        assert self.tool is None
        self.random_start()
        assert (self.sim.colours[fx] == 1).all() and self.sim.hole.any()
        print(f"smoke: sandbox edits OK ({p.name}), terrain OK ({p2.name})", flush=True)

    def loop(self, smoke_frames: int = 0) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        frame = 0
        running = True
        while running:
            dt = clock.tick(self.cfg["window"]["fps"]) / 1000.0
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.MOUSEBUTTONDOWN and ev.button in (1, 3):
                    self.click(ev.pos, ev.button)
                elif ev.type == pg.MOUSEBUTTONUP and ev.button in (1, 3):
                    self.dragging = False
                    self.painting = 0
                elif ev.type == pg.MOUSEMOTION and self.painting:
                    self.paint_terrain(ev.pos, erase=self.painting == 3)
                elif ev.type == pg.MOUSEMOTION and self.dragging:
                    self.pan(*ev.rel)
                elif ev.type == pg.MOUSEWHEEL:
                    mp = pg.mouse.get_pos()
                    if self.grid_rect.collidepoint(mp) and ev.y:
                        self.zoom(self.cfg["grid"]["zoom_step"] ** (1 if ev.y > 0 else -1), mp)
                elif ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_ESCAPE:
                        running = False
                    elif ev.key == pg.K_SPACE:
                        self.button("pause")
                    elif pg.K_1 <= ev.key <= pg.K_9:
                        if ev.key - pg.K_1 < len(self.cfg["speed"]["multipliers"]):
                            self.button(f"speed{ev.key - pg.K_1}")
                    elif ev.key in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS):
                        self.zoom(self.cfg["grid"]["zoom_step"], pg.mouse.get_pos())
                    elif ev.key in (pg.K_MINUS, pg.K_KP_MINUS):
                        self.zoom(1 / self.cfg["grid"]["zoom_step"], pg.mouse.get_pos())
                    elif ev.key in (pg.K_0, pg.K_KP0):
                        self.zoom_reset()
                    elif ev.key == pg.K_r:
                        self.button("restart")
                    elif ev.key == pg.K_s:
                        self.save(screenshot=True)
            if smoke_frames:
                dt = 1.0 / self.cfg["window"]["fps"]
                if frame == 1:
                    self.speed_i = len(self.cfg["speed"]["multipliers"]) - 1
                    self.zoom(4.0, (300, 300))
                    self.pan(37, -12)
                if self.sandbox and frame == 3:
                    self._smoke_sandbox()
                if self.sandbox and frame == smoke_frames // 2 - 2:
                    self.button("mode")  # play mode for the swap checks below
                if frame == smoke_frames // 2 - 2:
                    # paused: a block click must not select or swap anything
                    before = (
                        self.sim.reactions.multiset(),
                        [(r.inp[:], r.out[:]) for r in self.sim.reactions.reactions],
                    )
                    self.click(self.buttons["pause"].center)
                    self.draw()
                    blk = next(b for b in self.block_centres if b not in self.sim.reactions.locked)
                    self.click(self.block_centres[blk])
                    assert self.selected is None and self.paused
                    after = (
                        self.sim.reactions.multiset(),
                        [(r.inp[:], r.out[:]) for r in self.sim.reactions.reactions],
                    )
                    assert before == after
                    self.click(self.buttons["pause"].center)
                    assert not self.paused
                    print("smoke: pause blocks switching OK", flush=True)
                if frame == smoke_frames // 2:
                    free = [
                        b for b in self.sim.reactions.blocks() if b not in self.sim.reactions.locked
                    ]
                    self.click(self.block_centres[free[0]])
                    self.draw()
                    self.click(self.block_centres[free[-1]])
                if frame >= smoke_frames:
                    self.click(self.block_centres[free[0]])  # leave one block selected
                    self.draw()
                    self.save(screenshot=True)
                    running = False
            self.apply_terrain()
            self.step(dt)
            self.move_needles(dt)
            self.draw()
            pg.display.flip()
            frame += 1
        if any(e["event"] == "swap" for e in self.session) or self.run_dir:
            print(f"session saved to {self.save()}", flush=True)


def sandbox_level(cfg: dict, colours: int | None, start: int | None = None) -> Level:
    """Fresh sandbox: the [sandbox] defaults of viewer.toml; uniform random start with the
    first `start` colours only ([sandbox] start_colours by default)."""
    sb = cfg["sandbox"]
    n = colours or sb["colours"]
    m = min(n, start or sb.get("start_colours", n))
    return Level(
        id="sandbox",
        grid=dict(sb["grid"]),
        palette=n,
        reactions=[dict(r) for r in sb["reactions"]],
        locked=[],
        start={"seed": 1, "fractions": [1 / m] * m + [0.0] * (n - m), "burn_in_sweeps": 0},
        goals=[],
        title="design tool",
    )


def pad_slots(level: Level, cfg: dict, colours: int | None) -> Level:
    """Sandbox on a level: at least [sandbox] slots reactions (extra ones off, k = 0);
    --colours can widen the palette."""
    level.palette = max(level.palette, colours or 0)
    off = {"shape": "cell", "in": [0], "out": [0], "rate": 0.0}
    level.reactions = level.reactions + [off] * (cfg["sandbox"]["slots"] - len(level.reactions))
    return level


def _mult_label(m: float) -> str:
    return f"x1/{round(1 / m)}" if m < 1 else f"x{m:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("level", type=Path, nargs="?", help="level JSON (optional with --sandbox)")
    ap.add_argument("--sandbox", action="store_true", help="design tool: edit the reactions")
    ap.add_argument("--colours", type=int, default=None, help="sandbox: number of colours")
    ap.add_argument(
        "--start", type=int, default=None, help="sandbox: random start uses colours 0..START-1"
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shuffled", action="store_true", help="well-mixed control")
    ap.add_argument("--n", type=int, default=None, help="override the grid size")
    ap.add_argument("--palette", default=None, help="palette name from toy/viewer.toml")
    ap.add_argument("--world", default=None, help="torus:N | open:N | hextorus:S | hexagon:S")
    ap.add_argument("--smoke", type=int, default=0, help="headless: run N frames, save PNG")
    a = ap.parse_args()
    if a.smoke:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    pygame.init()
    cfg = load_cfg()
    if a.level is None and not a.sandbox:
        ap.error("give a level, or --sandbox")
    level = load_level(a.level) if a.level else sandbox_level(cfg, a.colours, a.start)
    if a.sandbox:
        level = pad_slots(level, cfg, a.colours)
    if a.n:
        level.grid = {"type": "hex", "n": a.n}  # rectangular torus N x N
    if a.world:
        kind, size = a.world.split(":")
        level.grid = {
            "torus": {"type": "hex", "n": int(size)},
            "open": {"type": "hex", "n": int(size), "periodic": False},
            "hextorus": {"type": "hexagon", "side": int(size)},
            "hexagon": {"type": "hexagon", "side": int(size), "periodic": False},
        }[kind]
    if a.palette:
        if a.palette not in cfg["palettes"]:
            ap.error(f"unknown palette {a.palette!r}; choose from {list(cfg['palettes'])}")
        cfg["palette"]["default"] = a.palette
    v = Viewer(level, cfg, a.seed, a.shuffled, sandbox=a.sandbox)
    v.loop(a.smoke)
    pygame.quit()


if __name__ == "__main__":
    main()
