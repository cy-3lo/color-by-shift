# Mosaic Shift: Game Off 2026 project brief

Working title "Mosaic Shift" (not cleared, see Open questions). This file is the single brief for the whole jam, from the Python twin to the submitted Godot game. Claude Code reads it at the start of every session. It replaces all earlier briefs (sand frame, boat race).

## How to work with this file

- The user, Chris, is a fluid-mechanics professor and a solo part-time developer. He judges the dynamics by watching and playing. Make the physics visible and every parameter easy to change.
- Work only on the phase marked CURRENT in Status. Do not start the next phase without the user's go.
- At the end of each task: run the tests, then report what passes, every assumption you made that is not in this file, and anything here you think is wrong.
- Never change an entry in Decisions on your own; propose the change and wait.
- At the end of each session, append one dated line to Status.
- Do not add features that are not in the current phase. When in doubt, ask.

## Status

- CURRENT: Phase 2 (solver and levels; started 2026-11-19 with the mean-field census and grid checks, go from Chris)
- 2026-11-18: brief written; nothing built yet.
- 2026-11-18: Phase 0 done: repo celoy/mosaic (MIT), .venv + pytest, Godot 4.7.2 web benchmark ≈ 890 sweeps/s at 64 × 64 in headless Chrome (docs/phase0_benchmark.md); grid size 64 × 64 proposed. Waiting for go on Phase 1.
- 2026-11-18: Phase 1 built (grid 64 × 64 confirmed by Chris): mosaic_model, tests 1–7 pass, science checks 1–4 run (docs/phase1_notes.md), pygame viewer, test levels T01–T03. Gate B pending: Chris plays the viewer.
- 2026-11-18: Chris accepted the rate rule and per-attempt shuffled control. Added empty colour (T04 May–Leonard), viewer at 1 time unit/s with partial sweeps, Pause/speed/Restart buttons, swaps only while running, arrow thickness ∝ k, --n grid override. Spirals visible at n = 128, not at 64.
- 2026-11-18: Chris: colours only, no special empty colour; no limit on the number of colours for now; no hex borders. Active-set vs rejection benchmark (Python + GDScript): rejection kept (active set ~3× slower at 15–30 % firing, faster only below ~5–10 % active).
- 2026-11-18: World shapes added (hex torus, hexagonal torus, open versions; grid spec "type": "hexagon", "side", "periodic"). Shape changes voter coarsening speed, not RPS/May–Leonard/chiral outcomes. Viewer: Chris's palette + palette menu (okabe_ito, tol_bright, greyscale), symbols off by default, hex equation blocks, periodic worlds tiled to the screen edge, thin chevron arrows.
- 2026-11-18: T05 (relative handedness, 4 colours): flipping one chiral reaction moves x0 from 0.35 to 0.02, mean field blind to it (0.46 for both). With 3 colours handedness is pinned by topology and does not matter. teal_coral palette added. Next: Gate B (Chris plays T02, T04, T05).
- 2026-11-18: Gate B (Chris played): live switching not interesting; lamps unreadable; levels too easy (move the target colour to the inputs, take it out of the products): kill criterion met, redesign before building levels. Chosen: hexagonal torus of side 74 (16 207 cells, about 128²), teal_coral default palette, 5 colours, keep chirality. Viewer: log-scale needle meters instead of lamps, x1/4 speed, zoom and pan, goal colours named with a swatch, framed locked blocks. Chris proposes a sandbox (random start, write up to 4 equations, place locks). Direction pending.
- 2026-11-18: Chris: sandbox first, as a design tool (toy/viewer.py --sandbox: paint blocks, right-click locks, shape and k per slot, 4 slots, random start, save to levels/sandbox/); game format decided after exploring with it. Arrows: same head, rate by line width or by three greys (toggle). Rate 0 = reaction off.
- 2026-11-18: 14 palettes from Chris added with colour names; the five with more than 5 colours (split_ui) give their darkest colour to the background and their lightest to the text; interface tones follow the palette. Separation table in the session report.
- 2026-11-18: Chris: long palettes no longer give colours to the interface; a level with n colours uses the n best-separated (largest smallest Delta E incl. against the background), list best_subset in toy/viewer.toml.
- 2026-11-18: 11 more palettes from Chris (duplicates of stormy_almond and teal_coral skipped); strawberry_slate (7 colours) in best_subset.
- 2026-11-18: palette menu ranked by worst Delta E (pairs and background, worst of normal / deuteranopia / protanopia) for the level's number of colours: best 10 in the first column.
- 2026-11-18: Chris: default palette eggshell_peach; menu = eggshell_peach + top 4 (dusk_coral, grape_sunflower, teal_coral, candy). Interface from the palette: darkest colour = background (its blocks get a light contour), lightest = text and meter face, most chromatic = needle; chart on a mid grey tinted with the darkest colour.
- 2026-11-18: sandbox has 10 reaction slots (window 1000 px high; equation rows shrink when they do not fit).
- 2026-11-18: sandbox random start uses only colours 0..n-1 (Start button: left +1, right -1; --start; [sandbox] start_colours = 3); saved levels keep these fractions.
- 2026-11-18: Chris's 5-colour tournament (S20261118-184837, each colour beats two): grid time-averages match the mean-field point A x* = 0 within 0.015; x*_i depends only on the rates of reactions not involving i; grid coexistence is stable (sd 0.02), shuffled runs lose 2 colours in 5 of 6 runs by t = 400 (tools/tournament.py).
- 2026-11-18: Chris's 4-cycle with chiral ally swaps (S20261118-190742): the swaps have no measurable effect (level, swaps off and mirrored agree within seed scatter, 4 seeds, t = 600); grid locks x0 = x2, x1 = x3, alliance balance drifts by +-0.015; shuffled lost an alliance in 1 of 4 runs (tools/variants.py).
- 2026-11-18: Chris's spiral attempt (S20261118-191619: zero-sum RPS + majority vertex invasions + chiral rotation): mean-field focus unstable (2 +- 6.9i, stronger than T04) but patches only a few cells wide, since mobility is missing; adding edge exchange at k = 10 grows the patterns to grid size (S20261118-191619_mobile, to be watched). Sandbox k button now goes up to 10.
- 2026-11-19: terrain added (Chris: reject the whole channel, fractions over free cells, no emitters): holes (walls) and fixed cells in model, mean field (exact in shuffled mode, test passes at 2 %), level JSON, sandbox painting tools (Wall / Fixed / Erase, brush). Patch-size observable S, size goal (proposed) and a log-scale size chart under the fractions. Science check terrain: voter with fixed columns and a wall matches the discrete harmonic solution (max error 0.006, se 0.004). Demo levels levels/sandbox/X01–X03 (tools/terrain_levels.py).
- 2026-11-19: Chris: goals and size will change; wants drawn (non-random) starts like a maze. Added map starts, contact goal, level colours, open rectangular grids (no torus for mazes). X04 maze (tools/maze.py): red source, yellow/blue corridors, one swap toggles which corridors red crosses; win red-green contact, fail red-black. Bots (20 seeds): never switch, switch at once and dithering fail (dither every 4: 1/20 wins); two timed switches win 20/20 with 1.5 s reaction, 11/12 at 8 s, 6/12 at 12 s. Waiting for Chris's playtest.
- 2026-11-19: chirality made to matter by walls: chiral vertex reactions that need fixed stone at one corner follow walls in one direction (vine (R,W,S)->(R,R,S) grows with the wall on its right; one swap of its input blocks mirrors it; one-cell gaps are two-way and link wall loops). X05 vine level (tools/chiral_levels.py): bots win 29/30 with mirror then mirror back when red touches the island (1.5 s reaction; 18/20 at 3 s, 7/20 at 5 s); no switch, one switch and dithering fail. Walker (R,W,S)->(W,R,S) needs two swaps to reverse (one swap = pause) and random-walks in gaps; it cannot win on the X05 map, so no walker level yet. Contact goals drawn as locked reactions (red|green -> star, red|black -> skull, Chris's idea); contact "all" = touch every target. Fixed-cell dots only where the colour also occurs on free cells.
- 2026-11-19: Chris: X04/X05 reduce the game to one front, one reaction and a well-timed click; wanted: few walls (a few % of cells), isotropic spread, many reactions, collective dynamics. Added zone goals (colour fraction in a map zone, chars or rect), per-character random map fills, zone outlines, tools/screen.py (every static arrangement x seeds), tools/living_bots.py. X09 tug of war (contact process, two sources, swap of death slots sets drift): feedback player wins 10/10 with about 6 switches (delay 2 or 5), statics fail, dithering 1/10. Contact-process threshold on the hex lattice measured: death ~1.95 at birth 0.5 per edge (mean field 3.0). X07 bloom and X08 rooms built but not working yet (screens: X07 blue home lost or red never reaches G; X08 rooms give the same fractions as the whole grid).
- 2026-11-19: Chris: X09 is still about timing; the game should be designing reactions. Proposed rule: all k equal, swaps only input<->input or output<->output (conserves the input and output colour multisets and the net production per colour; not the mean-field equilibrium, counterexample A>B,B>C,C>A vs A>C,B>B,C>A). tools/classify.py: mean-field census, 3 colours, k = 1, all sets up to relabelling (runs/classify/n3_20261119-111328): cell-only = linear Markov chain (exact on the grid, no sustained oscillation); pure invasions = zero-sum replicator (conserved H = sum p ln x: RPS neutral); no limit cycle or heteroclinic cycle found with 3 colours (apparent cycles at t = 150 were slow foci); swap orbits are rich: 3 edge reactions up to 7 distinct behaviours per orbit, 1 cell + 2 edge up to 7.
- 2026-11-19: census with 4 colours (runs/classify/n4_20261119-122352, 12 starts; classify.py now parallel, starts seeded per family): 3 reactions are too few for 4 colours (edge3: 15 808 of 18 394 sets are neutral families, often a colour left inert); no limit cycle, the only neutral set is RPS with an idle colour; up to 7 behaviours per orbit. Grid check (tools/gridcheck.py, runs/gridcheck/n3_20261119-111328_20261119-122518; 112 sets, game grid, t = 400): 65 differ from mean field. 37 jam: reactions that need an A|A pair leave isolated A cells and the grid freezes (no firing at all) with 8-16 % of a colour that mean field kills; 21 mixtures shifted (0.05-0.6); 4 change the majority (bistable 00>01;02>22;12>00: mean field -> colour 1, grid -> colour 2 in both seeds); 3 lose a colour; RPS: same mean, oscillation replaced by patches. Cell-only sets all agree (exact).
- 2026-11-19: orbit check (tools/orbitcheck.py, runs/orbitcheck/n3_20261119-111328_20261119-153623): every arrangement of the 22 richest 3-colour orbits (incl. RPS), 1313 arrangements, mean field vs grid (2 seeds, t = 400, x0 0.5/0.3/0.2). Grid end states: alive 247 (count-changing firing >= 0.01 per cell per time), trickle 43, exchanges only 60, jam 798, one colour 104, seeds disagree 61. Alive: median |x_grid - x_mf| 0.05, 97 above 0.1; 34 keep a colour mean field kills (e.g. 00>01;01>02;22>12: isolated 0 cells freeze at 0.16 and keep catalysing a living 1/2 mosaic); mean field names the wrong clear majority in only 4. Richest orbits give 3-4 alive majorities (e.g. edge3 in 000112 out 001112: 35 alive, majority 1/2/0 in 24/6/4). Earlier runs 132937 and 145205 had wrong alive labels (identity reactions counted, presence thresholds), superseded.
- 2026-11-19: Pillow added (optional dependency "tools"). Painting regrowth prototype (tools/painting.py; Hokusai Great Wave, Met scan via Wikimedia Commons, public domain, assets/paintings/SOURCES.md): open hex 150 x 116, k-means palette, random fixed tiles, scrambled free cells, all k = 1. Accuracy on free cells (runs/painting/great_wave_*) saturates at 0.55-0.64, close to the nearest-tile reference (0.56-0.60); voter + majority (vertex (A,A,B)->(A,A,A)) best at 2-5 % tiles, majority alone best at 10 %, alone at 5 % it erases minorities; voter alone stays noisy. Regrown images keep the large masses (dark left and bottom, grey band) but not the wave's curl and claws: random tiles carry too little information.
- 2026-11-19: Chris: go on dissolve as main painting mode, with hermetic outlines and the movie played in reverse. Cuts added (model, level JSON, viewer and raster drawing, 5 tests; 78 pass). tools/painting.py: --mode dissolve records frames at geometric times and writes reverse GIFs paced linearly or by amount of change (arclength in accuracy and interface density); outlines = the fraction q of edges with the largest colour difference in the blurred painting; target now sampled after a 0.5-cell blur (interface density 0.37 -> 0.20). Great Wave, 4 colours, t = 300 (runs/painting/great_wave_dissolve_20261119-175257): cells still at the painting's colour 0.40 voter, 0.54 majority, 0.53 both, 0.18 cycle; with outlines q = 0.12 (6199 cuts, leaky: only 3 closed regions of 10+ cells) 0.41, 0.80, 0.68, 0.20. Regrowth with 5 % tiles (great_wave_regrow_20261119-175348): voter + majority 0.70 without outlines, 0.78 with, nearest-tile reference 0.72. Waiting for Chris to watch the GIFs.
- 2026-11-19: Chris: the painting prototype is not a game yet; chosen: contemplative toy as the floor, matching tested by bot first. Match test (tools/matchtest.py, runs/matchtest/n3_20261119-111328_20261119-153623_20261119-180554; 3 block sets, 430 arrangements, signature = fractions, unlike-pair shares, patch size per colour, activity): 18-35 distinguishable living classes per block set, texture needed for 74-95 % of targets, but a greedy single-swap bot needs 14-24 tries against 13-47 blind (factor 3 not reached) and a mean-field player needs 2-3 tries in two sets: matching on a uniform mosaic not kept. Chris: in a painting texture differs by zone (sky, sea, foam). Zone test (tools/zonetest.py, runs/zonetest/open_20261119-181822; sky one colour, foam speckle of 1 and 2, sea mix of 0, 1, 2, zones touching, all 9139 sets of 3 edge reactions): 34 sets (0.37 %) keep sky still, foam still or frozen and sea alive; 25-27 of them hold on a 120 x 120 world at t = 400 (2 seeds); 29 of 637 block sets contain a solution (median 17 arrangements, 8 % solutions, 4 distinct outcomes); 4 % of single-swap neighbours of a solution are solutions. All solutions act through colour 0, present only in the sea. Waiting for Chris's verdict on zone puzzles.
- 2026-11-19: zone solutions on the Great Wave (3 colours, no outlines, t = 300; runs/painting/great_wave_dissolve_20261119-184802; painting.py accepts explicit sets such as "00>01;01>00;01>10"): 00>01;01>00;01>10 keeps sky, foam and the cream of the sea untouched and turns the dark water into a living blue-grey speckle, stationary from t = 10 (0.80 of cells at the painting's colour, wave silhouette intact); 00>01;01>02;02>01 leaves a grey sea with isolated blue dots; 00>01;01>10;11>12 thins everything to a dotted ghost (0.65); voter for comparison 0.45 and falling. Waiting for Chris to watch.
- 2026-11-19: Chris: Mondrian-style sealed flat zones judged not fun (independent conditions, no contact dynamics); toy first. Toy in the viewer: tools/painting.py --export-level writes levels/paintings/great_wave.json (drawn start, painting palette, 3 colours, living-sea set 00>01;01>00;01>10); viewer records the run (nearly geometric frame times, [rewind] in toy/viewer.toml) and a Rewind button (key B) plays it backwards in 12 s, paced by amount of change; pace_frames in mosaic_model.observables. 80 tests pass, headless smoke OK. Waiting for Chris to play: .venv/bin/python toy/viewer.py --sandbox levels/paintings/great_wave.json
- 2026-11-19: tools/met_fetch.py: search and download from the Met collection API (public-domain objects only, source appended to assets/paintings/SOURCES.md, no overwrite). The v1 search endpoint was retired on 2026-10-01; the tool uses v1.1/search and v1/objects. Tested: search "hokusai wave", download of object 45434 to a scratch folder. Chris: no clear objective beyond watching the movie; game question open.
- 2026-11-19: Chris: new direction, a colouring game: simple pictures made of zones sealed by fixed edges, each zone with a given start and a target look (flat colour or pattern of 2-3 colours), one rule set for the whole picture, Zach-like loop (write rules, play, read the failure, retry); free writing of rules in a few slots, the level gives the start (changes the Decision on swapping: to be reworded by Chris). Rule table (tools/ruletable.py, runs/ruletable/table_20261119-195955): 9919 sets of 1-3 edge reactions, 3 colours, sealed 30 x 30 patch, 4 canonical starts x 2 seeds, 13 start kinds by relabelling; 91 % of outcomes reliable; 42-63 distinct looks per start kind; 4578 distinct outcome vectors. Random puzzles from hidden 3-reaction sets: 3 zones give a median of 13 solutions (5 or fewer in 27 %, 41 % need all 3 reactions, median 151 near misses); 4 zones a median of 9 (5 or fewer in 36 %, 53 % need 3 reactions).
- 2026-11-20: Chris: change of plan, build the contemplative toy (pictures to choose from incl. random and regular pavings; rule presets plus the player's own reactions, all k = 1; export of the reversed movie as a GIF). Built toy/painting_toy.py (pygame, headless smoke --smoke N), toy/pictures.py, toy/toy.toml: 7 Met Open Access paintings (Hokusai, Hiroshige, van Gogh x2, Seurat, Vermeer, Klimt; tools/met_fetch.py, sources in assets/paintings/SOURCES.md) + 5 generated pictures (De Stijl-style composition generated, not copied; regular 3-colouring; rings; stripes; random); 2-8 colours by k-means++ (4 restarts; plain k-means had merged clusters); 10 presets (voter, smoothing, majority, shuffle, rps, spirals = May-Leonard with the lightest colour as empty space, living sea, ghost, swirl, none) + 6 own reaction slots, all applied live; Rewind and Save GIF (reversed, paced by amount of change, exact palette, gifs/ git-ignored). 92 tests pass. Waiting for Chris to play: .venv/bin/python toy/painting_toy.py
- 2026-11-20: Chris: toy panel redesigned, no text except the title: title with left/right arrows; 5 colours fixed, shown as a list, with a choice of colours per painting (present = k-means; contrast and accent = the 5 most different colours, largest smallest Delta E among 20 Lab clusters holding at least 2 % or 0.4 % of the picture; smallest Delta E goes from 9-16 to 15-33 on the seven paintings); a preset's reactions are listed as hex blocks (edge = two cells, vertex = three cells around a corner) and edited in place (click block = colour, click arrow = shape, right click = remove, + = add); every picture starts paused; four icon buttons (play/pause, fast-forward x16, backward, save GIF). Regular paving with 5 colours is (q + 2r) mod 5 (no equal neighbours). Crop option for scans with a frame. tools/palette_sheet.py, tools/gallery.py: 195 public-domain Met candidates with their three colour choices in runs/gallery/gallery_20261120-1530.html. 93 tests pass. Waiting for Chris's picks.
- 2026-11-20: Chris picked 9 paintings and their colours from the gallery (Hokusai Great Wave, Modigliani Lola de Valence, El Greco Portrait of an Old Man, Klimt Mada Primavesi, Memling Portinari diptych, Vermeer Study of a Young Woman, Cezanne Card Players, van Gogh Self-Portrait with a Straw Hat, Homer Cannon Rock; Met Open Access, assets/paintings/SOURCES.md) plus the 5 generated pictures. Toy: one colour set per picture (palette per entry in toy/toy.toml), random buttons for picture and rules, the preset's name shown with arrows like the title, floppy-disc save icon, every vertex reaction chiral (majority at half its former rate). Art Institute of Chicago: its API lists public-domain Kandinsky (5), Mondrian (3), Matisse (4), Hartley (9), Monet (33) paintings, but its image server answers 403 to the script (it asks for a contact address in the request): not downloaded. 93 tests pass. Waiting for Chris to play.
- 2026-11-20: more modern sources from Chris: National Gallery of Art open data (194 open-access paintings from 1900 on, small images fetched: Modigliani, Kirchner, Jawlensky, Gris, Marc, Macke, Delaunay, Vallotton, Vuillard...; no Kandinsky, Mondrian, Klee, Matisse, Picasso); Getty (51 paintings from 1890 on, nothing more modern in style than the Met, left out); rawpixel and the Chicago image server refuse scripts (403). tools/pick_page.py + pick_page.html: click-to-pick page of the 146 landscape candidates (width/height 1.15-1.65; Met 84, NGA 62), original + present / contrast 5 %, 2 %, 0.4 % colour sets, picks stored in the artifact's database (https://claude.ai/artifact/HGr5DFgpgyTF78pUb3h5iW, collection picks). tools/gallery.py shows the museum. Waiting for Chris's picks.
- 2026-11-20: Chris made 44 picks on the pick page (18 Met, 26 National Gallery of Art; landscape formats). tools/import_picks.py: 42 new paintings copied to assets/paintings with sources (SOURCES.md) and entries in toy/toy.toml; the two already present (Card Players, Portinari) take their new colour choice. A picture can carry its own contrast share (two picks at 5 %). Scan edges cropped on 11 Met oils. The toy now has 51 paintings + 5 generated pictures; assets/paintings 8.4 MB. 94 tests pass, headless smoke loads all 56. Waiting for Chris to play.

## Deadline and jam

- GitHub Game Off 2026, theme SHIFT. Submissions close 1 December 2026 at 21:37 UTC (13:37 PST). Target submission: Monday 30 November, evening (Paris time).
- The source must be in a public GitHub repository. Any engine is allowed.
- Rated categories: overall, gameplay, graphics, audio, innovation, theme interpretation.
- Before submitting, the user checks the jam page for any rule on AI-assisted code.

## The game in one page

Pitch. A living mosaic of coloured cells is held in balance by a few reactions. You cannot paint the cells. You can only shift colours between the reactions, live, and watch the mosaic tip toward a goal: wipe out a colour, hold a balance, reach a mixture.

The world. A large grid of cells (hexagonal first, foam later if time), with edges between neighbouring cells and vertices where three cells meet. Each cell has a colour from a palette of 3–4.

Reactions (given by the level: number, shape and rate are fixed):
- Cell reaction: A → B.
- Edge reaction: A|B → C|D, on any pair of neighbouring cells, either orientation.
- Vertex reaction: (A, B, C) → (D, E, F) around a vertex, read clockwise. Chirality matters: the mirror arrangement does not match unless the level says the reaction is achiral.

The only verb. Every colour in every reaction equation is a block. The player swaps any two unlocked blocks, anywhere among the equations, at any time, while the mosaic runs. The multiset of blocks is conserved, so every move is a permutation: a shift of colours between reactions. This is the theme.

The dynamics. Random sequential updating: pick a cell, edge or vertex at random, look up the reactions matching its colours, apply one according to the rates. Large grids make the large-scale behaviour nearly deterministic; the texture stays alive.

Start. Each level starts from a stationary state of its initial arrangement: fronts wandering, spirals turning, population fractions steady. The first switch perturbs a living system.

Goals only, no score. Goal types: extinct (a colour disappears), hold (all fractions stay within bands for a duration), reach (fractions within a tolerance of a target, held for a few seconds). Some levels add a failure condition (for example, a colour going extinct during a hold).

Legibility. A lamp beside each equation glows with that reaction's recent firing rate. Optional overlay: a brief flash on the grid where reactions fire. A population chart shows the fractions over time with the goal bands drawn on it. The theme is named in the tutorial ("shift a colour between reactions").

Why it is hard: effects are delayed and spread over the whole grid; spatial dynamics defeat well-mixed intuition (rock-paper-scissors spirals, voter-model coarsening); the blocks are conserved, so fixing one reaction changes another; timing matters, because the same switch made at a different moment meets a different landscape.

Fun hypothesis (falsifiable in Phases 1–2):
- In most levels a solution needs at least two switches with a wait between them, so timing matters.
- In at least a third of levels the well-mixed (mean-field) prediction is wrong, so the grid is not decoration.
- A correct solution fails in fewer than 10 % of random seeds on the full grid.
- A dithering bot (switching back and forth quickly) does not solve most levels trivially. If it does, add a switch cooldown.

Kill criterion, at Gate B (Sunday 22 November): if most test levels fall to a single switch, or the mean-field equations predict every outcome, the spatial game is decoration. Redesign the reaction sets before building levels.

Out of scope for the jam: painting cells, editing reaction shapes or rates, scores and leaderboards, level editor, topological changes of the foam, touch controls, voice, 3D.

## Model

Graph. Cells, edges and vertices with full adjacency: cell → neighbouring cells, edge → its two cells, vertex → its three cells in clockwise order. Hexagonal grid generated in code; foam grids loaded from JSON (generated offline in Python).

Element counts (derived from Euler's formula for trivalent tilings, boundaries aside): per cell, about 3 edges and 2 vertices.

Update rule (one attempt):
1. Pick an element uniformly among all cells, edges and vertices that can host at least one current reaction shape (skip element types with no reaction).
2. Read its colours. Use a lookup table indexed by the colour tuple to find matching reactions: for edges, both orientations; for vertices, the three cyclic rotations of the clockwise triple (and the mirror only for achiral reactions).
3. Every (reaction, orientation) that matches is a channel firing at rate k (so A|A → … fires at 2k; an achiral vertex reaction is the reaction plus its mirror). Fire with probability (sum of matching channel rates) / K, where K is the largest total rate of any element state, and pick the channel in proportion to its rate (accepted 2026-11-18).
4. Rebuild the lookup tables whenever the player switches blocks (cheap).

Time: one sweep = one attempt per eligible element; the simulation clock advances by 1 / K per sweep (1 / (K · n_elig) per attempt, so partial sweeps are allowed). Speed control sets sim time units per wall second (viewer ×1 = 1 time unit per second).

Mean-field equations (derived, product measure: each cell independently has colour c with probability x_c): each reaction contributes to dx/dt its rate times the number of orientations (1 cell, 2 edge, 3 chiral vertex, 6 achiral vertex) times the product of the input fractions times the net change in colour counts, with attempt weights 1 : 3 : 2 for cell, edge and vertex reactions. Generate these equations automatically from any reaction set; they are used by the solver and by tests.

Shuffled control: a well-mixed mode in which every attempt draws the element's cells at random (equivalent to a permutation before every attempt; accepted 2026-11-18, because shuffling once per sweep leaves errors of about 5 % against mean field). It must agree with the mean-field equations on a large grid; the difference between normal and shuffled runs is the spatial effect.

Assumptions and regime: discrete-time random sequential updates approximate a continuous-time Markov chain when each sweep changes a small fraction of elements; mean-field is exact only for the shuffled control; extinction on a finite grid is always eventually possible, so "hold" goals have finite durations.

## Python twin (Phases 1 and 2)

Package `mosaic_model` (numpy; ask before adding numba if the loops are too slow). Holds the graph, reactions, lookup tables, update, shuffled control, mean-field generator, goals, and a JSON level loader shared with Godot.

Minimal viewer (pygame-ce, Phase 1): coloured hexes, reaction equations with lamps, click two blocks to swap them, pause and speed keys, population chart with goal bands. Enough for Chris to feel the game at Gate B.

Science checks (headless, CSV outputs):
1. Voter model (A|B → A|A and B|A → B|B): the density of unlike edges decays slowly; plot it against time.
2. Rock-paper-scissors (cyclic invasion) plus a swap reaction A|B → B|A at varying rate: coexistence with spirals at low swap rate, loss of species at high rate. Qualitative check against Reichenbach, Mobilia and Frey (2007).
3. Chiral vertex rotation (A, B, C) → (C, A, B) with and without its mirror: compare patterns.
4. Normal against shuffled runs for each test level: the size of the spatial effect.

Solver (tools/solver.py, Phase 2): enumerate block arrangements reachable by swaps (deduplicated); screen each with the mean-field equations; validate promising ones with spatial runs over several seeds; search short switch sequences with waits (for example, up to 3 switches with waits from a small set). Report per level: solvable or not, smallest number of switches found, whether timing matters, mean-field verdict against spatial verdict, seed failure rate.

Bots: switch sequence bots from the solver; a dithering bot that alternates two arrangements quickly.

Tests (pytest):
1. Lookup tables agree with brute-force matching for all colour tuples.
2. A clockwise-only vertex reaction does not fire on the mirror arrangement.
3. Cell and edge counts: swap reactions conserve colour counts exactly.
4. Shuffled control matches the mean-field ODE within 2 % on a large grid.
5. Determinism with a fixed seed and switch timeline.
6. Goal detection for extinct, hold and reach.
7. Swapping blocks conserves the multiset of blocks and respects locks.

## Levels

Level file (levels/Lxx.json):

```
{
  "id": "L04",
  "grid": { "type": "hex", "n": 64 },
  "palette": 3,
  "reactions": [
    { "shape": "edge",   "in": [0, 1], "out": [0, 0], "rate": 1.0 },
    { "shape": "edge",   "in": [1, 2], "out": [1, 1], "rate": 1.0 },
    { "shape": "edge",   "in": [2, 0], "out": [2, 2], "rate": 1.0 },
    { "shape": "vertex", "in": [0, 1, 2], "out": [2, 0, 1], "rate": 0.3, "chiral": true }
  ],
  "locked": [[0, "in", 0]],
  "start": { "seed": 7, "fractions": [0.34, 0.33, 0.33], "burn_in_sweeps": 400 },
  "goals": [ { "type": "extinct", "color": 2 } ],
  "fail": [ ],
  "meta": { "min_switches_found": 2, "timing_matters": true, "mean_field_wrong": true }
}
```

Optional terrain (2026-11-19), axial cell coordinates of the level's world: `"terrain": {"holes": [[q, r], ...], "fixed": [[q, r, colour], ...]}`. A hole is a wall (not part of the world: no edges or vertices touch it). A fixed cell keeps its colour forever and is read by every reaction; a channel that would change it is rejected whole. Fractions, counts and goals use free cells only. Hermetic edges (2026-11-19): `"cuts": [[q1, r1, q2, r2], ...]` in terrain; the edge between the two neighbouring cells and the two vertices containing both are removed, so nothing reacts across the cut (an outline). Optional goal (proposed, pending Chris): `{"type": "size", "range": [lo, hi], "duration": T, "color": c}` on the patch size S = sum s² / sum s of same-colour connected patches (cells; all colours if no "color").

Drawn start (2026-11-19): `"map": {"rows": [...], "legend": {"#": "wall", ".": 0, "R": [1, "fixed"], ...}}` gives an odd-r hex block (open rectangle by default, no torus), its walls, fixed cells and start colours, replacing random fractions. Optional `"colours"` (hex) and `"names"` set the level's own palette. Goal `{"type": "contact", "colors": [a, b]}`: a cell of colour a next to one of colour b (as goal or fail).

Campaign of 8–10 levels: learn the swap on a two-reaction level; voter-model coarsening; tip a balance; rock-paper-scissors: keep all three alive; rock-paper-scissors: make one extinct; a chiral level; a timing level (switch, wait, switch back); mixed. Chris picks and orders levels from solver output.

## Godot game

- Godot 4.3 or later (confirm the user's version), GDScript only, Compatibility renderer, web export with thread support off. Flag any Godot 3 idiom (yield, export without @, onready without @, PoolByteArray...).
- Simulation in plain classes (RefCounted), no Nodes, testable headless: Graph, Reactions (with lookup tables), Sim, Goals, LevelLoader. Colours in a PackedByteArray. Scenes only display state and pass input.
- Golden tests: `godot --headless --script res://tests/run_tests.gd` checks lookup tables, chirality, conservation, goal detection, and compares population curves of fixed-seed runs with the Python twin statistically (same mean-field limit in shuffled mode; same qualitative outcome in normal mode).
- Rendering: one MultiMesh or a texture-based hex renderer; colours updated per frame from the byte array. Flat cells, thin dark borders.
- UI: reaction panel with equations drawn as colour blocks (edge: two blocks and an arrow; vertex: a triangle of three blocks with a rotation arrow), a lamp beside each, click or drag to swap two blocks, locked blocks marked. Pause, speed (×1, ×4, ×16), restart, level select. Population chart with goal bands. Optional firing-flash overlay toggle.
- Palette: no fixed limit on the number of colours (Chris, 2026-11-18: find out how many make the game interesting); colours distinguishable by colour-blind players (start from the Okabe–Ito palette, recalled; check), plus a distinct glyph on each block in the equations.
- Audio: each reaction has its own soft pluck; firing rate sets the density of plucks, so the mosaic sounds like a chord that changes with the balance. AudioStreamGenerator if it behaves in the web export, otherwise CC0 samples, credited.
- Progress: completed levels in user://progress.json.
- Performance: benchmark in Phase 0. Target: at least 4 sweeps per second at ×1 on the chosen grid in Chrome on the user's laptop (guess). If not met, shrink the grid (48 × 48, then 40 × 40) before anything else.

## Repository layout

```
mosaic/
  CLAUDE.md
  README.md              # Phase 5
  LICENSE                # chosen by the user
  model/                 # Python package mosaic_model + tests
  toy/                   # pygame-ce viewer
  tools/                 # science checks, solver, bots, foam generator, golden export
  levels/  grids/        # shared JSON (levels, precomputed foams)
  game/                  # Godot project: sim/ scenes/ ui/ audio/ tests/
  docs/                  # screenshots, GIFs for the itch page
  runs/                  # gitignored
```

## Conventions

- Python 3.11+, one virtual environment at the repo root (.venv). Dependencies: numpy, pygame-ce, pytest. Ask before adding anything (numba, matplotlib, scipy for the foam generator).
- Parameters and content in JSON or TOML, never hard-coded. One seed per run.
- Commit at the end of each task with a message naming the phase and task.

## Plan

Dates and hours are guesses. If a phase runs more than a day late, apply the cut list.

Phase 0, setup and benchmark (Wed 18 – Thu 19 Nov, about 2 h)
- Public GitHub repo, layout above, .venv, pytest running.
- Godot benchmark: random sequential updates with one edge reaction on 64 × 64 hexes, web export, sweeps per second in Chrome. Report and choose the grid size.

Phase 1, Python twin and viewer (Thu 19 – Sat 21 Nov, about 5 h)
- Model, tests 1–7, science checks 1–4, minimal pygame viewer, three test levels (voter, rock-paper-scissors with swaps, chiral).
- Gate B on Sunday 22: Chris plays the viewer. Is switching live interesting? Are the lamps and chart enough to understand what happens? Record the decision in Status.

Phase 2, solver and levels (Sun 22 – Tue 24 Nov, about 4 h)
- Solver, bots, fun-hypothesis table, 12–15 candidate levels with stationary starts. Chris picks 8–10.

Phase 3, Godot core (Tue 24 – Thu 26 Nov, about 6 h)
- Graph, reactions, sim, rendering, reaction panel with lamps and swapping, pause and speed, goals, population chart, golden tests.
- Acceptance: one level playable start to finish in a web export within the performance budget.

Phase 4, game (Fri 27 – Sat 28 Nov, about 5 h)
- Level select, all chosen levels, short tutorial on level 1, palette and glyphs, audio, progress save. Foam grids only if everything else is done.

Phase 5, freeze and submit (Sun 29 – Mon 30 Nov, about 3 h; Tue 1 Dec buffer)
- Feature freeze on Sunday 29; bug fixes only after that.
- Two or three outside players try it without help; note where they get stuck.
- README (how to play, the science in one paragraph, credits), itch page text and screenshots, web build, submission on Monday 30.

Cut list, in this order: foam grids; firing-flash overlay; AudioStreamGenerator (use samples); levels beyond six; chiral reactions (keep cell and edge only).

## Decisions

- Godot Wild Jam #99 dropped; entry for GitHub Game Off 2026, theme SHIFT.
- Sand frame, phase-space archipelago and the one-button boat race (never started) dropped in favour of the reaction mosaic.
- Reactions (number, shape, rate) are given by each level; the player can only switch colour blocks between reaction slots, live, at any time.
- Large grid, random sequential updates, start from a stationary state of the level's initial arrangement.
- A lamp beside each equation shows its firing rate.
- Goals only, no score: extinct, hold, reach.
- Hexagonal grid first; precomputed foam grids only if time remains.
- The parallel with First Rain (ecology) is accepted.

## Open questions (as tasks)

- Title: search Steam, itch.io and the web for clashes before adopting "Mosaic Shift" or any other title.
- Licence for the public repo.
- Confirm the jam's rule on AI-assisted code before submitting.
- Godot version installed.
- Grid size, from the Phase 0 benchmark.
- Switch cooldown: none by default; add one if the dithering bot trivialises levels.
- Palette and glyphs: final choice after the first playable build.

## Sources

- T. Reichenbach, M. Mobilia, E. Frey, "Mobility promotes and jeopardizes biodiversity in rock-paper-scissors games", Nature 448, 1046–1049 (2007), https://doi.org/10.1038/nature06095 (seen in search results): with cyclic invasion plus mobility, species coexist in spirals below a critical mobility and are lost above it.
- Nearest games (seen in search results): HOX, https://quintin-steiner.itch.io/hox (design rules to grow a seed into a shape); Automaton, https://mestiez.itch.io/automaton (devise a ruleset to satisfy level constraints). Both deterministic and synchronous.
- Recalled, not verified in this session: voter model coarsening in two dimensions; Kawasaki exchange dynamics; Gillespie's stochastic simulation algorithm (1976); Kolmogorov's criterion for detailed balance; the Okabe–Ito colour-blind-safe palette.
